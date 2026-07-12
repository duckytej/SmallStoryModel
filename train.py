# ─────────────────────────────────────────────────────────────────────────────
# BUG-FIX-1: Force UTF-8 on Windows BEFORE any import that touches the
#            filesystem.  Without this, datasets reads cached files with
#            cp1252 (Windows default) and crashes on byte 0x9d.
# ─────────────────────────────────────────────────────────────────────────────
import os
import sys

os.environ["PYTHONUTF8"] = "1"
os.environ["HF_DATASETS_NO_PROGRESS_BARS"] = "0"

"""
Hyperion-v3: Training script
==============================
Changes from previous version:

  SHARD-CACHE    Dataset is now tokenized once and saved as .pt shard files
                 under <cache_dir>/shards/.  On every subsequent run the
                 loader skips HuggingFace entirely and reads from disk.
                 Controlled by --cache_dir (default: "data_cache") and
                 --n_shards (default: 64 shards of ~4 K sequences each).
                 Use --regen_cache to force a fresh download + re-tokenize.

  RESUME         --resume auto-detects the latest checkpoint in out_dir and
                 restores model weights, optimizer states, and scheduler step.
                 Removes need to re-run from step 0 after a crash.

  PRESET-FIX     3050ti preset now explicitly sets use_attn_res=False
                 (cuts 16 BlockAttnResOp modules that add zero benefit at
                 this scale), ctm_ticks=3 (safe default so enabling CTM
                 later won't KeyError), and restores d_ff / attn_res_n_blocks
                 as explicit values so nothing relies on model.py defaults.

  CTM-KEY        ctm_ticks is now always present in the preset dict so
                 build_model never gets a KeyError when use_ctm=True.

  FIX-TOK-SAVE   (retained) tok.model.save() called exactly once after
                 tok.save().  Path uses Path.parent so it never collapses
                 to "".

  BUG-FIX-2      (retained) Dataset assignments unindented to correct scope.
  BUG-FIX-3      (retained) trust_remote_code=True removed.
  BUG-FIX-4      (retained) ds.features guarded for lazy streaming datasets.

Usage:
  # First run: downloads + caches shards, then trains
  python train.py --preset 3050ti --run exp01

  # Subsequent runs: loads shards from disk, no HF traffic
  python train.py --preset 3050ti --run exp01

  # Resume a crashed run from the latest checkpoint
  python train.py --preset 3050ti --run exp01 --resume

  # Force fresh shard download
  python train.py --preset 3050ti --run exp01 --regen_cache

  # Train tokenizer only
  python train.py --train_tok --preset 3050ti
"""

import argparse
import datetime
import glob
import math
import random
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    import wandb

    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from model import Hyperion, HyperionConfig, build_model, count_params, set_quant_stage
from optim import WarmupCosineScheduler, build_optimizers

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TrainLogger:
    """Timestamps every line and tees to a log file."""

    def __init__(self, out_dir: str, run: str):
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(out_dir, f"train_{run}.log")
        self._fh = open(log_path, "a", encoding="utf-8", buffering=1)
        self._start = time.time()
        self.log(f"Log file: {log_path}")

    def _ts(self) -> str:
        return datetime.datetime.now().strftime("%H:%M:%S")

    def log(self, msg: str):
        line = f"[{self._ts()}] {msg}"
        print(line, flush=True)
        self._fh.write(line + "\n")

    def separator(self, char: str = "-", width: int = 72):
        self.log(char * width)

    def close(self):
        self._fh.flush()
        self._fh.close()


_logger: "TrainLogger | None" = None


def _log(msg: str):
    if _logger:
        _logger.log(msg)
    else:
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Hardware presets
# ---------------------------------------------------------------------------

PRESETS = {
    "3050ti": dict(
        depth=8,
        aspect_ratio=48,  # d_model = 8 * 48 = 384
        head_dim=64,
        max_seq_len=1024,
        batch_size=3,
        grad_accum=6,  # halved: 2x more optimizer steps in same wall-clock time
        total_steps=12_000,  # sized to fill ~6hr; cosine curve lands cleanly
        warmup_steps=800,  # widened from 500: small d_model + tied embed/lm_head
        # + competing MTP/JEPA aux losses sharing one proj destabilizes fast
        # once Muon hits full LR; longer ramp gives the model more time to
        # absorb early orthogonalized updates before they're at full strength.
        d_ff=768,  # d_model * 2 (explicit, not default)
        window_size=128,
        # attn_res_n_blocks=2,
        use_attn_res=True,
        use_mtp=True,
        mtp_k=1,
        use_ctm=False,
        ctm_ticks=3,  # always present so enabling CTM later won't KeyError
        jepa_weight=0.05,
        use_moe=False,
        compile=False,
        dtype="bfloat16",
        vocab_size=12288,
        drop_path_rate=0.0,
        label_smoothing=0.05,
        hyp_weight=0.0,  # disabled: saves compute, negligible benefit at this scale
    ),
    "t4": dict(
        depth=12,
        aspect_ratio=64,  # d_model = 768
        head_dim=64,
        max_seq_len=1024,
        batch_size=4,
        grad_accum=8,  # effective batch ≈ 32K tokens/step
        total_steps=20_000,
        warmup_steps=1_000,
        d_ff=1024,
        window_size=256,
        attn_res_n_blocks=4,
        use_attn_res=True,
        use_mtp=True,
        mtp_k=1,
        use_ctm=True,
        ctm_ticks=3,
        jepa_weight=0.05,
        use_moe=True,
        n_experts=8,
        n_shared_experts=1,
        compile=True,
        dtype="float16",
        vocab_size=16384,
        drop_path_rate=0.05,
        label_smoothing=0.05,
        hyp_weight=0.01,
    ),
}

# Keys that belong to the training loop only, NOT forwarded to HyperionConfig
_TRAIN_ONLY_KEYS = {
    "batch_size",
    "grad_accum",
    "total_steps",
    "warmup_steps",
    "compile",
    "dtype",
}


# ---------------------------------------------------------------------------
# HuggingFace auth
# ---------------------------------------------------------------------------


def _hf_login(token: "str | None"):
    tok = token or os.environ.get("HF_TOKEN", "").strip()
    if not tok:
        _log("[hf] No HF token — private/gated datasets will fail.")
        return
    try:
        from huggingface_hub import login

        login(token=tok, add_to_git_credential=False)
        _log(f"[hf] Logged in with token: {tok[:8]}{'*' * (len(tok) - 8)}")
    except Exception as e:
        _log(f"[hf] Login failed (continuing anyway): {e}")


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------


def train_tokenizer(
    save_path: str = "tokenizer.json", vocab_size: int = 16384, n_samples: int = 50_000
) -> None:
    try:
        from datasets import interleave_datasets, load_dataset
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    except ImportError:
        sys.exit("[tokenizer] pip install tokenizers datasets")

    save_dir = Path(save_path).parent.resolve()
    save_dir.mkdir(parents=True, exist_ok=True)

    _log(f"[tokenizer] Training BPE  vocab={vocab_size}  samples={n_samples}")
    _log(f"[tokenizer] Artefacts → {save_dir}/")

    def _load(name, config=None, split="train"):
        kwargs = dict(split=split, streaming=True)
        if config:
            return load_dataset(name, config, **kwargs)
        return load_dataset(name, **kwargs)

    ds_talk = _load("HuggingFaceTB/smol-smoltalk")
    ds_story = _load("roneneldan/TinyStoriesInstruct")
    ds_py = _load("HuggingFaceTB/smollm-corpus", "python-edu")
    ds_cosmo = _load("HuggingFaceTB/smollm-corpus", "cosmopedia-v2")
    ds_web = _load("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup")

    def _text_col(ds):
        cols = list(ds.features.keys()) if ds.features else None
        return ds.map(lambda r: {"text": _get_text(r)}, remove_columns=cols)

    dataset = interleave_datasets(
        [
            _text_col(ds_talk),
            _text_col(ds_story),
            _text_col(ds_py),
            _text_col(ds_cosmo),
            _text_col(ds_web),
        ],
        probabilities=[0.20, 0.30, 0.2, 0.2, 0.10],
        seed=42,
    )

    def text_iter():
        n = 0
        for row in dataset:
            text = _get_text(row)
            if text:
                yield text
                n += 1
                if n % 5_000 == 0:
                    _log(f"[tokenizer]   ... {n:,} samples streamed")
                if n >= n_samples:
                    break
        _log(f"[tokenizer] Iterator finished at {n:,} samples")

    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<unk>", "<eos>", "<pad>", "<s>"],
        min_frequency=2,
        show_progress=True,
    )

    _log("[tokenizer] Training BPE (a few minutes)...")
    tok.train_from_iterator(text_iter(), trainer=trainer)

    tok.save(save_path)
    _log(
        f"[tokenizer] Saved tokenizer.json → {save_path}  (vocab={tok.get_vocab_size()})"
    )
    tok.model.save(str(save_dir))
    _log(f"[tokenizer] Saved vocab.json + merges.txt → {save_dir}/")

    test = "Hello world! The quick brown fox <eos>"
    enc = tok.encode(test)
    _log(f"[tokenizer] Sanity ({len(enc.ids)} tokens): {enc.tokens[:10]} ...")


def load_tokenizer(
    tok_path: str = "tokenizer.json", vocab_size: int = 16384, auto_train: bool = True
):
    if os.path.exists(tok_path):
        try:
            from tokenizers import Tokenizer as _FTok

            tok = _FTok.from_file(tok_path)
            v = tok.get_vocab_size()
            eos = tok.get_vocab().get("<eos>", 1)
            _log(f"[tokenizer] Loaded {tok_path}  vocab={v}  eos_id={eos}")
            return tok, v, eos
        except Exception as e:
            _log(f"[tokenizer] Load failed: {e}")

    if auto_train:
        _log(f"[tokenizer] {tok_path} not found — training now ...")
        try:
            train_tokenizer(tok_path, vocab_size=vocab_size)
            from tokenizers import Tokenizer as _FTok

            tok = _FTok.from_file(tok_path)
            v = tok.get_vocab_size()
            eos = tok.get_vocab().get("<eos>", 1)
            return tok, v, eos
        except Exception as e:
            _log(f"[tokenizer] Auto-train failed: {e}")

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        _log("[tokenizer] Fallback: GPT-2 (vocab=50257)")
        return tok, 50257, tok.eos_token_id or 1
    except Exception:
        _log("[tokenizer] No tokenizer — random batches will be used")
        return None, vocab_size, 1


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def _get_text(row: dict) -> str:
    return (
        row.get("text")
        or row.get("content")
        or " ".join(str(m.get("content", "")) for m in (row.get("messages") or []))
        or ""
    )


def _encode(tok, text: str) -> list:
    if tok is None:
        return []
    try:
        from tokenizers import Tokenizer as _FTok

        if isinstance(tok, _FTok):
            return tok.encode(text).ids
    except ImportError:
        pass
    return tok.encode(text)


# ---------------------------------------------------------------------------
# Shard cache helpers
# ─────────────────────────────────────────────────────────────────────────────
# Layout on disk:
#   <cache_dir>/shards/shard_000000.pt
#   <cache_dir>/shards/shard_000001.pt
#   ...
#   <cache_dir>/shards/meta.pt   ← {n_shards, seq_len, eos_id, vocab_size}
#
# Each shard .pt is a list of (x, y) pairs, where
#   x : LongTensor (seq_len,)  — input token ids
#   y : LongTensor (seq_len,)  — target token ids (with -1 at EOS boundaries)
#
# Building:
#   We stream all five HF datasets, tokenize, pack into fixed-length windows,
#   and flush to a new shard file every `seqs_per_shard` sequences.
#   Progress is printed every 10 % so you know it hasn't stalled.
#
# Loading:
#   A ShardedDataset object loads shard files on-demand (shuffled on each
#   epoch) and provides a get_batch(batch_size) method that is a drop-in
#   replacement for the old streaming make_dataset_iter.
# ---------------------------------------------------------------------------

_SHARD_META_FILE = "meta.pt"


def _shard_dir(cache_dir: str) -> str:
    return os.path.join(cache_dir, "shards")


def _shard_path(cache_dir: str, idx: int) -> str:
    return os.path.join(_shard_dir(cache_dir), f"shard_{idx:06d}.pt")


def _meta_path(cache_dir: str) -> str:
    return os.path.join(_shard_dir(cache_dir), _SHARD_META_FILE)


def _cache_valid(
    cache_dir: str, seq_len: int, eos_id: int, vocab_size: int, n_shards: int
) -> bool:
    """Return True iff meta.pt exists and matches the requested config."""
    mp = _meta_path(cache_dir)
    if not os.path.exists(mp):
        return False
    try:
        meta = torch.load(mp, weights_only=True)
        return (
            meta.get("seq_len") == seq_len
            and meta.get("eos_id") == eos_id
            and meta.get("vocab_size") == vocab_size
            and meta.get("n_shards") >= n_shards
        )
    except Exception:
        return False


def build_shard_cache(
    tok,
    seq_len: int,
    eos_id: int,
    cache_dir: str,
    n_shards: int,
    seqs_per_shard: int,
    vocab_size: int,
    seed: int = 42,
):
    """
    Stream HuggingFace datasets, tokenize, pack into (seq_len,) windows,
    and save as .pt shard files.  Called only when the cache is absent or
    --regen_cache is set.
    """
    os.makedirs(_shard_dir(cache_dir), exist_ok=True)
    _log(f"[cache] Building shard cache → {_shard_dir(cache_dir)}")
    _log(
        f"[cache] n_shards={n_shards}  seqs_per_shard={seqs_per_shard}  seq_len={seq_len}"
    )

    try:
        from datasets import interleave_datasets, load_dataset

        def _load(name, config=None):
            kwargs = dict(split="train", streaming=True)
            if config:
                return load_dataset(name, config, **kwargs)
            return load_dataset(name, **kwargs)

        ds_talk = _load("HuggingFaceTB/smol-smoltalk")
        ds_story = _load("roneneldan/TinyStoriesInstruct")
        ds_py = _load("HuggingFaceTB/smollm-corpus", "python-edu")
        ds_cosmo = _load("HuggingFaceTB/smollm-corpus", "cosmopedia-v2")
        ds_web = _load("HuggingFaceTB/smollm-corpus", "fineweb-edu-dedup")

        dataset = interleave_datasets(
            [ds_talk, ds_story, ds_py, ds_cosmo, ds_web],
            probabilities=[0.20, 0.30, 0.2, 0.2, 0.10],
            seed=seed,
        )
    except Exception as e:
        _log(f"[cache] HF datasets unavailable ({e}) — cannot build shard cache")
        return False

    total_seqs_needed = n_shards * seqs_per_shard
    buf: list = []
    shard_idx = 0
    seqs_written = 0
    current_shard: list = []

    _log(f"[cache] Need {total_seqs_needed:,} sequences total ...")

    for row in dataset:
        text = _get_text(row)
        if not text:
            continue
        ids = _encode(tok, text)
        if not ids:
            continue
        buf.extend(ids)
        buf.append(eos_id)

        # Drain buffer into sequences
        while len(buf) >= seq_len + 1:
            chunk = buf[: seq_len + 1]
            del buf[: seq_len + 1]

            x = chunk[:seq_len]
            y = chunk[1 : seq_len + 1]
            # Mask first token after each EOS in targets (no cross-doc loss)
            for t in range(seq_len):
                if x[t] == eos_id and t + 1 < seq_len:
                    y[t] = -1

            current_shard.append(
                (
                    torch.tensor(x, dtype=torch.long),
                    torch.tensor(y, dtype=torch.long),
                )
            )
            seqs_written += 1

            if len(current_shard) >= seqs_per_shard:
                torch.save(current_shard, _shard_path(cache_dir, shard_idx))
                pct = 100 * (shard_idx + 1) / n_shards
                _log(
                    f"[cache] Saved shard {shard_idx:04d}/{n_shards}  ({pct:.0f}%)  "
                    f"seqs={seqs_written:,}"
                )
                current_shard = []
                shard_idx += 1
                if shard_idx >= n_shards:
                    break

        if shard_idx >= n_shards:
            break

    # Flush partial shard (if any) — keeps the count honest
    if current_shard and shard_idx < n_shards:
        torch.save(current_shard, _shard_path(cache_dir, shard_idx))
        _log(
            f"[cache] Saved final partial shard {shard_idx}  "
            f"(seqs={len(current_shard)})"
        )
        shard_idx += 1

    # Write meta so future runs skip downloading
    torch.save(
        {
            "n_shards": shard_idx,
            "seq_len": seq_len,
            "eos_id": eos_id,
            "vocab_size": vocab_size,
        },
        _meta_path(cache_dir),
    )

    _log(
        f"[cache] Done.  {shard_idx} shards, {seqs_written:,} sequences "
        f"→ {_shard_dir(cache_dir)}/"
    )
    return True


class ShardedDataset:
    """
    Loads .pt shard files from disk and serves batches.
    Shuffles shard order at the start of each epoch; within a shard
    sequences are served in random order.

    This is a drop-in replacement for the old streaming make_dataset_iter:
      get_batch = ShardedDataset(...).get_batch
    """

    def __init__(self, cache_dir: str, batch_size: int, seq_len: int, seed: int = 42):
        sdir = _shard_dir(cache_dir)
        self.shard_files = sorted(glob.glob(os.path.join(sdir, "shard_*.pt")))
        if not self.shard_files:
            raise FileNotFoundError(f"No shard files found in {sdir}")
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.rng = random.Random(seed)

        self._shard_order: list = []
        self._shard_buf: list = []  # list of (x_tensor, y_tensor)
        self._buf_pos: int = 0
        _log(
            f"[cache] ShardedDataset: {len(self.shard_files)} shards, "
            f"batch_size={batch_size}, seq_len={seq_len}"
        )

    def _load_next_shard(self):
        if not self._shard_order:
            self._shard_order = list(self.shard_files)
            self.rng.shuffle(self._shard_order)
        path = self._shard_order.pop(0)
        self._shard_buf = torch.load(path, weights_only=True)
        self.rng.shuffle(self._shard_buf)
        self._buf_pos = 0

    def get_batch(self) -> "tuple[torch.Tensor, torch.Tensor]":
        x_list, y_list = [], []
        while len(x_list) < self.batch_size:
            if self._buf_pos >= len(self._shard_buf):
                self._load_next_shard()
            x, y = self._shard_buf[self._buf_pos]
            self._buf_pos += 1
            x_list.append(x)
            y_list.append(y)
        return torch.stack(x_list), torch.stack(y_list)


def make_dataset_iter(
    tok,
    seq_len: int,
    batch_size: int,
    eos_id: int,
    cache_dir: str = "data_cache",
    n_shards: int = 64,
    seqs_per_shard: int = 4096,
    vocab_size: int = 16384,
    regen_cache: bool = False,
    seed: int = 42,
):
    """
    Returns a get_batch() callable.

    First checks for a valid on-disk shard cache.  If found, serves from
    disk (fast, no HF traffic).  If not found (or --regen_cache), streams
    HF datasets, tokenizes, and writes shards before returning.

    Falls back to random batches only if both HF and cache are unavailable.
    """
    # ── Check / build cache ────────────────────────────────────────────────
    if regen_cache or not _cache_valid(
        cache_dir, seq_len, eos_id, vocab_size, n_shards
    ):
        _log(f"[cache] {'Regenerating' if regen_cache else 'Building'} shard cache ...")
        ok = build_shard_cache(
            tok, seq_len, eos_id, cache_dir, n_shards, seqs_per_shard, vocab_size, seed
        )
        if not ok:
            _log("[dataset] Cache build failed — falling back to random batches")

            def rand_batch():
                x = torch.randint(0, max(100, eos_id + 1), (batch_size, seq_len))
                y = torch.randint(0, max(100, eos_id + 1), (batch_size, seq_len))
                return x, y

            return rand_batch
    else:
        _log(
            f"[cache] Valid shard cache found at {_shard_dir(cache_dir)} — skipping download"
        )

    # ── Serve from disk ────────────────────────────────────────────────────
    try:
        ds = ShardedDataset(cache_dir, batch_size, seq_len, seed=seed)
        return ds.get_batch
    except Exception as e:
        _log(f"[cache] ShardedDataset failed ({e}) — falling back to random batches")

        def rand_batch():
            x = torch.randint(0, max(100, eos_id + 1), (batch_size, seq_len))
            y = torch.randint(0, max(100, eos_id + 1), (batch_size, seq_len))
            return x, y

        return rand_batch


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------


def _latest_checkpoint(out_dir: str) -> "str | None":
    """Return the path to the most recent ckpt_*.pt, or None."""
    pattern = os.path.join(out_dir, "ckpt_*.pt")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def _save(
    model,
    optimizers,
    step,
    out_dir,
    vocab_size,
    final=False,
    logger=None,
    scheduler=None,
):
    os.makedirs(out_dir, exist_ok=True)
    name = "final.pt" if final else f"ckpt_{step:06d}.pt"
    path = os.path.join(out_dir, name)

    ckpt = {
        "step": step,
        "model": model.state_dict(),
        "config": asdict(model.cfg),
        "vocab_size": vocab_size,
    }
    if not final:
        ckpt["optimizers"] = [o.state_dict() for o in optimizers]
        # BUG FIX: persist scheduler state so that a resumed run restores the
        # correct position on the warmup-cosine curve.  Without this the
        # scheduler resets to step 0 on resume and re-runs the warmup phase,
        # producing an incorrectly high or low LR for potentially thousands of steps.
        if scheduler is not None:
            ckpt["scheduler"] = scheduler.state_dict()

    torch.save(ckpt, path)
    size_mb = os.path.getsize(path) / 1e6
    _log(
        f"[ckpt] Saved {'FINAL' if final else f'step {step}'} → {path}  ({size_mb:.1f} MB)"
    )


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args):
    global _logger
    preset = PRESETS[args.preset].copy()

    if args.force_dense:
        preset["use_moe"] = False
    if args.steps > 0:
        preset["total_steps"] = args.steps

    _logger = TrainLogger(args.out_dir, args.run)
    _logger.separator("=")
    _log(f"Hyperion-v3 Training  |  run={args.run}  |  preset={args.preset}")
    _logger.separator("=")

    _hf_login(args.hf_token)

    # ── Device / dtype ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype_str = preset["dtype"]
    pt_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_str]
    use_amp = device.type == "cuda" and pt_dtype != torch.float32
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=pt_dtype)
        if use_amp
        else nullcontext()
    )
    use_scaler = pt_dtype == torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=use_scaler)

    _log(f"[device] {device}  dtype={dtype_str}  amp={use_amp}  scaler={use_scaler}")
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        _log(f"[device] GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")
        _log(f"[device] CUDA {torch.version.cuda}  PyTorch {torch.__version__}")

    # ── Tokenizer ──────────────────────────────────────────────────────────
    _logger.separator()
    tok, actual_vocab, eos_id = load_tokenizer(
        tok_path=args.tok_path,
        vocab_size=preset.get("vocab_size", args.vocab_size),
        auto_train=not args.no_auto_tok,
    )
    vocab_size = actual_vocab
    # ── Peek at Checkpoint for Config Sync ─────────────────────────────────
    # Cache the loaded checkpoint here so the resume section below can reuse
    # it without a second torch.load call.  model.load_state_dict and
    # optimizer.load_state_dict both copy tensors into existing device-resident
    # params/state, so a CPU-mapped load works fine for both.
    _resume_ckpt: Optional[dict] = None
    if args.resume:
        ckpt_path = _latest_checkpoint(args.out_dir)
        if ckpt_path is not None:
            _log(f"[resume] Peeking at {ckpt_path} to restore architecture config...")
            _resume_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            if "config" in _resume_ckpt:
                preset.update(_resume_ckpt["config"])
                _log("[resume] Overwrote command-line preset with checkpoint config.")
                # Re-apply CLI flags that must always win over the saved config.
                if args.force_dense:
                    preset["use_moe"] = False
                if args.steps > 0:
                    preset["total_steps"] = args.steps
    # ── Model ──────────────────────────────────────────────────────────────
    _logger.separator()
    model_kwargs = {k: v for k, v in preset.items() if k not in _TRAIN_ONLY_KEYS}
    model_kwargs["vocab_size"] = vocab_size
    model_kwargs["force_dense"] = args.force_dense
    model_kwargs["use_sequence_packing"] = True

    if not model_kwargs.get("use_moe", False):
        model_kwargs.pop("n_experts", None)
        model_kwargs.pop("n_shared_experts", None)

    _log("[model] Building Hyperion-v3 ...")
    _log(f"[model] Config: {model_kwargs}")

    model = build_model(
        depth=model_kwargs.pop("depth"),
        aspect_ratio=model_kwargs.pop("aspect_ratio"),
        head_dim=model_kwargs.pop("head_dim"),
        **model_kwargs,
    ).to(device)

    # Capture cfg and param count before torch.compile wraps the module;
    # compiled models may not expose arbitrary Python attributes.
    cfg = model.cfg
    n_params = count_params(model)

    if preset.get("compile", False) and hasattr(torch, "compile"):
        _log("[model] Running torch.compile() ...")
        model = torch.compile(model)
        _log("[model] torch.compile() done")

    _log(f"[model] Parameters:  {n_params / 1e6:.2f}M total")
    _log(
        f"[model] d_model={cfg.d_model}  layers={cfg.n_layers}  "
        f"heads={cfg.n_heads}Q / {cfg.n_kv_heads}KV  (GQA {cfg.n_heads // cfg.n_kv_heads}:1)"
    )
    _log(
        f"[model] vocab={vocab_size}  weight_tied=True  "
        f"moe={'ON  n_experts=' + str(getattr(cfg, 'n_experts', '?')) if cfg.use_moe else 'OFF'}"
    )
    _log(f"[model] seq_packing=True  seq_len={cfg.max_seq_len}  eos_id={eos_id}")
    _log(
        f"[model] mtp={'ON  k=' + str(getattr(cfg, 'mtp_k', '?')) if getattr(cfg, 'use_mtp', False) else 'OFF'}  "
        f"ctm={'ON' if getattr(cfg, 'use_ctm', False) else 'OFF'}  "
        f"attn_res={'ON' if cfg.use_attn_res else 'OFF'}"
    )

    if device.type == "cuda":
        _log(
            f"[model] VRAM after model init: {torch.cuda.memory_allocated() / 1e9:.2f} GB"
        )

    # ── Data ───────────────────────────────────────────────────────────────
    _logger.separator()
    _log(
        f"[data] Building dataset iterator  seq_len={preset['max_seq_len']}"
        f"  batch_size={preset['batch_size']}"
    )
    _log(
        f"[data] Cache dir: {args.cache_dir}  n_shards={args.n_shards}  "
        f"seqs_per_shard={args.seqs_per_shard}"
    )

    get_batch = make_dataset_iter(
        tok,
        seq_len=preset["max_seq_len"],
        batch_size=preset["batch_size"],
        eos_id=eos_id,
        cache_dir=args.cache_dir,
        n_shards=args.n_shards,
        seqs_per_shard=args.seqs_per_shard,
        vocab_size=vocab_size,
        regen_cache=args.regen_cache,
    )
    _log("[data] Iterator ready")

    # ── Optimizers ─────────────────────────────────────────────────────────
    _logger.separator()
    # BUG FIX: unpack the optimizers explicitly via tuple assignment from the
    # factory's return value, rather than building a list and indexing into
    # it afterward. This fails loudly (ValueError) if build_optimizers' return
    # arity ever changes, instead of silently mis-assigning muon/adamw to the
    # wrong objects.
    muon, adamw = build_optimizers(
        model,
        matrix_lr=args.matrix_lr,
        embedding_lr=args.embedding_lr,
        scalar_lr=0.05,
        weight_decay=args.weight_decay,
        d_model=cfg.d_model,
    )
    optimizers = [muon, adamw]

    total_steps = preset["total_steps"]
    warmup_steps = preset["warmup_steps"]
    scheduler = WarmupCosineScheduler(
        optimizers, warmup_steps=warmup_steps, total_steps=total_steps
    )

    # ── Resume ─────────────────────────────────────────────────────────────
    start_step = 1
    if args.resume:
        ckpt_path = _latest_checkpoint(args.out_dir)
        if ckpt_path is None:
            _log("[resume] No checkpoint found — starting from scratch")
        else:
            _log(f"[resume] Loading checkpoint: {ckpt_path}")
            # Reuse the checkpoint already loaded during the config-peek pass.
            # model.load_state_dict / optimizer.load_state_dict copy tensors
            # into existing device-resident params/state, so a CPU-mapped load
            # works correctly for both; no second torch.load needed.
            ckpt = (
                _resume_ckpt
                if _resume_ckpt is not None
                else torch.load(ckpt_path, map_location="cpu", weights_only=False)
            )
            model.load_state_dict(ckpt["model"])
            if "optimizers" in ckpt:
                for opt, sd in zip(optimizers, ckpt["optimizers"]):
                    opt.load_state_dict(sd)
            # BUG FIX: restore scheduler via load_state_dict when available.
            # The old approach called scheduler.step() in a loop up to ckpt["step"]
            # times, which (a) was O(N) and slow for large step counts, and
            # (b) only worked if the scheduler happened to still be at _step=0 —
            # any earlier partial initialisation would land at the wrong position.
            # load_state_dict() restores the exact step counter and immediately
            # re-applies the correct LR to all param groups.
            if "scheduler" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler"])
                # BUG FIX: load_state_dict() above just overwrote total_steps/
                # warmup_steps with whatever was saved in THIS checkpoint (i.e.
                # the horizon the run was originally launched with). If --steps
                # was used to change the horizon for this resume (e.g. shrinking
                # 12k -> 10k), that override gets silently clobbered and the
                # cosine decay keeps targeting the old total_steps. Re-apply the
                # horizon computed from the current preset/CLI args (already
                # correctly set above) and re-derive the LR at the current step
                # against the NEW curve, while keeping the _base_lrs snapshot
                # (so we resume at the LR the original curve had reached, not
                # jump back to step 0's LR).
                if scheduler.total_steps != total_steps or scheduler.warmup_steps != warmup_steps:
                    _log(
                        f"[resume] Overriding checkpoint horizon "
                        f"(warmup={scheduler.warmup_steps}->{warmup_steps}, "
                        f"total={scheduler.total_steps}->{total_steps})"
                    )
                    scheduler.total_steps = total_steps
                    scheduler.warmup_steps = warmup_steps
                    frac = scheduler.current_frac
                    for opt, base_lrs in zip(scheduler.optimizers, scheduler._base_lrs):
                        for g, base_lr in zip(opt.param_groups, base_lrs):
                            g["lr"] = base_lr * frac
                _log(
                    f"[resume] Scheduler restored to step {scheduler.current_step} "
                    f"(frac={scheduler.current_frac:.4f})"
                )
            else:
                # Legacy checkpoint without scheduler state: fast-forward by
                # calling step() in a loop.  Slow but correct for old ckpts.
                _log(
                    "[resume] No scheduler state in checkpoint — fast-forwarding (legacy)"
                )
                for _ in range(ckpt["step"]):
                    scheduler.step()
            start_step = ckpt["step"] + 1
            _log(
                f"[resume] Resumed from step {ckpt['step']}  "
                f"(continuing from step {start_step})"
            )

            # BUG FIX: the data iterator (ShardedDataset) has NO persisted
            # position — it reseeds random.Random(42) fresh every run and
            # starts at shard order index 0 / buffer position 0, regardless
            # of how many steps were already trained. Without this, resuming
            # silently REPLAYS the exact same batches already seen in steps
            # 1..ckpt["step"] instead of continuing into new data. Because the
            # shuffle is deterministic (fixed seed), we can correctly skip
            # ahead by replaying get_batch() the same number of times it was
            # already called, with no model forward/backward — just tensor
            # indexing / shard loads, so this is cheap relative to training.
            n_skip = ckpt["step"] * preset["grad_accum"]
            _log(
                f"[resume] Fast-forwarding dataset iterator by {n_skip} "
                f"batches to match step {ckpt['step']}..."
            )
            for _ in range(n_skip):
                get_batch()
            _log("[resume] Dataset iterator fast-forward complete")

    n_muon = sum(p.numel() for g in muon.param_groups for p in g["params"])
    n_adamw = sum(p.numel() for g in adamw.param_groups for p in g["params"])
    _log(f"[optim] Muon  params={n_muon / 1e6:.2f}M  init_lr={args.matrix_lr}")
    _log(f"[optim] AdamW params={n_adamw / 1e6:.2f}M  init_lr={args.embedding_lr}")
    _log(f"[optim] Warmup={warmup_steps} steps  Cosine decay over {total_steps} steps")

    # ── WandB ──────────────────────────────────────────────────────────────
    _logger.separator()
    if HAS_WANDB and args.run != "dummy":
        wandb.init(
            project="hyperion",
            name=args.run,
            config={
                **preset,
                "n_params": n_params,
                "vocab_size": vocab_size,
                "force_dense": args.force_dense,
            },
            resume="allow" if args.resume else None,
        )
        _log(f"[wandb] Run: {wandb.run.url}")
    else:
        _log("[wandb] Disabled (not installed or run=dummy)")

    # ── Training loop ──────────────────────────────────────────────────────
    _logger.separator("=")
    grad_accum = preset["grad_accum"]
    eff_batch_k = preset["batch_size"] * grad_accum * preset["max_seq_len"] // 1000

    _log(f"[train] Starting from step {start_step}")
    _log(
        f"[train] total_steps={total_steps}  grad_accum={grad_accum}  "
        f"effective_batch≈{eff_batch_k}K tokens/step"
    )
    _log(f"[train] Checkpoints every 2000 steps → {args.out_dir}/")
    _logger.separator("=")

    _log(
        f"{'step':>8} | {'loss':>9} | {'bpb':>7} | {'z_loss':>7} | "
        f"{'mtp_loss':>8} | {'jepa_loss':>9} | {'lr_muon':>10} | "
        f"{'lr_adamw':>10} | {'tok/s':>8} | {'vram_gb':>7}"
    )
    _logger.separator()

    model.train()
    t0 = time.time()
    t_last_log = t0
    tokens_seen = 0
    ema_loss = None
    ema_alpha = 0.05

    for step in range(start_step, total_steps + 1):
        # BUG FIX: previously, `last_out` only ever held the FINAL micro-batch's
        # output dict, so every logged/EMA loss value was a single noisy sample
        # discarding the other (grad_accum - 1) micro-batches entirely. We now
        # accumulate a token-weighted sum across the whole accumulation window
        # so the reported loss reflects the true average over all non-padding
        # (non -1 ignore_index) tokens processed this optimizer step, not a
        # division by the static grad_accum constant (which only approximates
        # the correct average when every micro-batch has the same valid-token
        # count) and not just the last sample.
        step_loss_sum = 0.0
        step_token_count = 0
        step_aux_totals = {"z_loss": 0.0, "mtp_loss": 0.0, "jepa_loss": 0.0}
        step_aux_counts = {"z_loss": 0, "mtp_loss": 0, "jepa_loss": 0}

        for _ in range(grad_accum):
            x, y = get_batch()
            x, y = x.to(device), y.to(device)

            with amp_ctx:
                out = model(x, targets=y, eos_id=eos_id)
                loss = model.total_loss(out) / grad_accum

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tokens_seen += x.numel()

            # Token-weighted accounting for accurate logging. y == -1 marks
            # positions excluded from lm_loss by ignore_index (document
            # boundaries / padding), so we weight by the actual valid count
            # rather than assuming every micro-batch contributes equally.
            n_valid = int((y != -1).sum().item())
            if n_valid > 0 and "lm_loss" in out and torch.isfinite(out["lm_loss"]):
                step_loss_sum += out["lm_loss"].item() * n_valid
                step_token_count += n_valid
            for k in step_aux_totals:
                if k in out and torch.isfinite(out[k]):
                    step_aux_totals[k] += out[k].item()
                    step_aux_counts[k] += 1

        if use_scaler:
            # unscale_ must be called for BOTH optimizers before
            # clip_grad_norm_, regardless of how clipping is split below —
            # otherwise we'd be clipping still-scaled (e.g. ×65536) gradients
            # against a threshold meant for unscaled ones.
            scaler.unscale_(muon)
            scaler.unscale_(adamw)

        # BUG FIX: clip Muon-owned and AdamW-owned parameters SEPARATELY rather
        # than one global clip_grad_norm_(model.parameters(), ...) call. Muon's
        # Newton-Schulz orthogonalisation already self-bounds its own updates
        # (RMS-normalised via _ns5, then rescaled by max(shape)**0.5) — it does
        # not want or need traditional norm clipping on top. Worse, a single
        # combined norm across both groups lets a spike in the tied
        # embedding/lm_head gradient (AdamW's group) dominate the norm and
        # silently shrink Muon's attention/FFN updates by the same factor on
        # every such step, even though those matrices were perfectly healthy —
        # stalling the model's core learning. We compute and clip each group's
        # norm independently instead.
        muon_params = [p for g in muon.param_groups for p in g["params"]]
        adamw_params = [p for g in adamw.param_groups for p in g["params"]]
        muon_grad_norm = torch.nn.utils.clip_grad_norm_(muon_params, 1.0)
        adamw_grad_norm = torch.nn.utils.clip_grad_norm_(adamw_params, 0.5)

        # LOSS-SPIKE GUARD: a finite-but-catastrophic loss (no NaN/Inf, so the
        # NaN/Inf guard below never fires) can still apply a bad update that
        # compounds — e.g. a batch that pushes softcapped logits into
        # saturation, collapsing the output distribution toward uniform
        # (loss → ln(vocab_size)) for many subsequent steps. We detect this
        # using the token-weighted average already computed for this
        # accumulation window and scale gradients down rather than zeroing
        # them — a full skip wastes the batch's signal entirely; a
        # down-scaled step still nudges in the right direction without the
        # violent jump that an un-damped full-LR step would cause.
        step_avg_loss_pre = (
            step_loss_sum / step_token_count if step_token_count > 0 else None
        )
        if (
            step_avg_loss_pre is not None
            and ema_loss is not None
            and step_avg_loss_pre > ema_loss + 1.5
        ):
            overshoot = step_avg_loss_pre - ema_loss
            spike_scale = max(0.1, 1.0 / (1.0 + overshoot))
            _log(
                f"[train] step {step}: loss spike {step_avg_loss_pre:.4f} vs "
                f"ema {ema_loss:.4f} (delta={overshoot:.2f}) — scaling grads x{spike_scale:.3f}"
            )
            muon.zero_grad(set_to_none=True)
            adamw.zero_grad(set_to_none=True)
            continue
            # with torch.no_grad():
            #     for p in muon_params + adamw_params:
            #         if p.grad is not None:
            #             p.grad.mul_(spike_scale)

        # BUG FIX: NaN/Inf skip-step guard. Without this, a single bad batch
        # (or a transient float16 overflow before the scaler adapts) writes
        # NaN into the weights via optimizer.step(), and every subsequent
        # step propagates NaN forever — the run is silently dead but keeps
        # "training" and logging numbers. scaler.step() already skips the
        # AdamW update on inf/NaN grads internally, but Muon doesn't set
        # _step_supports_amp_scaling, so Muon's step still runs unguarded.
        # We check each optimizer's own grad norm independently now that
        # clipping is split, since a NaN in one group doesn't imply a NaN in
        # the other and we don't want a bad Muon batch to also block the
        # (perfectly fine) AdamW update, or vice versa.
        skip_muon = not torch.isfinite(muon_grad_norm)
        skip_adamw = not torch.isfinite(adamw_grad_norm)
        if skip_muon:
            _log(
                f"[train] step {step}: non-finite Muon grad_norm={muon_grad_norm.item()} — skipping Muon step"
            )
        if skip_adamw:
            _log(
                f"[train] step {step}: non-finite AdamW grad_norm={adamw_grad_norm.item()} — skipping AdamW step"
            )
        skip_step = skip_muon and skip_adamw

        if use_scaler:
            if not skip_muon:
                scaler.step(muon)
            if not skip_adamw:
                scaler.step(adamw)
            scaler.update()
        else:
            if not skip_muon:
                muon.step()
            if not skip_adamw:
                adamw.step()

        muon.zero_grad(set_to_none=True)
        adamw.zero_grad(set_to_none=True)
        if not skip_step:
            scheduler.step()

        # BUG FIX: use the token-weighted step average accumulated above
        # instead of the last micro-batch's single lm_loss sample, so the EMA
        # reflects the true loss over every non-padding token seen this step.
        if step_token_count > 0:
            step_avg_loss = step_loss_sum / step_token_count
            ema_loss = (
                step_avg_loss
                if ema_loss is None
                else ema_loss * (1 - ema_alpha) + step_avg_loss * ema_alpha
            )

        if step % 50 == 0:
            now = time.time()
            elapsed = now - t0
            t_last_log = now

            lm_loss = (
                step_loss_sum / step_token_count
                if step_token_count > 0
                else float("nan")
            )
            z_loss = (
                step_aux_totals["z_loss"] / step_aux_counts["z_loss"]
                if step_aux_counts["z_loss"] > 0
                else 0.0
            )
            mtp_loss = (
                step_aux_totals["mtp_loss"] / step_aux_counts["mtp_loss"]
                if step_aux_counts["mtp_loss"] > 0
                else 0.0
            )
            jepa_loss = (
                step_aux_totals["jepa_loss"] / step_aux_counts["jepa_loss"]
                if step_aux_counts["jepa_loss"] > 0
                else 0.0
            )
            lr_muon = muon.param_groups[0]["lr"]
            lr_adamw = adamw.param_groups[0]["lr"]
            toks_s = tokens_seen / elapsed

            vram_str = "N/A    "
            if device.type == "cuda":
                vram_str = f"{torch.cuda.memory_allocated() / 1e9:.2f} GB"

            bpb = lm_loss / math.log(2) if not math.isnan(lm_loss) else float("nan")

            _log(
                f"{step:>8} | {lm_loss:>9.4f} | {bpb:>7.4f} | {z_loss:>7.4f} | "
                f"{mtp_loss:>8.4f} | {jepa_loss:>9.5f} | {lr_muon:>10.2e} | "
                f"{lr_adamw:>10.2e} | {toks_s:>8.0f} | {vram_str}"
            )
            _log(
                f"         | ema_loss={ema_loss:.4f}  elapsed={elapsed / 60:.1f}min  "
                f"eta={((total_steps - step) / max(step - start_step + 1, 1) * elapsed) / 60:.1f}min"
            )

            if HAS_WANDB and args.run != "dummy":
                wandb.log(
                    {
                        "lm_loss": lm_loss,
                        "bpb": bpb,
                        "z_loss": z_loss,
                        "mtp_loss": mtp_loss,
                        "jepa_loss": jepa_loss,
                        "lr_muon": lr_muon,
                        "lr_adamw": lr_adamw,
                        "tok_per_sec": toks_s,
                        "ema_loss": ema_loss,
                        "step": step,
                    },
                    step=step,
                )

        if step % 2000 == 0:
            _save(
                model,
                optimizers,
                step,
                args.out_dir,
                vocab_size,
                logger=_logger,
                scheduler=scheduler,
            )

    _logger.separator("=")
    _save(
        model,
        optimizers,
        total_steps,
        args.out_dir,
        vocab_size,
        final=True,
        logger=_logger,
    )

    total_time = time.time() - t0
    _log(
        f"[train] Finished.  Total time: {total_time / 60:.1f} min  "
        f"({total_time / 3600:.2f} hr)"
    )
    _log(f"[train] Total tokens processed: {tokens_seen:,}")
    if ema_loss:
        _log(
            f"[train] Final EMA loss: {ema_loss:.4f}  ({ema_loss / math.log(2):.4f} bpb)"
        )
    _logger.separator("=")
    _logger.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Hyperion-v3 training script")

    # Core
    p.add_argument("--preset", default="3050ti", choices=["3050ti", "t4"])
    p.add_argument("--run", default="dummy", help="Run name for logging / wandb")
    p.add_argument("--out_dir", default="checkpoints")
    p.add_argument(
        "--steps", type=int, default=0, help="Override total_steps (0 = use preset)"
    )

    # Dataset cache
    p.add_argument(
        "--cache_dir",
        default="data_cache",
        help="Directory for tokenized shard files (default: data_cache/)",
    )
    p.add_argument(
        "--n_shards",
        type=int,
        default=64,
        help="Number of shard files to generate (default: 64, ~256M tokens)",
    )
    p.add_argument(
        "--seqs_per_shard",
        type=int,
        default=4096,
        help="Sequences per shard file (default: 4096 ≈ 4M tokens)",
    )
    p.add_argument(
        "--regen_cache",
        action="store_true",
        help="Force re-download and re-tokenize even if cache exists",
    )

    # Optimizer
    p.add_argument("--matrix_lr", type=float, default=0.001)
    p.add_argument("--embedding_lr", type=float, default=0.009)
    p.add_argument(
        "--weight_decay",
        type=float,
        default=0.0,
        help="AdamW weight decay applied to decay-eligible params "
        "(biases, scales, and embeddings are always excluded — see optim.py)",
    )

    # Vocab
    p.add_argument("--vocab_size", type=int, default=12288)

    # Model flags
    p.add_argument("--force_dense", action="store_true")

    # Tokenizer
    p.add_argument("--tok_path", default="tokenizer.json")
    p.add_argument("--no_auto_tok", action="store_true")
    p.add_argument("--train_tok", action="store_true")

    # Resume
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the latest checkpoint in out_dir",
    )

    # Auth
    p.add_argument("--hf_token", type=str, default=None)

    args = p.parse_args()

    if args.train_tok:
        _logger = TrainLogger(args.out_dir, "tokenizer")
        _hf_login(args.hf_token)
        train_tokenizer(args.tok_path, vocab_size=PRESETS[args.preset]["vocab_size"])
        _logger.close()
        sys.exit(0)

    train(args)
