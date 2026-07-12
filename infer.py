"""
Hyperion-v3: Inference / text generation script
=================================================
Loads a checkpoint saved by train.py (e.g. ckpt_006000.pt or final.pt) and
generates text from a prompt.

Usage:
  python infer.py --ckpt checkpoints/ckpt_006000.pt --tok_path tokenizer.json \
      --prompt "Once upon a time" --max_new_tokens 200

  # Lower temperature, narrower sampling:
  python infer.py --ckpt checkpoints/ckpt_006000.pt --prompt "..." --temperature 0.7 --top_k 40

  # Interactive loop (omit --prompt):
  python infer.py --ckpt checkpoints/ckpt_006000.pt
"""

import argparse
from dataclasses import fields

import torch
from model import Hyperion, HyperionConfig


def load_model(ckpt_path: str, device: torch.device) -> Hyperion:
    print(f"[infer] Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    # The checkpoint's "config" dict was saved via dataclasses.asdict(model.cfg)
    # in train.py's _save(), so it contains exactly HyperionConfig's fields.
    # Filter defensively in case the checkpoint was written by an older/newer
    # train.py with extra/missing keys, so this script doesn't break on a
    # stray key mismatch.
    valid_keys = {f.name for f in fields(HyperionConfig)}
    cfg_dict = {k: v for k, v in ckpt["config"].items() if k in valid_keys}
    dropped = set(ckpt["config"].keys()) - valid_keys
    if dropped:
        print(
            f"[infer] Ignoring unknown config keys from checkpoint: {sorted(dropped)}"
        )

    cfg = HyperionConfig(**cfg_dict)
    model = Hyperion(cfg).to(device)

    state_dict = ckpt["model"]
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[infer] WARNING missing keys in state_dict: {missing}")
    if unexpected:
        print(f"[infer] WARNING unexpected keys in state_dict: {unexpected}")

    model.eval()

    step = ckpt.get("step", "?")
    vocab_size = ckpt.get("vocab_size", cfg.vocab_size)
    print(
        f"[infer] Loaded step={step}  d_model={cfg.d_model}  n_layers={cfg.n_layers}  "
        f"vocab_size={vocab_size}"
    )
    return model, vocab_size


def load_tokenizer(tok_path: str):
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tok_path)
    eos_id = tok.get_vocab().get("<eos>", 1)
    print(
        f"[infer] Tokenizer loaded: {tok_path}  vocab={tok.get_vocab_size()}  eos_id={eos_id}"
    )
    return tok, eos_id


@torch.no_grad()
def generate_text(
    model: Hyperion,
    tok,
    prompt: str,
    eos_id: int,
    device: torch.device,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 50,
    stop_at_eos: bool = True,
) -> str:
    ids = tok.encode(prompt).ids
    if not ids:
        # Empty prompt: seed with eos so the model has at least one token of context.
        ids = [eos_id]
    input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    # model.generate() runs the full sequence through forward() at every step
    # (no KV cache), generating max_new_tokens greedily/sampled one at a time.
    # NOTE: model.generate() calls self.eval() internally but does not restore
    # the model's prior training mode afterward. Since this script only does
    # inference, that's fine here — but if you reuse load_model()'s returned
    # model object for further training in the same process, call model.train()
    # explicitly afterward.
    out_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )

    gen_ids = out_ids[0].tolist()
    if stop_at_eos and eos_id in gen_ids[len(ids) :]:
        cut = len(ids) + gen_ids[len(ids) :].index(eos_id)
        gen_ids = gen_ids[:cut]

    return tok.decode(gen_ids)


def main():
    p = argparse.ArgumentParser(description="Hyperion-v3 text generation")
    p.add_argument(
        "--ckpt",
        required=True,
        help="Path to checkpoint, e.g. checkpoints/ckpt_006000.pt",
    )
    p.add_argument("--tok_path", default="tokenizer.json")
    p.add_argument(
        "--prompt", default=None, help="If omitted, drops into an interactive loop"
    )
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument(
        "--no_stop_at_eos",
        action="store_true",
        help="Don't truncate output at the first <eos>",
    )
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[infer] Device: {device}")

    model, _ = load_model(args.ckpt, device)
    tok, eos_id = load_tokenizer(args.tok_path)

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        stop_at_eos=not args.no_stop_at_eos,
    )

    if args.prompt is not None:
        text = generate_text(model, tok, args.prompt, eos_id, device, **gen_kwargs)
        print("\n" + "=" * 72)
        print(text)
        print("=" * 72)
        return

    print("[infer] Interactive mode — type a prompt and press Enter (Ctrl+C to quit)")
    while True:
        try:
            prompt = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            print("\n[infer] Exiting")
            break
        if not prompt.strip():
            continue
        text = generate_text(model, tok, prompt, eos_id, device, **gen_kwargs)
        print(text)


if __name__ == "__main__":
    main()
