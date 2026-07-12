# Hyperion-v3

A from-scratch transformer language model (custom attention, expert-choice MoE,
hyperbolic regularization, ternary-quant linear layers) trained from raw text
to a tokenizer-level checkpoint.

## Demo

```bash
pip install -r requirements.txt
python app.py --ckpt checkpoints/final.pt --tok_path tokenizer.json --share
```

## CLI inference

```bash
python infer.py --ckpt checkpoints/final.pt --tok_path tokenizer.json \
    --prompt "Once upon a time"
```

## Benchmarks

```bash
python benchmark.py speed   --ckpt checkpoints/final.pt --tok_path tokenizer.json
python benchmark.py quality --ckpt checkpoints/final.pt --tok_path tokenizer.json --eval_file holdout.txt
python benchmark.py compare --ckpt checkpoints/final.pt --tok_path tokenizer.json --eval_file holdout.txt --baseline gpt2
```

| Metric | Hyperion-v3 | gpt2 (baseline) |
|---|---|---|
| Bits-per-byte (held-out) | _fill in_ | _fill in_ |
| Tokens/sec (CPU/GPU) | _fill in_ | — |

## Training

See `train.py` / `optim.py`. Resume with:

```bash
python train.py --preset <name> --run <name> --resume --steps 10000
```

## Architecture

- Config: `model.py::HyperionConfig`
- d_model / n_layers / n_heads set via `build_model()`
- Notable components: `HyperionAttention`, `ExpertChoiceMoE`, `HyperbolicRegularizer`, `TernaryQuantSTE`

## Limitations
- No KV cache at inference — `generate()` re-runs the full forward pass per token.
