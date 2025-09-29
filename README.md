Speculative Decoding (PyTorch + Transformers)
============================================

Concise proof-of-concep of speculative decoding for causal LMs.
It compares standard sampling vs speculative decoding, and reports tokens/sec and average tokens accepted per draft
call.

Features

- Baseline vs speculative decoding side‑by‑side
- Works with any Hugging Face causal LM (defaults: `gpt2` + `distilgpt2`)
- Simple config for `draft_k`, temperature, top‑p, device, and dtype

Requirements

- Python 3.11+
- PyTorch 2.2+, Transformers 4.41+, Accelerate
- CUDA GPU recommended for speed (CPU works but slower)

Setup

- Using uv (recommended):
    - `uv sync` (creates `.venv` and installs from `pyproject.toml`/`uv.lock`)
- Using pip:
    - `python -m venv .venv && source .venv/bin/activate`
    - `pip install torch>=2.2 transformers>=4.41 accelerate`

Run

- `python main.py`
- The script prints baseline output, speculative output, tokens/sec, and tokens accepted per call.

Configuration

- Edit `main.py`:
    - Models: `load_models(base_checkpoint="gpt2", draft_checkpoint="distilgpt2", ...)`
    - Decoding: tweak `SpeculativeConfig(draft_k, max_new_tokens, temperature, top_p, device, dtype)`
- Tips:
    - On GPU: `device="cuda"`, `dtype=torch.float16` (or `bfloat16` if preferred)
    - On CPU: use `device="cpu"` and `dtype=torch.float32` for compatibility

Notes

- The implementation follows the canonical chain speculative decoding algorithm and preserves the base model’s sampling
  distribution when using the same temperature/top‑p.
