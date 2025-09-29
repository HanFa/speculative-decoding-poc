import time
import torch
import logging
from typing import List
from torch.nn.functional import softmax
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.basicConfig(level=logging.INFO)


@dataclass
class SpeculativeConfig:
    draft_k: int = 4
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 1.0
    device: str = "cpu"
    dtype: torch.dtype = torch.float16  # 4090 is fine with fp16; switch to bfloat16 if you like


def load_models(base_checkpoint="gpt2", draft_checkpoint="distilgpt2", device="cuda", dtype=torch.float16):
    tokenizer = AutoTokenizer.from_pretrained(base_checkpoint)
    base = AutoModelForCausalLM.from_pretrained(base_checkpoint, torch_dtype=dtype, device_map=device)
    draft = AutoModelForCausalLM.from_pretrained(draft_checkpoint, torch_dtype=dtype, device_map=device)
    base.eval()
    draft.eval()
    return base, draft, tokenizer


def apply_temperature_and_top_p(logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0) -> torch.Tensor:
    """
    Convert logits -> probabilities with temperature and (optional) nucleus top-p.
    Returns probabilities (sum to 1).
    logits: [vocab]
    """
    logits = logits / max(1e-6, temperature)
    probs = softmax(logits.float(), dim=-1)

    # nucleus filtering on probabilities
    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    # keep smallest set with cumulative <= top_p (ensure at least 1 token kept)
    keep_sorted = cumulative <= top_p
    keep_sorted[..., 0] = True

    keep_mask = torch.zeros_like(probs, dtype=torch.bool)
    keep_mask.scatter_(dim=-1, index=sorted_idx, src=keep_sorted)

    filtered = probs * keep_mask
    filtered = filtered / filtered.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    return filtered


def sample_from_probs(probs: torch.Tensor) -> int:
    """Multinomial sample from probability vector"""
    return torch.multinomial(probs, num_samples=1).item()


@torch.inference_mode()
def run_non_speculative_decoding_inference(cfg: SpeculativeConfig, prompt: str, base,
                                           draft, tokenizer):
    t0 = time.perf_counter()
    out = base.generate(
        **tokenizer(prompt, return_tensors="pt").to(cfg.device),
        do_sample=True, temperature=cfg.temperature, top_p=cfg.top_p,
        max_new_tokens=cfg.max_new_tokens, use_cache=True
    )
    t1 = time.perf_counter()
    baseline_text = tokenizer.decode(out[0], skip_special_tokens=True)
    baseline_tps = cfg.max_new_tokens / (t1 - t0)
    logging.info("\n=== OUTPUT (Baseline) ===")
    logging.info(baseline_text)
    logging.info(f"[Baseline] tokens / s = {baseline_tps:.2f}")


@torch.inference_mode()
def run_speculative_decoding_inference(cfg: SpeculativeConfig, prompt: str, base,
                                       draft, tokenizer):
    t0 = time.perf_counter()
    out_spec, tpc, calls = _run_speculative_decoding_inference(cfg, prompt, base, draft, tokenizer)
    t1 = time.perf_counter()
    spec_txt = tokenizer.decode(out_spec[0], skip_special_tokens=True)
    spec_tps = (out_spec.shape[1] - tokenizer(prompt, return_tensors="pt").input_ids.shape[1]) / (t1 - t0)

    logging.info("=== OUTPUT (Speculative) ===")
    logging.info(spec_txt)
    logging.info(
        f"[Speculative] draft_k={cfg.draft_k} | TPC (tokens accepted per call) ≈ {tpc:.2f} | rounds = {calls} "
        f"| tokens/s ≈ {spec_tps:.2f}")


def _run_speculative_decoding_inference(cfg: SpeculativeConfig, prompt: str, base,
                                        draft, tokenizer):
    """
    Implements the canonical chain speculative decoding loop.

    Each round:
      1) Draft proposes k tokens autoregressively: y1..yk (we keep q_i(.))
      2) Base validates in one pass and produces p_i(.), and p_{k+1}(.)
      3) Accept y1..y_{j-1} with probabilities a_i = min(1, p_i(y_i)/q_i(y_i)); on first rejection j,
         draw z ~ r = (p_j - a_j * q_j) / (1 - a_j).
         If all accepted, draw z ~ p_{k+1}.
      4) Append accepted_prefix + z to the output; repeat until max_new_tokens.

    This preserves the base model's sampling distribution (given same temperature/top-p used for p_i(.)).
    """

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(cfg.device)
    output_ids = input_ids.clone()

    n_generated = 0
    calls = 0
    total_tokens_accepted_per_call = 0
    while n_generated < cfg.max_new_tokens:
        calls += 1
        draft_context = output_ids.clone()

        # -----------------------
        # 1) DRAFT: y_1..y_k with q_i(.)
        # -----------------------
        proposed = []  # list of sampled proposed tokens from vectors in each step
        q_dists = []  # list of probability vectors for each step

        for _ in range(cfg.draft_k):
            draft_logits = draft(draft_context).logits[:, -1, :].squeeze(0)
            q_i = apply_temperature_and_top_p(draft_logits, cfg.temperature, cfg.top_p)
            y_i = sample_from_probs(q_i)

            proposed.append(y_i)
            q_dists.append(q_i)

            draft_context = torch.cat([draft_context, torch.tensor([[y_i]], device=cfg.device)], dim=1)

        # -----------------------
        # 2) BASE: validate all at once (and get p_{k+1})
        # -----------------------
        base_inputs = torch.cat([output_ids, torch.tensor([proposed], device=cfg.device)], dim=1)
        base_logits = base(base_inputs).logits.squeeze(0)  # [seq_len, vocab]

        seq_len = base_inputs.shape[1]
        base_start_pos = seq_len - len(proposed) - 1  # where y1 is predicted
        p_dists = []

        for i in range(len(proposed)):
            logits_i = base_logits[base_start_pos + i]
            p_i = apply_temperature_and_top_p(logits_i, cfg.temperature, cfg.top_p)
            p_dists.append(p_i)

        logits_k1 = base_logits[seq_len - 1]
        p_k1 = apply_temperature_and_top_p(logits_k1, cfg.temperature, cfg.top_p)

        # -----------------------
        # 3) ACCEPT / CORRECT
        # -----------------------
        accepted_prefix: List[int] = []
        generated_prefix: List[int] = []

        reject_idx = None
        for idx, y_i in enumerate(proposed):
            p_i = p_dists[idx]
            q_i = q_dists[idx]

            prob_accept = min(1.0, (p_i[y_i].item() / (q_i[y_i] + 1e-12)))

            if torch.rand(()) < prob_accept:  # accept with probability min(1, p/q)
                accepted_prefix.append(y_i)
                continue

            reject_idx = idx  # if rejected, samples from the corrected base model distribution
            corrected_dist = (p_i - prob_accept * q_i).clamp_min(0.0)  # correction draw should be [p - q]+
            corrected_dist = corrected_dist / corrected_dist.sum().clamp_min(1e-12)

            corrected_yi = sample_from_probs(corrected_dist)
            generated_prefix = accepted_prefix + [corrected_yi]
            break

        if reject_idx is None:
            # everything accepted; sample z ~ p_{k+1}
            y_k1 = sample_from_probs(p_k1)
            generated_prefix = accepted_prefix + [y_k1]

        total_tokens_accepted_per_call += len(generated_prefix)
        output_ids = torch.cat([output_ids, torch.tensor(
            [generated_prefix], device=cfg.device
        )], dim=1)
        n_generated += len(generated_prefix)

        # early stop if eos
        if output_ids[0, -1].item() == tokenizer.eos_token_id:
            break

    tpc = total_tokens_accepted_per_call / max(1, calls)
    return output_ids, tpc, calls


def main():
    prompt = "Write a short, 2-sentence explanation of speculative decoding."
    cfg = SpeculativeConfig(draft_k=4, max_new_tokens=80, temperature=1.0, top_p=1.0, device="cuda")

    base, draft, tokenizer = load_models(device=cfg.device, dtype=cfg.dtype)

    run_non_speculative_decoding_inference(cfg, prompt, base, draft, tokenizer)
    run_speculative_decoding_inference(cfg, prompt, base, draft, tokenizer)


if __name__ == '__main__':
    main()
