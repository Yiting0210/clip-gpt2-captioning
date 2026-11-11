"""Decoding strategies: greedy, beam search, nucleus (top-p) sampling.

All functions take a batch of CLIP features and return one caption string per image.
Strategy specs used on the command line / in configs:
    "greedy"  |  "beam_w5" (beam width 5)  |  "nucleus_p0.9_t0.7" (top-p 0.9, temperature 0.7)
Any spec can end in "_nr3": no 3-gram may appear twice in a caption (no-repeat n-gram blocking).
"""
from __future__ import annotations

import re

import torch

NEG = -1e9


def _bos(model, B: int, device, tokenizer) -> torch.Tensor:
    return torch.full((B, 1), tokenizer.eos_token_id, dtype=torch.long, device=device)


def _no_empty(logits: torch.Tensor, step: int, eos: int) -> torch.Tensor:
    """Forbid <|endoftext|> as the very first token, so no caption comes out empty."""
    if step == 0:
        logits = logits.clone()
        logits[:, eos] = NEG
    return logits


def _ban_repeats(logits: torch.Tensor, seqs: torch.Tensor, n: int) -> torch.Tensor:
    """No-repeat n-gram blocking: forbid any token that would complete an n-gram already in the sequence.
    seqs[:, 0] is the bos token and is ignored."""
    if not n or seqs.shape[1] - 1 < n - 1:
        return logits
    logits = logits.clone()
    for row, toks in enumerate(seqs[:, 1:].tolist()):
        prefix = tuple(toks[len(toks) - n + 1:]) if n > 1 else ()
        banned = {toks[i + n - 1] for i in range(len(toks) - n + 1) if tuple(toks[i:i + n - 1]) == prefix}
        if banned:
            logits[row, list(banned)] = NEG
    return logits


def _to_text(seqs: torch.Tensor, tokenizer) -> list[str]:
    """Drop the leading bos, cut at the first eos, decode."""
    eos = tokenizer.eos_token_id
    out = []
    for row in seqs.tolist():
        row = row[1:]
        if eos in row:
            row = row[:row.index(eos)]
        out.append(" ".join(tokenizer.decode(row).split()))  # collapse newlines / repeated spaces
    return out


@torch.no_grad()
def greedy(model, feats, tokenizer, max_len: int = 40, return_ids: bool = False, no_repeat: int = 0):
    return _sample_loop(model, feats, tokenizer, max_len, lambda logits: logits.argmax(-1), return_ids, no_repeat)


@torch.no_grad()
def nucleus(model, feats, tokenizer, top_p: float = 0.9, temperature: float = 1.0,
            max_len: int = 40, generator: torch.Generator | None = None, no_repeat: int = 0) -> list[str]:
    def pick(logits):
        probs = torch.softmax(logits / max(temperature, 1e-5), dim=-1)
        sp, si = probs.sort(dim=-1, descending=True)
        keep = sp.cumsum(-1) - sp < top_p          # smallest set whose mass reaches top_p (always >= 1 token)
        sp = sp * keep
        choice = torch.multinomial(sp / sp.sum(-1, keepdim=True), 1, generator=generator)
        return si.gather(-1, choice).squeeze(-1)
    return _sample_loop(model, feats, tokenizer, max_len, pick, no_repeat=no_repeat)


def _sample_loop(model, feats, tokenizer, max_len, pick, return_ids: bool = False, no_repeat: int = 0):
    eos = tokenizer.eos_token_id
    ctx = model.encode_context(feats)
    seqs = _bos(model, feats.shape[0], feats.device, tokenizer)
    done = torch.zeros(feats.shape[0], dtype=torch.bool, device=feats.device)
    for step in range(max_len + 1):
        tok = pick(_ban_repeats(_no_empty(model.next_logits(ctx, seqs), step, eos), seqs, no_repeat))
        tok = torch.where(done, torch.full_like(tok, eos), tok)
        seqs = torch.cat([seqs, tok[:, None]], dim=1)
        done |= tok == eos
        if done.all():
            break
    return seqs if return_ids else _to_text(seqs, tokenizer)


@torch.no_grad()
def beam_search(model, feats, tokenizer, beam_width: int = 5, max_len: int = 40,
                length_penalty: float = 1.0, return_ids: bool = False, no_repeat: int = 0):
    """Standard beam search; the final pick maximises sum(log p) / length**length_penalty."""
    eos, K, B = tokenizer.eos_token_id, beam_width, feats.shape[0]
    ctx = model.expand_context(model.encode_context(feats), K)          # (B*K, ...)
    seqs = _bos(model, B * K, feats.device, tokenizer)
    scores = torch.full((B, K), NEG, device=feats.device)
    scores[:, 0] = 0.0                                                   # identical beams at start: keep one
    lengths = torch.zeros(B, K, device=feats.device)
    done = torch.zeros(B, K, dtype=torch.bool, device=feats.device)

    for step in range(max_len + 1):
        logits = _ban_repeats(_no_empty(model.next_logits(ctx, seqs), step, eos), seqs, no_repeat)
        logp = torch.log_softmax(logits, dim=-1).view(B, K, -1)
        V = logp.shape[-1]
        # finished beams may only extend with eos at no cost
        frozen = torch.full_like(logp, NEG)
        frozen[..., eos] = 0.0
        logp = torch.where(done[..., None], frozen, logp)
        cand = (scores[..., None] + logp).view(B, K * V)
        scores, idx = cand.topk(K, dim=-1)
        beam, tok = idx // V, idx % V
        flat = (beam + torch.arange(B, device=feats.device)[:, None] * K).view(-1)
        seqs = torch.cat([seqs[flat], tok.view(-1, 1)], dim=1)
        prev_done = done.gather(1, beam)
        lengths = lengths.gather(1, beam) + (~prev_done).float()
        done = prev_done | (tok == eos)
        if done.all():
            break

    norm = scores / lengths.clamp(min=1) ** length_penalty
    best = norm.argmax(-1) + torch.arange(B, device=feats.device) * K
    return seqs[best] if return_ids else _to_text(seqs[best], tokenizer)


# ---------------------------------------------------------------------- strategy specs
def parse_strategy(spec: str) -> tuple[str, dict]:
    base, kw = spec, {}
    m = re.fullmatch(r"(.+)_nr(\d+)", spec)
    if m:
        base, kw = m.group(1), {"no_repeat": int(m.group(2))}
    if base == "greedy":
        return "greedy", kw
    m = re.fullmatch(r"beam_w(\d+)", base)
    if m:
        return "beam", {"beam_width": int(m.group(1)), **kw}
    m = re.fullmatch(r"nucleus_p([\d.]+)_t([\d.]+)", base)
    if m:
        return "nucleus", {"top_p": float(m.group(1)), "temperature": float(m.group(2)), **kw}
    raise ValueError(f"unknown decoding spec {spec!r}; use greedy | beam_w5 | nucleus_p0.9_t0.7 (+ optional _nr3)")


def generate(model, feats, tokenizer, spec: str = "beam_w3", max_len: int = 40, seed: int | None = None) -> list[str]:
    name, kw = parse_strategy(spec)
    if name == "greedy":
        return greedy(model, feats, tokenizer, max_len, **kw)
    if name == "beam":
        return beam_search(model, feats, tokenizer, max_len=max_len, **kw)
    gen = None
    if seed is not None:
        gen = torch.Generator(device=feats.device).manual_seed(seed)
    return nucleus(model, feats, tokenizer, max_len=max_len, generator=gen, **kw)


def generate_batched(model, feats, tokenizer, spec: str, batch_size: int = 64, **kw) -> list[str]:
    out = []
    for s in range(0, feats.shape[0], batch_size):
        out += generate(model, feats[s:s + batch_size], tokenizer, spec, **kw)
    return out
