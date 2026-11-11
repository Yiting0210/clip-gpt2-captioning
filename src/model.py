"""CLIP-feature -> GPT-2 captioning model.

Two ways to inject the image into GPT-2:
  prefix           - an MLP maps the pooled CLIP embedding (512) to `prefix_length` pseudo-token
                     embeddings that are prepended to the caption (ClipCap, Mokady et al. 2021).
  cross_attention  - GPT-2 gets a cross-attention layer in every block that attends to the
                     50 CLIP patch features (projected to GPT-2 width).

Three fine-tuning strategies (what is trained besides the mapping network):
  prefix  - GPT-2 frozen (in cross_attention mode the new cross-attention layers still train)
  lora    - low-rank adapters on GPT-2 self-attention (peft)
  full    - every GPT-2 weight
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import yaml
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

INJECTIONS = ("prefix", "cross_attention")
FINETUNES = ("prefix", "lora", "full")


def _is_cross(name: str) -> bool:
    return "crossattention" in name or "ln_cross_attn" in name


def build_gpt2(name: str, add_cross_attention: bool) -> GPT2LMHeadModel:
    if name == "tiny-random":  # for unit tests: no download, 2 layers, width 64
        cfg = GPT2Config(n_layer=2, n_head=2, n_embd=64, add_cross_attention=add_cross_attention)
        return GPT2LMHeadModel(cfg)
    return GPT2LMHeadModel.from_pretrained(name, add_cross_attention=add_cross_attention)


def load_tokenizer(gpt2_model: str = "gpt2") -> GPT2TokenizerFast:
    return GPT2TokenizerFast.from_pretrained("gpt2" if gpt2_model == "tiny-random" else gpt2_model)


class ClipCaptionModel(nn.Module):
    def __init__(self, gpt2_model: str = "gpt2", injection: str = "prefix", finetune: str = "prefix",
                 feat_dim: int = 512, prefix_length: int = 10, mapping_hidden: int | None = None,
                 lora: dict | None = None):
        super().__init__()
        assert injection in INJECTIONS, injection
        assert finetune in FINETUNES, finetune
        self.injection, self.finetune, self.prefix_length = injection, finetune, prefix_length

        gpt = build_gpt2(gpt2_model, add_cross_attention=(injection == "cross_attention"))
        d = gpt.config.n_embd
        self.embed_dim = d
        if injection == "prefix":
            hidden = mapping_hidden or (d * prefix_length) // 2
            self.mapper = nn.Sequential(nn.Linear(feat_dim, hidden), nn.Tanh(),
                                        nn.Linear(hidden, d * prefix_length))
        else:
            self.mapper = nn.Sequential(nn.Linear(feat_dim, d), nn.LayerNorm(d))

        if finetune == "full":
            for p in gpt.parameters():
                p.requires_grad = True
        elif finetune == "prefix":
            for n, p in gpt.named_parameters():
                p.requires_grad = _is_cross(n)
        else:  # lora
            from peft import LoraConfig, get_peft_model
            lora = lora or {}
            cfg = LoraConfig(r=lora.get("r", 8), lora_alpha=lora.get("alpha", 16),
                             lora_dropout=lora.get("dropout", 0.05), bias="none", fan_in_fan_out=True,
                             target_modules=lora.get("target_modules", r".*\.attn\.(c_attn|c_proj)"))
            gpt = get_peft_model(gpt, cfg)
            for n, p in gpt.named_parameters():
                if _is_cross(n):
                    p.requires_grad = True
        self.gpt = gpt

    # ------------------------------------------------------------------ helpers
    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.gpt.get_input_embeddings()(input_ids)

    def encode_context(self, feats: torch.Tensor) -> dict:
        """Image features -> what GPT-2 is conditioned on."""
        if self.injection == "prefix":
            return {"prefix": self.mapper(feats).view(feats.shape[0], self.prefix_length, self.embed_dim)}
        return {"enc": self.mapper(feats)}  # (B, 50, d)

    @staticmethod
    def expand_context(ctx: dict, k: int) -> dict:
        return {key: v.repeat_interleave(k, dim=0) for key, v in ctx.items()}

    # ------------------------------------------------------------------ training
    def forward(self, feats, input_ids, attention_mask=None, labels=None):
        ctx = self.encode_context(feats)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if self.injection == "prefix":
            pre = ctx["prefix"]
            B, L = pre.shape[:2]
            emb = torch.cat([pre, self.embed_tokens(input_ids)], dim=1)
            mask = torch.cat([attention_mask.new_ones(B, L), attention_mask], dim=1)
            if labels is not None:
                labels = torch.cat([labels.new_full((B, L), -100), labels], dim=1)
            return self.gpt(inputs_embeds=emb, attention_mask=mask, labels=labels)
        return self.gpt(input_ids=input_ids, attention_mask=attention_mask,
                        encoder_hidden_states=ctx["enc"], labels=labels)

    # ------------------------------------------------------------------ generation
    def next_logits(self, ctx: dict, input_ids: torch.Tensor) -> torch.Tensor:
        """Logits for the next token after `input_ids` (no padding inside generated sequences)."""
        if self.injection == "prefix":
            emb = torch.cat([ctx["prefix"], self.embed_tokens(input_ids)], dim=1)
            out = self.gpt(inputs_embeds=emb)
        else:
            out = self.gpt(input_ids=input_ids, encoder_hidden_states=ctx["enc"])
        return out.logits[:, -1, :].float()


# ---------------------------------------------------------------------- utilities
def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable": trainable, "total": total, "trainable_pct": round(100 * trainable / total, 3)}


def trainable_state_dict(model: nn.Module) -> dict:
    return {n: p.detach().cpu() for n, p in model.named_parameters() if p.requires_grad}


def build_model(cfg: dict) -> ClipCaptionModel:
    return ClipCaptionModel(gpt2_model=cfg["gpt2_model"], injection=cfg["injection"], finetune=cfg["finetune"],
                            feat_dim=cfg["feat_dim"], prefix_length=cfg.get("prefix_length", 10),
                            mapping_hidden=cfg.get("mapping_hidden"), lora=cfg.get("lora"))


def load_run(run_dir: str | Path, device: str = "cpu", checkpoint: str = "best.pt"):
    """Rebuild a trained model from runs/<name>/ (config.yaml + checkpoint). Returns (model, tokenizer, cfg)."""
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    model = build_model(cfg)
    ckpt = torch.load(run_dir / checkpoint, map_location="cpu")
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing = [n for n, p in model.named_parameters() if p.requires_grad and n not in state]
    if missing:
        raise RuntimeError(f"checkpoint lacks {len(missing)} trainable tensors, e.g. {missing[:3]}")
    model.load_state_dict(state, strict=False)
    return model.to(device).eval(), load_tokenizer(cfg["gpt2_model"]), cfg
