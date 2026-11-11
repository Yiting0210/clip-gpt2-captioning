"""Train a captioning model from a YAML config.

    python -m src.train --config configs/prefix_frozen.yaml            # full run -> runs/<experiment>/
    python -m src.train --config configs/prefix_frozen.yaml --debug    # 32 images, 3 steps, whole pipeline
    python -m src.train --config configs/prefix_frozen.yaml --overfit-batch   # sanity check: memorise 8 captions
    python -m src.train --config configs/prefix_frozen.yaml --resume   # continue from runs/<experiment>/last.pt
    python -m src.train --config configs/base.yaml --set finetune=lora train.lr=2e-4   # override any key

Outputs in runs/<experiment>/: config.yaml, train_log.jsonl (one JSON line per event, including
"preview" events = captions for fixed val images after every epoch), best.pt, last.pt, summary.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data import CaptionDataset, load_captions, load_embeddings, load_splits
from src.decode import generate
from src.model import _is_cross, build_model, count_parameters, load_tokenizer, trainable_state_dict


# ---------------------------------------------------------------------- config
def _merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in b.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path, overrides: list[str] | None = None) -> dict:
    """YAML with optional `base: other.yaml` inheritance and `a.b=value` overrides."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "base" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("base")), cfg)
    for ov in overrides or []:
        key, val = ov.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = _parse_value(val)
    return cfg


def _parse_value(val: str):
    v = yaml.safe_load(val)
    if isinstance(v, str):  # PyYAML reads "5e-5" as a string
        try:
            return float(v)
        except ValueError:
            pass
    return v


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(name: str | None = None) -> str:
    return name or ("cuda" if torch.cuda.is_available() else "cpu")


def autocast_dtype(device: str):
    if device != "cuda":
        return None
    # Native bf16 needs Ampere (compute capability 8.x) or newer; V100 (7.0) uses fp16 + GradScaler.
    return torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16


# ---------------------------------------------------------------------- data
def make_datasets(cfg: dict, tokenizer, debug: bool = False):
    caps, splits = load_captions(cfg["data_dir"]), load_splits(cfg["data_dir"])
    emb = load_embeddings(cfg["clip_model"], cfg["data_dir"])
    key = "pooled" if cfg["injection"] == "prefix" else "patch"
    if emb[key] is None:
        raise RuntimeError("patch features missing: rerun `python -m src.data embed` without --no-patch")
    feats = emb[key]
    cfg["feat_dim"] = int(feats.shape[-1])
    train_ids, val_ids = splits["train"], splits["val"]
    if cfg.get("train_images"):  # data-size experiment: the first N of the (already shuffled) training split
        train_ids = train_ids[:int(cfg["train_images"])]
    if debug:
        train_ids, val_ids = train_ids[:32], val_ids[:8]
    mt = cfg["max_caption_tokens"]
    train = CaptionDataset(train_ids, caps, feats, emb["index"], tokenizer, mt, "random")
    val = CaptionDataset(val_ids, caps, feats, emb["index"], tokenizer, mt, "all")
    preview = CaptionDataset(val_ids[:cfg["eval"]["num_preview"]], caps, feats, emb["index"], tokenizer, mt, "first")
    return train, val, preview


# ---------------------------------------------------------------------- helpers
class Logger:
    def __init__(self, path: Path):
        self.path = path

    def __call__(self, **event):
        event["time"] = round(time.time(), 1)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
        msg = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in event.items() if k != "time")
        print(msg, flush=True)


@torch.no_grad()
def evaluate_loss(model, loader, device, amp) -> float:
    model.eval()
    tot, n = 0.0, 0
    for b in loader:
        with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
            out = model(b["feats"].to(device), b["input_ids"].to(device),
                        b["attention_mask"].to(device), b["labels"].to(device))
        k = int((b["labels"] != -100).sum())
        tot, n = tot + out.loss.item() * k, n + k
    model.train()
    return tot / max(n, 1)


@torch.no_grad()
def preview_captions(model, preview_ds, tokenizer, device, cfg) -> list[dict]:
    model.eval()
    batch = preview_ds.collate([preview_ds[i] for i in range(len(preview_ds))])
    caps = generate(model, batch["feats"].to(device), tokenizer, cfg["eval"]["strategy"],
                    max_len=cfg["max_caption_tokens"])
    model.train()
    refs = [preview_ds.tok.decode(it[1][0]).strip() for it in preview_ds.items]
    return [{"image_id": i, "generated": c, "reference": r} for i, c, r in zip(batch["image_ids"], caps, refs)]


def param_groups(model, tc: dict) -> list[dict]:
    """Newly initialised weights (mapping network, cross-attention layers) at train.lr;
    pretrained GPT-2 weights at train.gpt_lr if given (useful for full fine-tuning)."""
    lr = float(tc["lr"])
    gpt_lr = float(tc.get("gpt_lr") or lr)
    is_new = lambda n: not n.startswith("gpt.") or _is_cross(n)
    mapper = [p for n, p in model.named_parameters() if p.requires_grad and is_new(n)]
    gpt = [p for n, p in model.named_parameters() if p.requires_grad and not is_new(n)]
    return [g for g in ({"params": mapper, "lr": lr}, {"params": gpt, "lr": gpt_lr}) if g["params"]]


def atomic_save(obj, path: Path) -> None:
    """Write to a temporary file, then rename: a full disk leaves the old file intact, never half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        torch.save(obj, tmp)
        tmp.replace(path)
    finally:
        tmp.unlink(missing_ok=True)


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, state: dict) -> None:
    atomic_save({"model": trainable_state_dict(model), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict() if scaler else None,
                 "state": state}, path)


# ---------------------------------------------------------------------- main loops
def train(cfg: dict, debug: bool = False, resume: bool = False, device: str | None = None) -> Path:
    device = pick_device(device)
    tc = cfg["train"]
    set_seed(tc["seed"])
    tokenizer = load_tokenizer(cfg["gpt2_model"])
    train_ds, val_ds, preview_ds = make_datasets(cfg, tokenizer, debug)

    name = cfg["experiment"] + ("_debug" if debug else "")
    out = Path(cfg["output_root"]) / name
    out.mkdir(parents=True, exist_ok=True)
    if resume and (out / "summary.json").exists() and (out / "best.pt").exists() and not (out / "last.pt").exists():
        print(f"{out} already finished (summary.json + best.pt, no last.pt); skipping training", flush=True)
        return out
    if resume and (out / "last.pt").exists():
        try:
            ck = torch.load(out / "last.pt", map_location="cpu")
        except Exception as e:  # e.g. written while the disk was full
            print(f"[warn] {out / 'last.pt'} unreadable ({e}); training this run from scratch", flush=True)
            (out / "last.pt").unlink()
            resume = False
    if not resume:
        (out / "train_log.jsonl").unlink(missing_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    log = Logger(out / "train_log.jsonl")

    model = build_model(cfg).to(device)
    params = count_parameters(model)
    bs = 4 if debug else tc["batch_size"]
    g = torch.Generator().manual_seed(tc["seed"])
    train_dl = DataLoader(train_ds, bs, shuffle=True, collate_fn=train_ds.collate,
                          num_workers=tc["num_workers"], generator=g, drop_last=False)
    val_dl = DataLoader(val_ds, bs * 2, shuffle=False, collate_fn=val_ds.collate, num_workers=tc["num_workers"])

    epochs = 1 if debug else tc["epochs"]
    max_steps = 3 if debug else None
    total_steps = max_steps or epochs * len(train_dl)
    optimizer = torch.optim.AdamW(param_groups(model, tc), weight_decay=tc["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, min(tc["warmup_steps"], total_steps // 10), total_steps)
    amp = autocast_dtype(device)
    scaler = torch.amp.GradScaler("cuda") if amp == torch.float16 else None

    state = {"epoch": 0, "step": 0, "best_val": math.inf, "bad_epochs": 0}
    if resume and (out / "last.pt").exists():
        model.load_state_dict(ck["model"], strict=False)
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        if scaler and ck["scaler"]:
            scaler.load_state_dict(ck["scaler"])
        state = ck["state"]
        log(event="resume", **state)

    log(event="start", experiment=name, device=device, injection=cfg["injection"], finetune=cfg["finetune"],
        train_images=len(train_ds), val_pairs=len(val_ds), steps_per_epoch=len(train_dl), **params)

    # Step-0 check: the evaluation path runs before any long training, so its bugs surface in a minute.
    if state["step"] == 0:
        log(event="val", epoch=0, step=0, val_loss=evaluate_loss(model, val_dl, device, amp))
        for p in preview_captions(model, preview_ds, tokenizer, device, cfg):
            log(event="preview", epoch=0, **p)

    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    model.train()
    first = state["epoch"] + 1
    if state["bad_epochs"] >= tc["early_stopping_patience"]:   # resumed a run that already stopped early
        log(event="already_finished", epoch=state["epoch"])
        first = epochs + 1
    for epoch in range(first, epochs + 1):
        t0, run_loss, n_batches = time.time(), 0.0, 0
        for b in train_dl:
            with torch.autocast("cuda", dtype=amp, enabled=amp is not None):
                loss = model(b["feats"].to(device), b["input_ids"].to(device),
                             b["attention_mask"].to(device), b["labels"].to(device)).loss
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], tc["grad_clip"])
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            scheduler.step()
            state["step"] += 1
            run_loss, n_batches = run_loss + loss.item(), n_batches + 1
            if state["step"] % tc["log_every"] == 0 or debug:
                log(event="train", epoch=epoch, step=state["step"], loss=loss.item(), lr=scheduler.get_last_lr()[0])
            if max_steps and state["step"] >= max_steps:
                break

        epoch_time = time.time() - t0
        val_loss = evaluate_loss(model, val_dl, device, amp)
        state["epoch"] = epoch
        improved = val_loss < state["best_val"]
        if improved:
            state["best_val"], state["bad_epochs"] = val_loss, 0
        else:
            state["bad_epochs"] += 1
        peak = torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None
        log(event="val", epoch=epoch, step=state["step"], train_loss=run_loss / max(n_batches, 1),
            val_loss=val_loss, epoch_time_s=round(epoch_time, 1), peak_mem_gb=peak, improved=improved)
        for p in preview_captions(model, preview_ds, tokenizer, device, cfg):
            log(event="preview", epoch=epoch, **p)

        save_checkpoint(out / "last.pt", model, optimizer, scheduler, scaler, state)
        if improved:
            atomic_save({"model": trainable_state_dict(model), "state": state}, out / "best.pt")
        if state["bad_epochs"] >= tc["early_stopping_patience"]:
            log(event="early_stop", epoch=epoch)
            break

    events = [json.loads(l) for l in (out / "train_log.jsonl").read_text(encoding="utf-8").splitlines()]
    epoch_times = [e["epoch_time_s"] for e in events if e.get("event") == "val" and "epoch_time_s" in e]
    summary = {"experiment": name, "injection": cfg["injection"], "finetune": cfg["finetune"], **params,
               "best_val_loss": state["best_val"], "epochs_run": state["epoch"],
               "mean_epoch_time_s": round(sum(epoch_times) / max(len(epoch_times), 1), 1),
               "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else None,
               "device": device}
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not tc.get("keep_last", False):
        # last.pt (weights + optimizer state, ~1.9 GB for full fine-tuning) only serves to resume an unfinished run
        (out / "last.pt").unlink(missing_ok=True)
    log(event="done", best_val_loss=state["best_val"], run_dir=str(out))
    return out


def overfit_batch(cfg: dict, steps: int = 300, n: int = 8, device: str | None = None) -> float:
    """Train repeatedly on one fixed batch. A correct model drives the loss toward 0 and reproduces the captions."""
    device = pick_device(device)
    set_seed(cfg["train"]["seed"])
    tokenizer = load_tokenizer(cfg["gpt2_model"])
    train_ds, _, _ = make_datasets(cfg, tokenizer, debug=True)
    train_ds.choice = "first"
    batch = train_ds.collate([train_ds[i] for i in range(min(n, len(train_ds)))])
    model = build_model(cfg).to(device).train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=float(cfg["train"]["overfit_lr"]))
    b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
    for step in range(1, steps + 1):
        loss = model(b["feats"], b["input_ids"], b["attention_mask"], b["labels"]).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step == 1 or step % 25 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}", flush=True)
    model.eval()
    gen = generate(model, b["feats"], tokenizer, "greedy", max_len=cfg["max_caption_tokens"])
    exact = 0
    for i, g in enumerate(gen):
        ref = tokenizer.decode(train_ds.items[i][1][0]).strip()
        exact += g == ref
        print(f"[{'OK ' if g == ref else 'DIFF'}] gen: {g}\n       ref: {ref}")
    print(f"final loss {loss.item():.4f}; exact matches {exact}/{len(gen)}")
    return loss.item()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--set", nargs="*", default=[], help="overrides like train.lr=2e-4 finetune=lora")
    p.add_argument("--debug", action="store_true", help="tiny subset, 3 steps, whole pipeline")
    p.add_argument("--overfit-batch", action="store_true", help="memorise one batch (sanity check)")
    p.add_argument("--overfit-steps", type=int, default=300)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--device")
    a = p.parse_args()
    cfg = load_config(a.config, a.set)
    if a.overfit_batch:
        overfit_batch(cfg, a.overfit_steps, device=a.device)
    else:
        train(cfg, debug=a.debug, resume=a.resume, device=a.device)


if __name__ == "__main__":
    main()
