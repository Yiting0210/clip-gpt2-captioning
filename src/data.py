"""Data pipeline: download COCO val2017, clean captions, split, precompute CLIP embeddings.

Usage (from the repo root):
    python -m src.data download [--limit 64]      # annotations + images (limit = only N images, for debugging)
    python -m src.data prepare  [--limit 64]      # clean captions, verify images, train/val/test split
    python -m src.data embed    [--device cuda]   # CLIP image embeddings -> data/processed/clip_*.pt
    python -m src.data stats                      # caption length statistics in GPT-2 tokens
"""
from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from pathlib import Path

import requests
import torch
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
IMG_ZIP_URL = "http://images.cocodataset.org/zips/val2017.zip"
ANN_MEMBERS = ["annotations/captions_val2017.json", "annotations/instances_val2017.json"]

DATA_ROOT = Path("data")
RAW_DIR = DATA_ROOT / "raw"
IMG_DIR = RAW_DIR / "val2017"
PROC_DIR = DATA_ROOT / "processed"


# --------------------------------------------------------------------------- download
def _download(url: str, dest: Path) -> Path:
    """Stream a file to disk; skip if it already exists with the right size."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        if dest.exists() and total and dest.stat().st_size == total:
            return dest
        tmp = dest.with_suffix(dest.suffix + ".part")
        with open(tmp, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=dest.name) as bar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                bar.update(len(chunk))
        tmp.replace(dest)
    return dest


def download(limit: int | None = None, seed: int = 42) -> None:
    ann_ok = all((RAW_DIR / m).exists() for m in ANN_MEMBERS)
    if not ann_ok:
        zpath = _download(ANN_URL, RAW_DIR / "annotations_trainval2017.zip")
        with zipfile.ZipFile(zpath) as z:
            for m in ANN_MEMBERS:
                z.extract(m, RAW_DIR)
        print("annotations ready:", [str(RAW_DIR / m) for m in ANN_MEMBERS])

    if limit is None:
        if IMG_DIR.exists() and len(list(IMG_DIR.glob("*.jpg"))) >= 5000:
            print("images already present")
            return
        zpath = _download(IMG_ZIP_URL, RAW_DIR / "val2017.zip")
        with zipfile.ZipFile(zpath) as z:
            z.extractall(RAW_DIR)
        print("images ready:", IMG_DIR)
        return

    # Debug mode: fetch only `limit` images one by one via their coco_url.
    coco = json.loads((RAW_DIR / ANN_MEMBERS[0]).read_text(encoding="utf-8"))
    images = sorted(coco["images"], key=lambda im: im["id"])
    random.Random(seed).shuffle(images)
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for im in tqdm(images[:limit], desc="images"):
        dest = IMG_DIR / im["file_name"]
        if not dest.exists():
            r = requests.get(im["coco_url"], timeout=60)
            r.raise_for_status()
            dest.write_bytes(r.content)


# --------------------------------------------------------------------------- cleaning
def clean_caption(text: str) -> str:
    """Normalise whitespace and end every caption with exactly one period."""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = text.rstrip(" .")
    if not text:
        return ""
    return text[0].upper() + text[1:] + "."


def _image_ok(path: Path) -> tuple[bool, str]:
    try:
        with Image.open(path) as im:
            mode = im.mode
            im.verify()
        return True, mode
    except Exception:
        return False, ""


def prepare(limit: int | None = None, seed: int = 42, val_frac: float = 0.1, test_frac: float = 0.1) -> dict:
    coco = json.loads((RAW_DIR / ANN_MEMBERS[0]).read_text(encoding="utf-8"))
    caps: dict[int, list[str]] = {}
    for a in coco["annotations"]:
        caps.setdefault(a["image_id"], []).append(a["caption"])

    report = {"raw_images": len(coco["images"]), "raw_captions": len(coco["annotations"]),
              "missing_files": 0, "corrupt_images": 0, "non_rgb_images": 0,
              "empty_captions": 0, "duplicate_captions": 0}
    records = {}
    for im in tqdm(sorted(coco["images"], key=lambda x: x["id"]), desc="verify"):
        path = IMG_DIR / im["file_name"]
        if not path.exists():
            report["missing_files"] += 1
            continue
        ok, mode = _image_ok(path)
        if not ok:
            report["corrupt_images"] += 1
            continue
        if mode != "RGB":
            report["non_rgb_images"] += 1  # converted to RGB at embedding time
        seen, cleaned = set(), []
        for c in caps.get(im["id"], []):
            c2 = clean_caption(c)
            if not c2:
                report["empty_captions"] += 1
                continue
            if c2.lower() in seen:
                report["duplicate_captions"] += 1
                continue
            seen.add(c2.lower())
            cleaned.append(c2)
        if cleaned:
            records[str(im["id"])] = {"file_name": im["file_name"], "coco_url": im["coco_url"],
                                      "height": im["height"], "width": im["width"], "captions": cleaned}

    ids = sorted(records, key=int)
    random.Random(seed).shuffle(ids)
    if limit is not None:
        ids = ids[:limit]
    n_val = max(1, round(len(ids) * val_frac))
    n_test = max(1, round(len(ids) * test_frac))
    splits = {"val": ids[:n_val], "test": ids[n_val:n_val + n_test], "train": ids[n_val + n_test:]}
    records = {i: records[i] for i in ids}

    report.update({"kept_images": len(ids), "kept_captions": sum(len(r["captions"]) for r in records.values()),
                   **{f"{k}_images": len(v) for k, v in splits.items()}})
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    (PROC_DIR / "captions.json").write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    (PROC_DIR / "splits.json").write_text(json.dumps(splits), encoding="utf-8")
    (PROC_DIR / "prepare_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


# --------------------------------------------------------------------------- loading helpers
def load_captions(proc_dir: Path = PROC_DIR) -> dict:
    return json.loads((Path(proc_dir) / "captions.json").read_text(encoding="utf-8"))


def load_splits(proc_dir: Path = PROC_DIR) -> dict:
    return json.loads((Path(proc_dir) / "splits.json").read_text(encoding="utf-8"))


def embedding_path(clip_model: str, proc_dir: Path = PROC_DIR) -> Path:
    return Path(proc_dir) / f"clip_{clip_model.split('/')[-1]}.pt"


def load_embeddings(clip_model: str, proc_dir: Path = PROC_DIR) -> dict:
    """Returns {"image_ids": [str], "index": {id: row}, "pooled": (N,512), "patch": (N,50,768) or None}."""
    d = torch.load(embedding_path(clip_model, proc_dir), map_location="cpu")
    d["index"] = {iid: i for i, iid in enumerate(d["image_ids"])}
    return d


def caption_length_stats(captions: dict, tokenizer, max_tokens: int = 40) -> dict:
    lengths = sorted(len(tokenizer.encode(" " + c)) for r in captions.values() for c in r["captions"])
    n = len(lengths)
    pct = lambda p: lengths[min(n - 1, int(p * n))]
    return {"n_captions": n, "mean": round(sum(lengths) / n, 2), "p50": pct(0.5), "p95": pct(0.95),
            "p99": pct(0.99), "max": lengths[-1], f"over_{max_tokens}": sum(l > max_tokens for l in lengths)}


# --------------------------------------------------------------------------- embeddings
@torch.no_grad()
def embed(clip_model: str = "openai/clip-vit-base-patch32", device: str | None = None,
          batch_size: int = 64, save_patch: bool = True) -> Path:
    from transformers import CLIPImageProcessor, CLIPModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    records = load_captions()
    ids = sorted(records, key=int)
    model = CLIPModel.from_pretrained(clip_model).to(device).eval()
    proc = CLIPImageProcessor.from_pretrained(clip_model)

    pooled, patch = [], []
    for s in tqdm(range(0, len(ids), batch_size), desc="CLIP"):
        batch = ids[s:s + batch_size]
        imgs = [Image.open(IMG_DIR / records[i]["file_name"]).convert("RGB") for i in batch]
        px = proc(images=imgs, return_tensors="pt")["pixel_values"].to(device)
        vis = model.vision_model(pixel_values=px)
        pooled.append(model.visual_projection(vis.pooler_output).float().cpu().half())
        if save_patch:
            patch.append(vis.last_hidden_state.float().cpu().half())

    out = {"image_ids": ids, "clip_model": clip_model, "pooled": torch.cat(pooled),
           "patch": torch.cat(patch) if save_patch else None}
    path = embedding_path(clip_model)
    torch.save(out, path)
    print(f"saved {path}: pooled {tuple(out['pooled'].shape)}"
          + (f", patch {tuple(out['patch'].shape)}" if save_patch else ""))
    return path


# --------------------------------------------------------------------------- torch dataset
class CaptionDataset(Dataset):
    """Pairs a precomputed CLIP feature with a tokenized caption.

    caption_choice:
      "random" - one random caption per image each time it is drawn (train; len = #images)
      "first"  - always the first caption (overfit test, deterministic previews)
      "all"    - every (image, caption) pair (validation loss)
    Token layout:  input_ids = [bos] + caption + [eos];  labels = [-100] + caption + [eos]
    (GPT-2 uses <|endoftext|> for both bos and eos.)
    """

    def __init__(self, image_ids, captions, features, feat_index, tokenizer,
                 max_tokens: int = 40, caption_choice: str = "random"):
        self.tok = tokenizer
        self.features, self.feat_index = features, feat_index
        self.choice = caption_choice
        self.bos = self.eos = tokenizer.eos_token_id
        self.items = []  # (image_id, [token lists])
        for iid in image_ids:
            toks = [tokenizer.encode(" " + c) for c in captions[iid]["captions"]]
            toks = [t for t in toks if len(t) <= max_tokens] or [toks[0][:max_tokens]]
            self.items.append((iid, toks))
        self.index = ([(i, j) for i, (_, t) in enumerate(self.items) for j in range(len(t))]
                      if caption_choice == "all" else None)

    def __len__(self):
        return len(self.index) if self.index is not None else len(self.items)

    def __getitem__(self, k):
        if self.index is not None:
            i, j = self.index[k]
        else:
            i = k
            j = random.randrange(len(self.items[i][1])) if self.choice == "random" else 0
        iid, toks = self.items[i]
        cap = toks[j]
        return {"image_id": iid,
                "feat": self.features[self.feat_index[iid]].float(),
                "input_ids": torch.tensor([self.bos] + cap + [self.eos]),
                "labels": torch.tensor([-100] + cap + [self.eos])}

    def collate(self, batch):
        L = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), L), self.eos, dtype=torch.long)
        labels = torch.full((len(batch), L), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), L), dtype=torch.long)
        for r, b in enumerate(batch):
            n = len(b["input_ids"])
            ids[r, :n], labels[r, :n], mask[r, :n] = b["input_ids"], b["labels"], 1
        return {"image_ids": [b["image_id"] for b in batch], "feats": torch.stack([b["feat"] for b in batch]),
                "input_ids": ids, "attention_mask": mask, "labels": labels}


# --------------------------------------------------------------------------- CLI
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download"); d.add_argument("--limit", type=int); d.add_argument("--seed", type=int, default=42)
    pr = sub.add_parser("prepare"); pr.add_argument("--limit", type=int); pr.add_argument("--seed", type=int, default=42)
    e = sub.add_parser("embed")
    e.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    e.add_argument("--device"); e.add_argument("--batch-size", type=int, default=64)
    e.add_argument("--no-patch", action="store_true", help="skip patch features (cross-attention needs them)")
    s = sub.add_parser("stats"); s.add_argument("--gpt2-model", default="gpt2"); s.add_argument("--max-tokens", type=int, default=40)
    a = p.parse_args()

    if a.cmd == "download":
        download(a.limit, a.seed)
    elif a.cmd == "prepare":
        prepare(a.limit, a.seed)
    elif a.cmd == "embed":
        embed(a.clip_model, a.device, a.batch_size, not a.no_patch)
    elif a.cmd == "stats":
        from transformers import GPT2TokenizerFast
        tok = GPT2TokenizerFast.from_pretrained(a.gpt2_model)
        print(json.dumps(caption_length_stats(load_captions(), tok, a.max_tokens), indent=2))


if __name__ == "__main__":
    main()
