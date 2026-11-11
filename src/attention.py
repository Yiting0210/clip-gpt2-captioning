"""Where does a cross-attention model look while it writes each word?

    python -m src.attention --runs runs/xattn_lora_s0 runs/xattn_full_s0 --figure 6

For every test image the model writes a greedy caption; the cross-attention weights of each generated
token over the 7x7 CLIP patches (CLS dropped, averaged over heads and layers) are read off a second
forward pass. Two outputs per run:

  results/attention/<run>_heatmaps.png   images in rows, one heatmap per content word
  results/attention/<run>_grounding.json for every caption word that names a COCO object present in the
                                         image: the share of attention inside that object's boxes,
                                         against the share of the image the boxes cover (what uniform
                                         attention would give). ratio > 1 = the model looks at the object.
"""
from __future__ import annotations

import argparse
import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.data import IMG_DIR, RAW_DIR, load_captions, load_embeddings, load_splits
from src.decode import greedy
from src.evaluate import RESULTS, _synonym_table
from src.model import load_run

GRID, SIZE = 7, 224   # ViT-B/32 at 224 px -> 7x7 patches
STOP = set("a an the of on in with and is are to at its it his her their near next by for from top front "
           "some two three while that this there".split())


# ---------------------------------------------------------------------- image geometry (CLIP preprocessing)
def crop_box(w: int, h: int) -> tuple[float, float, float]:
    """CLIP resizes the short side to 224 and centre-crops. Returns (scale, left, top) in original pixels."""
    s = SIZE / min(w, h)
    return s, (w * s - SIZE) / 2, (h * s - SIZE) / 2


def clip_view(path: Path):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    s, left, top = crop_box(*im.size)
    im = im.resize((round(im.width * s), round(im.height * s)))
    return im.crop((round(left), round(top), round(left) + SIZE, round(top) + SIZE))


def box_coverage(bboxes, w: int, h: int) -> np.ndarray:
    """7x7 array: share of each patch covered by the union of the boxes (COCO [x, y, bw, bh])."""
    s, left, top = crop_box(w, h)
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    for x, y, bw, bh in bboxes:
        x0, y0 = int(max(0, x * s - left)), int(max(0, y * s - top))
        x1, y1 = int(min(SIZE, (x + bw) * s - left)), int(min(SIZE, (y + bh) * s - top))
        if x1 > x0 and y1 > y0:
            mask[y0:y1, x0:x1] = True
    p = SIZE // GRID
    return mask.reshape(GRID, p, GRID, p).mean(axis=(1, 3))


@lru_cache(maxsize=1)
def instance_boxes() -> dict[str, dict[str, list]]:
    """image id -> category name -> list of boxes."""
    d = json.loads((RAW_DIR / "annotations" / "instances_val2017.json").read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in d["categories"]}
    out: dict[str, dict[str, list]] = {}
    for a in d["annotations"]:
        out.setdefault(str(a["image_id"]), {}).setdefault(names[a["category_id"]], []).append(a["bbox"])
    return out


# ---------------------------------------------------------------------- attention
def _set_eager(model) -> None:
    """SDPA kernels do not return attention weights; switch GPT-2 to the eager implementation."""
    for m in model.gpt.modules():
        cfg = getattr(m, "config", None)
        if cfg is not None and hasattr(cfg, "_attn_implementation"):
            cfg._attn_implementation = "eager"


@torch.no_grad()
def caption_attention(model, tok, feats: torch.Tensor, max_len: int = 40):
    """Greedy captions + per-token attention maps. Returns a list of (words, maps) per image,
    maps = array (n_words, 7, 7); a word's map averages its BPE pieces."""
    seqs = greedy(model, feats, tok, max_len, return_ids=True)            # (B, T): bos, tokens, eos...
    ctx = model.encode_context(feats)
    out = model.gpt(input_ids=seqs, encoder_hidden_states=ctx["enc"], output_attentions=True)
    if not out.cross_attentions or out.cross_attentions[0] is None:
        raise RuntimeError("model returned no cross-attention weights")
    att = torch.stack(out.cross_attentions).float().mean(dim=(0, 2))     # (B, T, 50): mean over layers, heads
    att = att[..., 1:]                                                   # drop CLS
    att = att / att.sum(-1, keepdim=True)
    results = []
    eos = tok.eos_token_id
    for b, row in enumerate(seqs.tolist()):
        words, maps, cur, cur_maps = [], [], "", []
        for k in range(1, len(row)):                  # generated token k was predicted at query position k-1
            if row[k] == eos:
                break
            piece = tok.convert_ids_to_tokens(row[k])
            m = att[b, k - 1].cpu().numpy().reshape(GRID, GRID)
            if piece.startswith("Ġ") and cur:         # a new word starts
                words.append(cur)
                maps.append(np.mean(cur_maps, axis=0))
                cur, cur_maps = "", []
            cur += piece.lstrip("Ġ")
            cur_maps.append(m)
        if cur:
            words.append(cur)
            maps.append(np.mean(cur_maps, axis=0))
        results.append((words, np.array(maps)))
    return results


def grounding(ids, caps, att_results) -> dict:
    """Attention inside the boxes of objects the caption names correctly vs the area those boxes cover."""
    uni, _ = _synonym_table()
    boxes = instance_boxes()
    rows = []
    for iid, (words, maps) in zip(ids, att_results):
        w, h = caps[iid]["width"], caps[iid]["height"]
        for word, m in zip(words, maps):
            cat = uni.get("".join(ch for ch in word.lower() if ch.isalpha()))
            if not cat or cat not in boxes.get(iid, {}):
                continue
            cov = box_coverage(boxes[iid][cat], w, h)
            if cov.mean() <= 0:
                continue
            inside = float((m * cov).sum())
            rows.append({"image_id": iid, "word": word, "category": cat, "attention_in_box": round(inside, 4),
                         "box_area": round(float(cov.mean()), 4), "ratio": round(inside / float(cov.mean()), 3)})
    ratios = [r["ratio"] for r in rows]
    summary = {"grounded_words": len(rows),
               "mean_attention_in_box": round(float(np.mean([r["attention_in_box"] for r in rows])), 4) if rows else None,
               "mean_box_area": round(float(np.mean([r["box_area"] for r in rows])), 4) if rows else None,
               "median_ratio": round(float(np.median(ratios)), 3) if rows else None,
               "share_ratio_above_1": round(float(np.mean([x > 1 for x in ratios])), 4) if rows else None}
    return {"summary": summary, "words": rows}


def heatmap_figure(ids, caps, att_results, path: Path, n_words: int = 6) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(ids), n_words + 1, figsize=(2.1 * (n_words + 1), 2.3 * len(ids)))
    axes = np.atleast_2d(axes)
    for r, (iid, (words, maps)) in enumerate(zip(ids, att_results)):
        img = np.asarray(clip_view(IMG_DIR / caps[iid]["file_name"]))
        axes[r, 0].imshow(img)
        axes[r, 0].set_title(" ".join(words)[:38], fontsize=7, loc="left")
        keep = [k for k, w in enumerate(words) if w.lower().strip(".,") not in STOP][:n_words]
        for c in range(1, n_words + 1):
            ax = axes[r, c]
            ax.axis("off")
            if c - 1 >= len(keep):
                continue
            k = keep[c - 1]
            m = np.kron(maps[k], np.ones((SIZE // GRID, SIZE // GRID)))
            ax.imshow(img)
            ax.imshow(m, cmap="magma", alpha=0.55, extent=(0, SIZE, SIZE, 0), interpolation="bilinear")
            ax.set_title(words[k], fontsize=9)
        axes[r, 0].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)
    return path


def run(run_dir: Path, split: str, figure: int, batch_size: int, device: str) -> dict:
    model, tok, cfg = load_run(run_dir, device)
    if cfg["injection"] != "cross_attention":
        raise ValueError(f"{run_dir} is a prefix model; attention maps need a cross-attention run")
    _set_eager(model)
    caps, splits = load_captions(cfg["data_dir"]), load_splits(cfg["data_dir"])
    emb = load_embeddings(cfg["clip_model"], cfg["data_dir"])
    ids = splits[split]
    results = []
    for s in range(0, len(ids), batch_size):
        b = ids[s:s + batch_size]
        feats = emb["patch"][[emb["index"][i] for i in b]].float().to(device)
        results += caption_attention(model, tok, feats, cfg["max_caption_tokens"])
    g = grounding(ids, caps, results)
    out = RESULTS / "attention"
    out.mkdir(parents=True, exist_ok=True)
    g["run"], g["split"] = run_dir.name, split
    (out / f"{run_dir.name}_grounding.json").write_text(json.dumps(g, indent=1), encoding="utf-8")
    print(run_dir.name, json.dumps(g["summary"]))
    if figure:
        print("figure:", heatmap_figure(ids[:figure], caps, results[:figure], out / f"{run_dir.name}_heatmaps.png"))
    return g["summary"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--figure", type=int, default=6, help="images in the heatmap figure (0 = none)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--device")
    a = p.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    for r in a.runs:
        run(Path(r), a.split, a.figure, a.batch_size, device)


if __name__ == "__main__":
    main()
