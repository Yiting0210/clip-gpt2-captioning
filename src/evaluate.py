"""Evaluate trained runs and baselines on the val/test split.

    python -m src.evaluate --runs runs/prefix_frozen --decode greedy beam_w3 nucleus_p0.9_t0.7
    python -m src.evaluate --runs runs/prefix_frozen --decode sweep        # beam-width + temperature sweep
    python -m src.evaluate --baseline retrieval                             # CLIP nearest-neighbour captions
    python -m src.evaluate --baseline blip2                                 # zero-shot BLIP-2 (GPU)
    python -m src.evaluate --runs runs/prefix_frozen_debug --limit 8        # quick check

Writes results/<name>/<decode>.json (metrics + every caption) and upserts one row per
(name, decode, split) into results/metrics.csv, which notebooks/03_evaluation.ipynb reads.

Metrics: BLEU-1..4, METEOR, ROUGE-L, CIDEr (pycocoevalcap; METEOR and the PTB tokenizer need Java),
plus CHAIR-i / CHAIR-s object hallucination, gender-word error rate, CLIPScore, diversity and repetition.
CIDEr, CLIPScore and CHAIR-s also get a 95% bootstrap interval over images (columns <metric>_lo / _hi);
the per-image scores are saved in the JSON so two systems can be compared with a paired bootstrap.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from src.data import PROC_DIR, RAW_DIR, IMG_DIR, load_captions, load_embeddings, load_splits
from src.decode import generate_batched

RESULTS = Path("results")
SWEEP = ["greedy", "beam_w1", "beam_w3", "beam_w5", "beam_w10",
         "nucleus_p0.9_t0.3", "nucleus_p0.9_t0.5", "nucleus_p0.9_t0.7", "nucleus_p0.9_t1.0",
         "nucleus_p0.5_t1.0", "nucleus_p0.8_t1.0", "nucleus_p0.95_t1.0",
         "greedy_nr3", "beam_w3_nr3", "beam_w5_nr3", "beam_w10_nr3"]
BOOT = 1000


# ---------------------------------------------------------------------- caption metrics
def _simple_tokenize(caps: dict) -> dict:
    return {k: [" ".join(re.findall(r"[a-z0-9]+", c.lower())) for c in v] for k, v in caps.items()}


def coco_metrics(gts: dict[str, list[str]], res: dict[str, str], meteor: bool = True,
                 per_image: dict | None = None) -> dict:
    """Corpus-level scores. If `per_image` is a dict, per-image CIDEr is stored in it (same order as gts)."""
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    ids = list(gts)
    g_raw = {i: gts[i] for i in ids}
    r_raw = {i: [res[i] or "."] for i in ids}
    try:
        from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
        tok = PTBTokenizer()
        g = tok.tokenize({i: [{"caption": c} for c in v] for i, v in g_raw.items()})
        r = tok.tokenize({i: [{"caption": c} for c in v] for i, v in r_raw.items()})
    except Exception as e:  # no Java -> plain lowercase tokenization (scores differ slightly from official)
        print(f"[warn] PTBTokenizer failed ({e}); using simple tokenizer")
        g, r = _simple_tokenize(g_raw), _simple_tokenize(r_raw)

    out = {}
    bleu, _ = Bleu(4).compute_score(g, r, verbose=0)
    out.update({f"BLEU-{n + 1}": bleu[n] for n in range(4)})
    if meteor:
        try:
            from pycocoevalcap.meteor.meteor import Meteor
            m = Meteor()
            out["METEOR"], _ = m.compute_score(g, r)
            del m
        except Exception as e:
            print(f"[warn] METEOR failed ({e}); skipped")
    out["ROUGE-L"], _ = Rouge().compute_score(g, r)
    out["CIDEr"], cider_each = Cider().compute_score(g, r)
    if per_image is not None:
        per_image["CIDEr"] = [round(float(x), 4) for x in cider_each]
    return {k: round(float(v), 4) for k, v in out.items()}


# ---------------------------------------------------------------------- CHAIR (simplified)
SYNONYMS = {
    "person": "person persons people man men woman women boy boys girl girls child children kid kids guy guys lady "
              "ladies baby babies adult adults player players skier skiers surfer surfers skateboarder rider riders "
              "gentleman crowd teenager toddler someone",
    "bicycle": "bicycle bicycles bike bikes cycle", "car": "car cars sedan taxi suv",
    "motorcycle": "motorcycle motorcycles motorbike motorbikes scooter", "airplane": "airplane airplanes plane "
                  "planes jet jets aircraft airliner", "bus": "bus buses", "train": "train trains locomotive",
    "truck": "truck trucks pickup", "boat": "boat boats ship ships sailboat kayak canoe yacht ferry",
    "traffic light": "traffic light, traffic signal, stoplight", "fire hydrant": "fire hydrant, hydrant",
    "stop sign": "stop sign", "parking meter": "parking meter", "bench": "bench benches",
    "bird": "bird birds pigeon seagull duck ducks goose", "cat": "cat cats kitten kitty",
    "dog": "dog dogs puppy pup", "horse": "horse horses pony", "sheep": "sheep lamb lambs",
    "cow": "cow cows cattle bull calf", "elephant": "elephant elephants", "bear": "bear bears",
    "zebra": "zebra zebras", "giraffe": "giraffe giraffes", "backpack": "backpack backpacks",
    "umbrella": "umbrella umbrellas parasol", "handbag": "handbag handbags purse purses",
    "tie": "tie ties necktie", "suitcase": "suitcase suitcases luggage", "frisbee": "frisbee frisbees",
    "skis": "skis ski", "snowboard": "snowboard snowboards", "sports ball": "ball, balls, soccer ball, football",
    "kite": "kite kites", "baseball bat": "baseball bat, bat", "baseball glove": "baseball glove, mitt, glove",
    "skateboard": "skateboard skateboards", "surfboard": "surfboard surfboards",
    "tennis racket": "tennis racket, racket, racquet", "bottle": "bottle bottles",
    "wine glass": "wine glass, wineglass", "cup": "cup cups mug mugs", "fork": "fork forks",
    "knife": "knife knives", "spoon": "spoon spoons", "bowl": "bowl bowls", "banana": "banana bananas",
    "apple": "apple apples", "sandwich": "sandwich sandwiches burger hamburger sub",
    "orange": "oranges", "broccoli": "broccoli", "carrot": "carrot carrots", "hot dog": "hot dog, hotdog",
    "pizza": "pizza pizzas", "donut": "donut donuts doughnut doughnuts", "cake": "cake cakes cupcake",
    "chair": "chair chairs stool", "couch": "couch couches sofa sofas", "potted plant": "potted plant, houseplant",
    "bed": "bed beds", "dining table": "dining table, table, tables, desk", "toilet": "toilet toilets",
    "tv": "tv tvs television televisions", "laptop": "laptop laptops computer notebook",
    "mouse": "mouse", "remote": "remote remotes controller", "keyboard": "keyboard keyboards",
    "cell phone": "cell phone, phone, phones, cellphone, smartphone", "microwave": "microwave microwaves",
    "oven": "oven ovens stove", "toaster": "toaster", "sink": "sink sinks", "refrigerator": "refrigerator "
    "refrigerators fridge", "book": "book books", "clock": "clock clocks", "vase": "vase vases",
    "scissors": "scissors", "teddy bear": "teddy bear, teddy", "hair drier": "hair drier, hair dryer, blow dryer",
    "toothbrush": "toothbrush toothbrushes",
}
# Format: single-word categories list space-separated words; multi-word categories list
# comma-separated phrases (one or two words each).


@lru_cache(maxsize=1)
def _synonym_table() -> tuple[dict, dict]:
    uni, bi = {}, {}
    for cat, spec in SYNONYMS.items():
        phrases = [s.strip() for s in spec.split(",")] if " " in cat else spec.split()
        for ph in phrases:
            (bi if " " in ph else uni)[ph] = cat
    return uni, bi


def mentioned_objects(caption: str) -> set[str]:
    uni, bi = _synonym_table()
    toks = re.findall(r"[a-z]+", caption.lower())
    found, i = set(), 0
    while i < len(toks):
        if i + 1 < len(toks):
            pair = f"{toks[i]} {toks[i + 1]}"
            hit = next((bi[c] for c in (pair, pair[:-1], pair[:-2]) if c in bi), None)  # plural "s"/"es"
            if hit:
                found.add(hit)
                i += 2
                continue
        if toks[i] in uni:
            found.add(uni[toks[i]])
        i += 1
    return found


@lru_cache(maxsize=1)
def _instance_objects() -> dict[str, set[str]]:
    path = RAW_DIR / "annotations" / "instances_val2017.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text(encoding="utf-8"))
    names = {c["id"]: c["name"] for c in d["categories"]}
    out: dict[str, set[str]] = {}
    for a in d["annotations"]:
        out.setdefault(str(a["image_id"]), set()).add(names[a["category_id"]])
    return out


def chair(res: dict[str, str], gts: dict[str, list[str]], per_image: dict | None = None) -> dict:
    """CHAIR (Rohrbach et al. 2018), simplified synonym list. Ground truth = COCO instance labels
    plus objects named in the reference captions."""
    inst = _instance_objects()
    mentions = halluc = bad_sents = 0
    flags = []
    for iid, cap in res.items():
        truth = set(inst.get(iid, set()))
        for ref in gts[iid]:
            truth |= mentioned_objects(ref)
        m = mentioned_objects(cap)
        h = m - truth
        mentions, halluc, bad_sents = mentions + len(m), halluc + len(h), bad_sents + bool(h)
        flags.append(int(bool(h)))
    if per_image is not None:
        per_image["CHAIR-s"] = flags
    return {"CHAIR-i": round(halluc / max(mentions, 1), 4), "CHAIR-s": round(bad_sents / max(len(res), 1), 4)}


# ---------------------------------------------------------------------- gender words
MALE = set("man men boy boys gentleman gentlemen guy guys male males he his him himself father dad husband son "
           "brother businessman".split())
FEMALE = set("woman women girl girls lady ladies female females she her hers herself mother mom wife daughter "
             "sister businesswoman".split())


def gender_of(text: str) -> str:
    toks = set(re.findall(r"[a-z]+", text.lower()))
    m, f = bool(toks & MALE), bool(toks & FEMALE)
    return "both" if m and f else "male" if m else "female" if f else "none"


def gender_eval(res: dict[str, str], gts: dict[str, list[str]], min_agree: int = 3) -> dict:
    """Images whose references agree on one gender (>= min_agree refs, none for the other).
    error = caption uses the opposite gender word; neutral = no gender word."""
    n = wrong = neutral = 0
    for iid, cap in res.items():
        g = [gender_of(r) for r in gts[iid]]
        label = ("male" if g.count("male") >= min_agree and "female" not in g and "both" not in g else
                 "female" if g.count("female") >= min_agree and "male" not in g and "both" not in g else None)
        if label is None:
            continue
        pred = gender_of(cap)
        n += 1
        wrong += pred in ({"female"} if label == "male" else {"male"})
        neutral += pred == "none"
    return {"gender_images": n, "gender_error": round(wrong / max(n, 1), 4),
            "gender_neutral": round(neutral / max(n, 1), 4)}


# ---------------------------------------------------------------------- CLIPScore
@lru_cache(maxsize=1)
def _clip_text(clip_model: str, device: str):
    from transformers import CLIPModel, CLIPTokenizerFast
    return CLIPModel.from_pretrained(clip_model).to(device).eval(), CLIPTokenizerFast.from_pretrained(clip_model)


@torch.no_grad()
def clip_score(res: dict[str, str], img_emb: dict[str, torch.Tensor], clip_model: str, device: str,
               per_image: dict | None = None) -> dict:
    """CLIPScore (Hessel et al. 2021) = 2.5 * max(cos(image, caption), 0), reference-free."""
    model, tok = _clip_text(clip_model, device)
    ids = list(res)
    scores = []
    for s in range(0, len(ids), 64):
        b = ids[s:s + 64]
        t = tok([res[i] or "." for i in b], padding=True, truncation=True, max_length=77, return_tensors="pt").to(device)
        te = torch.nn.functional.normalize(model.get_text_features(**t).float(), dim=-1)
        ie = torch.nn.functional.normalize(torch.stack([img_emb[i] for i in b]).float().to(device), dim=-1)
        scores.append(2.5 * (te * ie).sum(-1).clamp(min=0))
    scores = torch.cat(scores)
    if per_image is not None:
        per_image["CLIPScore"] = [round(float(x), 4) for x in scores]
    return {"CLIPScore": round(float(scores.mean()), 4)}


def bootstrap_ci(values, n: int = BOOT, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile interval of the mean when images are resampled with replacement."""
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n, len(v)))].mean(axis=1)
    return round(float(np.quantile(means, alpha / 2)), 4), round(float(np.quantile(means, 1 - alpha / 2)), 4)


def paired_bootstrap(a, b, n: int = BOOT, seed: int = 0) -> dict:
    """Difference of means a - b over the same images: 95% interval and share of resamples where a <= b."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return {"diff": round(float(a.mean() - b.mean()), 4), "lo": round(float(np.quantile(d, 0.025)), 4),
            "hi": round(float(np.quantile(d, 0.975)), 4), "p_not_better": round(float((d <= 0).mean()), 4)}


# ---------------------------------------------------------------------- baselines
def retrieval_captions(ids, emb, caps, splits) -> dict[str, str]:
    """Copy the first caption of the most similar *training* image (cosine on pooled CLIP)."""
    train = splits["train"]
    tr = torch.nn.functional.normalize(emb["pooled"][[emb["index"][i] for i in train]].float(), dim=-1)
    q = torch.nn.functional.normalize(emb["pooled"][[emb["index"][i] for i in ids]].float(), dim=-1)
    nn_idx = (q @ tr.T).argmax(-1).tolist()
    return {i: caps[train[j]]["captions"][0] for i, j in zip(ids, nn_idx)}


@torch.no_grad()
def blip2_captions(ids, caps, model_name: str = "Salesforce/blip2-opt-2.7b", batch_size: int = 8) -> dict[str, str]:
    from PIL import Image
    from transformers import Blip2ForConditionalGeneration, Blip2Processor

    if not torch.cuda.is_available():
        raise RuntimeError("BLIP-2 baseline needs a GPU (run it on Colab or the HPC)")
    kw = {"device_map": "auto"}
    try:
        from transformers import BitsAndBytesConfig
        import bitsandbytes  # noqa: F401
        kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
    except ImportError:
        kw["torch_dtype"] = torch.float16
    proc = Blip2Processor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(model_name, **kw).eval()
    out = {}
    for s in range(0, len(ids), batch_size):
        b = ids[s:s + batch_size]
        imgs = [Image.open(IMG_DIR / caps[i]["file_name"]).convert("RGB") for i in b]
        inp = proc(images=imgs, return_tensors="pt").to(model.device, torch.float16)
        gen = model.generate(**inp, max_new_tokens=30)
        for i, t in zip(b, proc.batch_decode(gen, skip_special_tokens=True)):
            out[i] = t.strip()
    return out


# ---------------------------------------------------------------------- bookkeeping
def upsert_csv(row: dict, path: Path = RESULTS / "metrics.csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(path, encoding="utf-8"))) if path.exists() else []
    key = (row["name"], row["decode"], row["split"])
    rows = [r for r in rows if (r["name"], r["decode"], r["split"]) != key] + [{k: str(v) for k, v in row.items()}]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def diversity(res: dict[str, str]) -> dict:
    """How varied the captions are: share of distinct captions, distinct uni/bi-grams, vocabulary size."""
    toks = [re.findall(r"[a-z]+", c.lower()) for c in res.values()]
    uni = [w for t in toks for w in t]
    bi = [(a, b) for t in toks for a, b in zip(t, t[1:])]
    rep3 = sum(len(set(zip(t, t[1:], t[2:]))) < max(len(t) - 2, 0) for t in toks)
    four = {}
    for t in toks:
        for g in set(zip(t, t[1:], t[2:], t[3:])):
            four[g] = four.get(g, 0) + 1
    top = max(four.items(), key=lambda kv: kv[1]) if four else ((), 0)
    n = max(len(res), 1)
    return {"unique_captions": round(len(set(res.values())) / n, 4),
            "distinct-1": round(len(set(uni)) / max(len(uni), 1), 4),
            "distinct-2": round(len(set(bi)) / max(len(bi), 1), 4), "vocab": len(set(uni)),
            "rep-3": round(rep3 / n, 4),                      # captions that repeat a 3-gram inside themselves
            "top_4gram": " ".join(top[0]), "top_4gram_share": round(top[1] / n, 4)}  # most common phrase


def score_all(res, gts, img_emb, clip_model, device, meteor=True, extras=True, per_image: dict | None = None) -> dict:
    """All metrics for one caption set; per-image lists follow the order of gts."""
    res = {i: res[i] for i in gts}
    per_image = {} if per_image is None else per_image
    m = coco_metrics(gts, res, meteor, per_image)
    m.update(diversity(res))
    if extras:
        m.update(chair(res, gts, per_image))
        m.update(gender_eval(res, gts))
        m.update(clip_score(res, img_emb, clip_model, device, per_image))
    for k, v in per_image.items():
        m[f"{k}_lo"], m[f"{k}_hi"] = bootstrap_ci(v)
    return m


def save_gallery(ids: list[str], columns: dict[str, dict[str, str]], caps: dict, path: Path, width: int = 46) -> Path:
    """Images in rows; the reference and each system's caption beside each image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(len(ids), 2, figsize=(11, 3.2 * len(ids)), gridspec_kw={"width_ratios": [1, 1.6]})
    axes = axes.reshape(len(ids), 2)
    for r, iid in enumerate(ids):
        axes[r, 0].imshow(Image.open(IMG_DIR / caps[iid]["file_name"]).convert("RGB"))
        axes[r, 0].axis("off")
        lines = [f"REF: {caps[iid]['captions'][0]}"] + [f"{k}: {v[iid]}" for k, v in columns.items()]
        txt = "\n".join(textwrap.fill(l, width, subsequent_indent="   ") for l in lines)
        axes[r, 1].text(0, 0.5, txt, va="center", fontsize=9, family="monospace")
        axes[r, 1].axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------- main
def evaluate_run(run_dir: Path, specs: list[str], split: str, limit: int | None, device: str,
                 meteor: bool, extras: bool, seeds: int, batch_size: int, gallery: int) -> list[dict]:
    from src.model import load_run

    model, tok, cfg = load_run(run_dir, device)
    caps, splits = load_captions(cfg["data_dir"]), load_splits(cfg["data_dir"])
    emb = load_embeddings(cfg["clip_model"], cfg["data_dir"])
    ids = splits[split][:limit] if limit else splits[split]
    key = "pooled" if cfg["injection"] == "prefix" else "patch"
    feats = emb[key][[emb["index"][i] for i in ids]].float().to(device)
    gts = {i: caps[i]["captions"] for i in ids}
    img_emb = {i: emb["pooled"][emb["index"][i]] for i in ids}
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8")) if (run_dir / "summary.json").exists() else {}
    name = run_dir.name
    rows, gallery_cols = [], {}
    for spec in specs:
        n_seeds = seeds if spec.startswith("nucleus") else 1
        per_seed, first_caps = [], None
        for s in range(n_seeds):
            res = dict(zip(ids, generate_batched(model, feats, tok, spec, batch_size=batch_size,
                                                 max_len=cfg["max_caption_tokens"], seed=s)))
            each = {}
            per_seed.append(score_all(res, gts, img_emb, cfg["clip_model"], device, meteor, extras, each))
            if first_caps is None:
                first_caps, first_each = res, each
        metrics = {k: (round(sum(m[k] for m in per_seed) / n_seeds, 4) if isinstance(per_seed[0][k], (int, float))
                       else per_seed[0][k]) for k in per_seed[0]}
        out = RESULTS / name / f"{spec}_{split}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"metrics": metrics, "seeds": n_seeds, "image_ids": ids, "captions": first_caps,
                                   "per_image": first_each}, indent=1), encoding="utf-8")
        row = {"name": name, "injection": cfg["injection"], "finetune": cfg["finetune"], "decode": spec,
               "split": split, "n_images": len(ids), **metrics,
               **{k: summary.get(k) for k in ("trainable", "total", "trainable_pct", "mean_epoch_time_s", "peak_mem_gb")}}
        upsert_csv(row)
        rows.append(row)
        gallery_cols[spec] = first_caps
        print(json.dumps(row))
    if gallery:
        print("gallery:", save_gallery(ids[:gallery], gallery_cols, caps, RESULTS / name / f"gallery_{split}.png"))
    return rows


def evaluate_baseline(which: str, split: str, limit: int | None, device: str, meteor: bool, extras: bool,
                      clip_model: str) -> dict:
    caps, splits = load_captions(PROC_DIR), load_splits(PROC_DIR)
    emb = load_embeddings(clip_model, PROC_DIR)
    ids = splits[split][:limit] if limit else splits[split]
    gts = {i: caps[i]["captions"] for i in ids}
    name = f"baseline_{which}"
    out = RESULTS / name / f"{which}_{split}.json"
    old = json.loads(out.read_text(encoding="utf-8"))["captions"] if out.exists() else {}
    if which == "blip2" and all(i in old for i in ids):   # BLIP-2 is slow: rescore the saved captions
        print("reusing saved BLIP-2 captions from", out)
        res = {i: old[i] for i in ids}
    else:
        res = retrieval_captions(ids, emb, caps, splits) if which == "retrieval" else blip2_captions(ids, caps)
    each = {}
    metrics = score_all(res, gts, {i: emb["pooled"][emb["index"][i]] for i in ids}, clip_model, device, meteor,
                        extras, each)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metrics": metrics, "image_ids": ids, "captions": res, "per_image": each}, indent=1),
                   encoding="utf-8")
    row = {"name": name, "injection": "-", "finetune": "-", "decode": which, "split": split, "n_images": len(ids), **metrics}
    upsert_csv(row)
    print(json.dumps(row))
    return row


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", nargs="*", default=[], help="run directories under runs/")
    p.add_argument("--decode", nargs="*", default=["greedy", "beam_w3", "nucleus_p0.9_t0.7"],
                   help="decoding specs, or 'sweep' for the sensitivity grid")
    p.add_argument("--baseline", choices=["retrieval", "blip2"])
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--limit", type=int, help="only the first N images (quick check)")
    p.add_argument("--seeds", type=int, default=3, help="seeds averaged for nucleus sampling")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--gallery", type=int, default=6, help="images in the qualitative gallery (0 = none)")
    p.add_argument("--no-meteor", action="store_true")
    p.add_argument("--no-extras", action="store_true", help="skip CHAIR, gender and CLIPScore")
    p.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    p.add_argument("--device")
    a = p.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    specs = SWEEP if a.decode == ["sweep"] else a.decode
    for r in a.runs:
        evaluate_run(Path(r), specs, a.split, a.limit, device, not a.no_meteor, not a.no_extras,
                     a.seeds, a.batch_size, a.gallery)
    if a.baseline:
        evaluate_baseline(a.baseline, a.split, a.limit, device, not a.no_meteor, not a.no_extras, a.clip_model)
    if not a.runs and not a.baseline:
        p.error("give --runs and/or --baseline")


if __name__ == "__main__":
    main()
