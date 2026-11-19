"""Small blind human evaluation: 50 test images, three systems, sources hidden.

    python -m src.human_eval make  --systems results/prefix_full_s0/beam_w3_test.json \
                                             results/baseline_blip2/blip2_test.json \
                                             results/baseline_retrieval/retrieval_test.json
    (open results/human_eval/sheet.html, fill the rating column of results/human_eval/ratings.csv)
    python -m src.human_eval score

For each image the three captions are shown as A / B / C in a random order, so the rater cannot tell
which system wrote which. Each caption gets one rating:
    2 = correct        everything it says is in the image
    1 = partly correct main content right, but a wrong detail (count, colour, action)
    0 = hallucination  names an object or action that is not in the image
The key that maps A / B / C back to the systems is kept in key.json; `score` joins the two.
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import random
import shutil
from pathlib import Path

from src.data import IMG_DIR, PROC_DIR, load_captions, load_splits
from src.evaluate import RESULTS

OUT = RESULTS / "human_eval"
LABELS = {"2": "correct", "1": "partly correct", "0": "hallucination"}


def system_name(path: str) -> str:
    p = Path(path)
    return f"{p.parent.name}:{p.stem.removesuffix('_test')}"


def same_style(caption: str) -> str:
    """One surface form for every system (capital first letter, one final period), so that style does not
    give away the source: BLIP-2 writes lower case without a period, the other systems do not."""
    c = " ".join(caption.split()).rstrip(" .")
    return (c[:1].upper() + c[1:] + ".") if c else c


def make(systems: list[str], n: int, seed: int) -> None:
    caps, splits = load_captions(PROC_DIR), load_splits(PROC_DIR)
    sets = {system_name(s): {i: same_style(c) for i, c in json.loads(Path(s).read_text(encoding="utf-8"))["captions"].items()}
            for s in systems}
    rng = random.Random(seed)
    ids = rng.sample(splits["test"], n)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    key, rows, cards = {}, [], []
    for k, iid in enumerate(ids, 1):
        names = list(sets)
        rng.shuffle(names)
        key[iid] = dict(zip("ABC", names))
        fn = caps[iid]["file_name"]
        shutil.copy(IMG_DIR / fn, OUT / "images" / fn)
        items = []
        for slot, name in key[iid].items():
            rows.append({"item": k, "image_id": iid, "slot": slot, "caption": sets[name][iid], "rating": ""})
            items.append(f"<li><b>{slot}</b> {html.escape(sets[name][iid])}</li>")
        cards.append(f'<section><h2>{k}. image {iid}</h2><img src="images/{fn}" alt="test image {iid}">'
                     f'<ol>{"".join(items)}</ol></section>')
    with open(OUT / "ratings.csv", "w", newline="", encoding="utf-8-sig") as f:   # BOM: Excel opens it cleanly
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (OUT / "key.json").write_text(json.dumps({"seed": seed, "systems": list(sets), "key": key}, indent=1),
                                  encoding="utf-8")
    guide = " · ".join(f"{k} = {v}" for k, v in LABELS.items())
    page = f"""<!doctype html><meta charset="utf-8"><title>Caption rating sheet</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:24px auto;padding:0 16px;color:#222;background:#fff}}
section{{border-top:1px solid #ddd;padding:12px 0}}img{{max-width:100%;max-height:380px;display:block}}
ol{{list-style:none;padding:0}}li{{margin:6px 0}}h2{{font-size:16px;margin:4px 0}}</style>
<h1>Caption rating sheet ({n} images)</h1>
<p>Rate every caption in <code>ratings.csv</code> (rating column): {guide}.
Judge only against the image, not against the other captions.</p>
{''.join(cards)}"""
    (OUT / "sheet.html").write_text(page, encoding="utf-8")
    print(f"wrote {OUT / 'sheet.html'} and {OUT / 'ratings.csv'} ({len(rows)} captions to rate)")


def score() -> dict:
    key = json.loads((OUT / "key.json").read_text(encoding="utf-8"))["key"]
    rows = list(csv.DictReader(open(OUT / "ratings.csv", encoding="utf-8-sig")))
    todo = [r for r in rows if r["rating"].strip() not in LABELS]
    if todo:
        raise SystemExit(f"{len(todo)} captions still unrated or rated outside 0/1/2, e.g. item {todo[0]['item']}")
    out: dict[str, dict] = {}
    for r in rows:
        name = key[r["image_id"]][r["slot"]]
        d = out.setdefault(name, {"n": 0, "correct": 0, "partly correct": 0, "hallucination": 0, "sum": 0})
        d["n"] += 1
        d[LABELS[r["rating"].strip()]] += 1
        d["sum"] += int(r["rating"])
    for d in out.values():
        for k in ("correct", "partly correct", "hallucination"):
            d[f"{k} %"] = round(100 * d[k] / d["n"], 1)
        d["mean rating"] = round(d.pop("sum") / d["n"], 3)
    (OUT / "summary.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--systems", nargs=3, required=True, help="three results JSON files")
    m.add_argument("--n", type=int, default=50)
    m.add_argument("--seed", type=int, default=7)
    sub.add_parser("score")
    a = p.parse_args()
    make(a.systems, a.n, a.seed) if a.cmd == "make" else score()


if __name__ == "__main__":
    main()
