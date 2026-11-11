"""Local Gradio demo: upload a photo, pick a trained model and a decoding strategy, get a caption.

    python demo/app.py                 # then open http://127.0.0.1:7860
    python demo/app.py --runs-dir runs --port 7860

Every run folder under runs/ that holds best.pt + config.yaml shows up in the model list.
The image goes through the same frozen CLIP ViT-B/32 as in training (pooled 512-d vector for prefix
models, 50 x 768 patch features for cross-attention models), then GPT-2 writes the caption.
Runs on CPU; a caption takes about a second.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
import torch

from src.decode import generate
from src.model import load_run

CLIP_MODEL = "openai/clip-vit-base-patch32"
INJECTION = {"prefix": "prefix", "cross_attention": "cross-attention"}
FINETUNE = {"prefix": "GPT-2 frozen", "lora": "LoRA", "full": "full fine-tuning"}

ap = argparse.ArgumentParser()
ap.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
ap.add_argument("--port", type=int, default=7860)
args, _ = ap.parse_known_args()


# ---------------------------------------------------------------------- models
def find_runs(runs_dir: Path) -> dict[str, Path]:
    """label -> run folder, for runs that can be loaded."""
    out = {}
    for d in sorted(runs_dir.glob("*")):
        if not ((d / "best.pt").exists() and (d / "config.yaml").exists()) or "debug" in d.name:
            continue
        cfg = (d / "config.yaml").read_text(encoding="utf-8")
        inj = "cross_attention" if "injection: cross_attention" in cfg else "prefix"
        ft = next((k for k in FINETUNE if f"finetune: {k}\n" in cfg), "prefix")
        out[f"{INJECTION[inj]} + {FINETUNE[ft]}  ({d.name})"] = d
    return out


RUNS = find_runs(args.runs_dir)


@lru_cache(maxsize=1)
def clip():
    from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizerFast
    model = CLIPModel.from_pretrained(CLIP_MODEL).eval()
    return model, CLIPImageProcessor.from_pretrained(CLIP_MODEL), CLIPTokenizerFast.from_pretrained(CLIP_MODEL)


@lru_cache(maxsize=8)
def captioner(label: str):
    return load_run(RUNS[label], "cpu")


@torch.no_grad()
def image_features(image) -> dict:
    """Same features as src/data.py embed(), including the fp16 round trip used for training."""
    model, proc, _ = clip()
    px = proc(images=[image.convert("RGB")], return_tensors="pt")["pixel_values"]
    vis = model.vision_model(pixel_values=px)
    return {"pooled": model.visual_projection(vis.pooler_output).half().float(),
            "patch": vis.last_hidden_state.half().float()}


@torch.no_grad()
def clip_score(feats: dict, caption: str) -> float:
    model, _, tok = clip()
    t = tok([caption or "."], padding=True, truncation=True, max_length=77, return_tensors="pt")
    te = torch.nn.functional.normalize(model.get_text_features(**t).float(), dim=-1)
    ie = torch.nn.functional.normalize(feats["pooled"], dim=-1)
    return float(2.5 * (te * ie).sum(-1).clamp(min=0))


def spec_of(strategy: str, width: int, top_p: float, temperature: float, no_repeat: bool) -> str:
    s = {"greedy": "greedy", "beam search": f"beam_w{int(width)}",
         "nucleus sampling": f"nucleus_p{top_p:g}_t{temperature:g}"}[strategy]
    return s + ("_nr3" if no_repeat else "")


def caption_with(label: str, feats: dict, spec: str, seed: int | None = None) -> str:
    model, tok, cfg = captioner(label)
    f = feats["pooled"] if cfg["injection"] == "prefix" else feats["patch"]
    return generate(model, f, tok, spec, max_len=cfg["max_caption_tokens"], seed=seed)[0]


# ---------------------------------------------------------------------- callbacks
def run_one(image, label, strategy, width, top_p, temperature, no_repeat):
    if image is None:
        raise gr.Error("Upload an image or pick an example first.")
    if label not in RUNS:
        raise gr.Error("No trained model found under runs/.")
    t0 = time.time()
    feats = image_features(image)
    spec = spec_of(strategy, width, top_p, temperature, no_repeat)
    seed = int(time.time() * 1000) % 100000 if strategy == "nucleus sampling" else None
    cap = caption_with(label, feats, spec, seed)
    info = (f"**decoding spec** `{spec}` · **CLIPScore** {clip_score(feats, cap):.3f} "
            f"(image-text agreement, higher is better) · **time** {time.time() - t0:.2f} s")
    return cap, info


def run_samples(image, label, top_p, temperature, no_repeat):
    if image is None:
        raise gr.Error("Upload an image or pick an example first.")
    feats = image_features(image)
    spec = spec_of("nucleus sampling", 3, top_p, temperature, no_repeat)
    caps = [caption_with(label, feats, spec, seed=s) for s in range(5)]
    return "\n".join(f"{i + 1}. {c}" for i, c in enumerate(caps))


def run_compare(image, no_repeat):
    if image is None:
        raise gr.Error("Upload an image or pick an example first.")
    feats = image_features(image)
    nr = "_nr3" if no_repeat else ""
    rows = []
    for label in RUNS:
        rows.append([label] + [caption_with(label, feats, s + nr, seed=0)
                               for s in ("greedy", "beam_w3", "nucleus_p0.9_t0.7")])
    return rows


def examples() -> list[str]:
    """A few local test-split images (the demo never shows training images as examples)."""
    try:
        splits = json.loads((ROOT / "data/processed/splits.json").read_text())
        caps = json.loads((ROOT / "data/processed/captions.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    paths = [ROOT / "data/raw/val2017" / caps[i]["file_name"] for i in splits["test"]]
    return [str(p) for p in paths if p.exists()][:8]



# ---------------------------------------------------------------------- page (presentation only)
REPO_URL = "https://github.com/Yiting0210/clip-gpt2-captioning"   # from `git remote get-url origin`
MODES = ["One caption", "Sample 5 times", "Compare all models"]
STRATEGIES = ["greedy", "beam search", "nucleus sampling"]
BUTTON_IDLE, BUTTON_BUSY = "Generate caption", "Generating…"

GITHUB_SVG = ('<svg class="cg-gh-icon" viewBox="0 0 16 16" width="16" height="16" aria-hidden="true" focusable="false">'
              '<path fill="currentColor" d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 '
              '0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 '
              '1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15'
              '-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82'
              '.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 '
              '1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>')

HEADER = f"""
<header class="cg-header">
  <div class="cg-header-inner">
    <div class="cg-brand">
      <h1>CLIP + GPT-2 Image Captioner</h1>
      <p>A frozen CLIP encoder understands the image. GPT-2 generates a caption token by token.</p>
    </div>
    <nav class="cg-meta" aria-label="Model facts and project links">
      <ul class="cg-badges">
        <li>CLIP (Frozen)</li><li>GPT-2</li><li>COCO (4,000)</li>
      </ul>
      <span class="cg-divider" aria-hidden="true"></span>
      <a class="cg-github" href="{REPO_URL}" target="_blank" rel="noopener noreferrer"
         aria-label="Open the project repository on GitHub (opens in a new tab)" title="GitHub repository">{GITHUB_SVG}</a>
    </nav>
  </div>
</header>
"""

CSS = """
:root, .gradio-container {
  --cg-primary: #6842F5; --cg-primary-hover: #5634DC; --cg-bg: #F7F8FC; --cg-surface: #FFFFFF;
  --cg-text: #17213B; --cg-muted: #64748B; --cg-border: #E4E8F2; --cg-header: #17223F;
  --cg-tint: #F3F0FF; --cg-track: #EEF1F7; --cg-radius: 14px;
  --cg-shadow: 0 1px 2px rgba(23, 33, 59, .04), 0 4px 16px rgba(23, 33, 59, .05);
  /* Gradio theme variables, so built-in controls pick up the palette */
  --color-accent: var(--cg-primary); --color-accent-soft: var(--cg-tint);
  --button-primary-background-fill: var(--cg-primary); --button-primary-background-fill-hover: var(--cg-primary-hover);
  --button-primary-text-color: #fff; --button-primary-border-color: var(--cg-primary);
  --slider-color: var(--cg-primary); --checkbox-background-color-selected: var(--cg-primary);
  --checkbox-border-color-selected: var(--cg-primary); --body-text-color: var(--cg-text);
  --body-text-color-subdued: var(--cg-muted); --block-border-color: var(--cg-border);
  --input-border-color-focus: var(--cg-primary); --background-fill-primary: var(--cg-surface);
  --block-label-text-color: var(--cg-muted); --block-title-text-color: var(--cg-text);
}
body, gradio-app, .gradio-container { background: var(--cg-bg) !important; }
.gradio-container { max-width: none !important; padding: 0 !important; margin: 0 !important; color: var(--cg-text); }
.gradio-container > .main, .gradio-container .contain, .gradio-container main.fillable { padding: 0 !important; }
#cg-header-block { border-radius: 0 !important; overflow: visible !important; }
footer { display: none !important; }

/* header: full width band, content aligned with the page container */
.cg-header { background: linear-gradient(115deg, #17223F 0%, #1F2350 55%, #33257A 100%); color: #fff; }
.cg-header-inner { max-width: 1280px; margin: 0 auto; box-sizing: border-box;
  padding: 22px max(24px, env(safe-area-inset-right, 0px)) 22px max(24px, env(safe-area-inset-left, 0px));
  display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 14px 24px; }
.cg-brand { min-width: 0; }
.cg-brand h1 { margin: 0; font-size: 1.45rem; font-weight: 650; letter-spacing: -.01em; color: #fff; }
.cg-brand p { margin: 4px 0 0; color: rgba(255, 255, 255, .72); font-size: .92rem; }
.cg-meta { display: flex; flex-wrap: wrap; align-items: center; gap: 10px; margin-left: auto; }
.cg-badges { display: flex; flex-wrap: wrap; gap: 8px; list-style: none; margin: 0; padding: 0; }
.cg-badges li { font-size: .78rem; font-weight: 500; padding: 4px 10px; border-radius: 999px;
  background: rgba(255, 255, 255, .08); border: 1px solid rgba(255, 255, 255, .16); color: rgba(255, 255, 255, .9); white-space: nowrap; }
.cg-divider { width: 1px; height: 22px; background: rgba(255, 255, 255, .22); }
.cg-github { display: inline-flex; align-items: center; justify-content: center; width: 38px; height: 38px;
  box-sizing: border-box; padding: 0; border-radius: 10px;
  border: 1px solid rgba(160, 135, 255, .75); background: rgba(104, 66, 245, .18); color: #fff !important;
  font-size: .85rem; font-weight: 600; text-decoration: none !important; white-space: nowrap;
  transition: background .15s ease, border-color .15s ease; }
.cg-github:hover { background: rgba(104, 66, 245, .4); border-color: #B9A6FF; }
.cg-github:focus-visible { outline: 2px solid #fff; outline-offset: 2px; }
.cg-github svg { color: #fff !important; width: 20px; height: 20px; }
.cg-github svg path { fill: #fff !important; }

/* page container and the two columns (44 / 56) */
#cg-page { max-width: 1280px; margin: 0 auto; box-sizing: border-box; gap: 24px !important;
  padding: 24px max(24px, env(safe-area-inset-right, 0px)) 40px max(24px, env(safe-area-inset-left, 0px)) !important; }
#cg-page { align-items: stretch !important; }
#cg-left { flex: 44 1 0 !important; min-width: 0 !important; gap: 24px !important; }
#cg-right { flex: 56 1 0 !important; min-width: 0 !important; gap: 24px !important; }
#cg-left, #cg-right { display: flex !important; flex-direction: column !important; }
#cg-left > .cg-card:first-child, #cg-right > .cg-card:first-child { flex: 1 1 auto !important; }
#cg-left > .cg-card:first-child #cg-image { flex: 1 1 auto !important; }
#cg-left > .cg-card:last-child, #cg-right > .cg-card:last-child { flex: 0 0 auto !important; }

/* cards */
.cg-card { background: var(--cg-surface) !important; border: 1px solid var(--cg-border) !important;
  border-radius: var(--cg-radius) !important; box-shadow: var(--cg-shadow); padding: 18px 20px !important; gap: 14px !important; }
.cg-card .block, .cg-card .form { border: 0 !important; box-shadow: none !important; background: transparent !important; }
.cg-card-title { margin: 0; font-size: .98rem; font-weight: 650; color: var(--cg-text); }
.cg-card-title small { font-weight: 400; color: var(--cg-muted); margin-left: 6px; }
.cg-hint { margin: 0; color: var(--cg-muted); font-size: .86rem; }

/* input image: soft background, whole image visible */
#cg-image { background: var(--cg-bg) !important; border-radius: 12px !important; overflow: hidden; }
#cg-image img { object-fit: contain !important; }

/* example thumbnails: one row, scrolls sideways when narrow */
#cg-examples .grid-wrap { overflow-x: auto !important; overflow-y: hidden !important; padding: 3px !important; }
#cg-examples .grid-container { display: flex !important; flex-wrap: nowrap !important; gap: 10px !important; }
#cg-examples .gallery-item { flex: 0 0 108px !important; width: 108px !important; min-width: 0 !important; }
#cg-examples .thumbnail-item { position: relative; flex: 0 0 108px !important; width: 108px !important; height: 84px !important;
  border-radius: 10px !important; border: 2px solid transparent !important; box-shadow: none !important; transition: border-color .15s ease, transform .15s ease; }
#cg-examples .thumbnail-item:hover { border-color: #C9BDFB !important; transform: translateY(-1px); }
#cg-examples .thumbnail-item.selected { border-color: var(--cg-primary) !important; box-shadow: 0 0 0 3px rgba(104, 66, 245, .18) !important; }
#cg-examples .thumbnail-item.selected::after { content: "\2713"; position: absolute; top: 5px; right: 5px; width: 18px; height: 18px;
  border-radius: 50%; background: var(--cg-primary); color: #fff; font-size: 11px; font-weight: 700; line-height: 18px; text-align: center; }

/* segmented control for the generation mode */
#cg-mode > .wrap:not(.default) { display: flex !important; flex-wrap: wrap; gap: 4px !important; background: var(--cg-track);
  border-radius: 12px; padding: 4px; }
#cg-mode label { flex: 1 1 0; justify-content: center; margin: 0 !important; border: 0 !important;
  background: transparent !important; border-radius: 9px !important; padding: 8px 10px !important;
  color: var(--cg-muted); font-weight: 600; box-shadow: none !important; cursor: pointer; white-space: nowrap; }
#cg-mode label.selected { background: var(--cg-surface) !important; color: var(--cg-primary);
  box-shadow: 0 1px 3px rgba(23, 33, 59, .12) !important; }
#cg-mode input[type=radio] { position: absolute; opacity: 0; width: 1px; height: 1px; }
#cg-mode label:has(input:focus-visible) { outline: 2px solid var(--cg-primary); outline-offset: 1px; }

/* decoding strategy radios */
#cg-strategy > .wrap:not(.default) { display: grid !important; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px !important; }
#cg-strategy label { border: 1px solid var(--cg-border) !important; border-radius: 10px !important; padding: 9px 12px !important;
  background: var(--cg-surface) !important; margin: 0 !important; }
#cg-strategy label.selected { border-color: var(--cg-primary) !important; background: var(--cg-tint) !important; color: var(--cg-text); }
#cg-strategy input[type=radio]:checked { background-color: var(--cg-primary) !important; border-color: var(--cg-primary) !important; }

/* primary action */
#cg-generate { width: 100%; min-height: 48px; border-radius: 12px !important; font-size: 1rem !important; font-weight: 650 !important;
  background: var(--cg-primary) !important; border-color: var(--cg-primary) !important; color: #fff !important; }
#cg-generate:hover:not(:disabled) { background: var(--cg-primary-hover) !important; }
#cg-generate:disabled { opacity: .65; cursor: progress; }

/* caption output: violet tinted area */
#cg-caption textarea, #cg-samples textarea { background: var(--cg-tint) !important; border: 1px solid #E2DBFF !important;
  border-radius: 10px !important; font-size: 1.05rem !important; line-height: 1.55 !important; color: var(--cg-text) !important; }
#cg-info { color: var(--cg-muted); font-size: .86rem; }
#cg-compare table { font-size: .88rem; }

/* tablet and phone */
@media (max-width: 900px) {
  #cg-page { flex-direction: column !important; }
  #cg-left, #cg-right { flex: 1 1 auto !important; width: 100% !important; }
}
@media (max-width: 560px) {
  .cg-header-inner { padding-top: 18px; padding-bottom: 18px; }
  .cg-brand h1 { font-size: 1.2rem; }
  .cg-header-inner { position: relative; }
  .cg-brand { padding-right: 52px; }
  .cg-meta { margin-left: 0; width: 100%; }
  .cg-divider { display: none; }
  /* the GitHub button stays in the top-right corner, beside the title */
  .cg-github { position: absolute; top: 18px; right: max(16px, env(safe-area-inset-right, 0px)); }
  #cg-strategy > .wrap:not(.default) { grid-template-columns: 1fr; }
  #cg-mode label { flex: 1 1 100%; }
}
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

# The palette above is light-only; keep Gradio in light mode even when the OS is dark.
FORCE_LIGHT = "() => { document.body.classList.remove('dark'); document.documentElement.classList.remove('dark'); }"


def generate_any(mode, image, label, strategy, width, top_p, temperature, no_repeat, s_top_p, s_temp):
    """One button for the three modes; each mode calls its existing callback unchanged."""
    if mode == "Sample 5 times":
        return gr.update(), gr.update(), run_samples(image, label, s_top_p, s_temp, no_repeat), gr.update()
    if mode == "Compare all models":
        return gr.update(), gr.update(), gr.update(), run_compare(image, no_repeat)
    cap, info = run_one(image, label, strategy, width, top_p, temperature, no_repeat)
    return cap, info, gr.update(), gr.update()


def show_mode(mode):
    one, sample, compare = (mode == m for m in MODES)
    return (gr.update(visible=one), gr.update(visible=sample), gr.update(visible=compare),
            gr.update(visible=one), gr.update(visible=one), gr.update(visible=sample), gr.update(visible=compare))


def show_strategy(strategy):
    return (gr.update(visible=strategy == "beam search"), gr.update(visible=strategy == "nucleus sampling"),
            gr.update(visible=strategy == "nucleus sampling"))


EXAMPLES = examples()


def pick_example(evt: gr.SelectData):
    return EXAMPLES[evt.index]


theme = gr.themes.Base(primary_hue="violet", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
                       radius_size=gr.themes.sizes.radius_lg)

with gr.Blocks(title="CLIP + GPT-2 Image Captioner", css=CSS, theme=theme, js=FORCE_LIGHT) as app:
    gr.HTML(HEADER, padding=False, elem_id="cg-header-block")
    with gr.Row(elem_id="cg-page", equal_height=False):
        with gr.Column(elem_id="cg-left"):
            with gr.Column(elem_classes="cg-card"):
                gr.HTML('<h2 class="cg-card-title">Input image</h2>', padding=False)
                image = gr.Image(type="pil", show_label=False, height=380, elem_id="cg-image",
                                 sources=["upload", "clipboard"])
            if EXAMPLES:
                with gr.Column(elem_classes="cg-card"):
                    gr.HTML('<h2 class="cg-card-title">Example images <small>test split, never seen in training</small></h2>',
                            padding=False)
                    gallery = gr.Gallery(EXAMPLES, show_label=False, columns=len(EXAMPLES), rows=1, height=112,
                                         allow_preview=False, object_fit="cover", elem_id="cg-examples",
                                         show_download_button=False, show_share_button=False, show_fullscreen_button=False)
                    gallery.select(pick_example, None, image)

        with gr.Column(elem_id="cg-right"):
            with gr.Column(elem_classes="cg-card"):
                gr.HTML('<h2 class="cg-card-title">Model configuration</h2>', padding=False)
                labels = list(RUNS)
                model = gr.Dropdown(labels, value=next((l for l in labels if "prefix + LoRA" in l), labels[0] if labels else None),
                                    label="Model (injection + fine-tuning)")
                mode = gr.Radio(MODES, value=MODES[0], label="Generation mode", elem_id="cg-mode")
                with gr.Column(visible=True) as one_group:
                    strategy = gr.Radio(STRATEGIES, value="beam search", label="Decoding strategy", elem_id="cg-strategy")
                    width = gr.Slider(1, 10, value=3, step=1, label="Beam width", visible=True)
                    top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p", visible=False)
                    temperature = gr.Slider(0.1, 1.5, value=0.7, step=0.1, label="Temperature", visible=False)
                with gr.Column(visible=False) as sample_group:
                    gr.HTML('<p class="cg-hint">Nucleus sampling draws a different caption each time: '
                            'the generative side of the model.</p>', padding=False)
                    s_top_p = gr.Slider(0.5, 1.0, value=0.9, step=0.05, label="Top-p")
                    s_temp = gr.Slider(0.1, 1.5, value=1.0, step=0.1, label="Temperature")
                with gr.Column(visible=False) as compare_group:
                    gr.HTML('<p class="cg-hint">Every trained model captions the image with greedy, beam search '
                            '(width 3) and nucleus sampling (p 0.9, T 0.7).</p>', padding=False)
                no_repeat = gr.Checkbox(False, label="Block repeated 3-grams (no-repeat n-gram)")
                go = gr.Button(BUTTON_IDLE, variant="primary", elem_id="cg-generate")

            with gr.Column(elem_classes="cg-card"):
                gr.HTML('<h2 class="cg-card-title">Caption</h2>', padding=False)
                out = gr.Textbox(show_label=False, lines=2, placeholder="The caption appears here.",
                                 show_copy_button=True, interactive=False, elem_id="cg-caption")
                info = gr.Markdown(elem_id="cg-info")
                s_out = gr.Textbox(show_label=False, lines=6, placeholder="Five sampled captions appear here.",
                                   show_copy_button=True, interactive=False, visible=False, elem_id="cg-samples")
                c_out = gr.Dataframe(headers=["model", "greedy", "beam (width 3)", "nucleus (p 0.9, T 0.7)"],
                                     wrap=True, interactive=False, visible=False, elem_id="cg-compare")

    # visibility only: skip the queue so the controls react at once, even while a caption is being generated
    mode.change(show_mode, mode, [one_group, sample_group, compare_group, out, info, s_out, c_out], queue=False)
    strategy.change(show_strategy, strategy, [width, top_p, temperature], queue=False)
    (go.click(lambda: gr.update(value=BUTTON_BUSY, interactive=False), None, go, queue=False)
       .then(generate_any, [mode, image, model, strategy, width, top_p, temperature, no_repeat, s_top_p, s_temp],
             [out, info, s_out, c_out])
       .then(lambda: gr.update(value=BUTTON_IDLE, interactive=True), None, go, queue=False))

if __name__ == "__main__":
    if not RUNS:
        print(f"no runs with best.pt + config.yaml under {args.runs_dir}")
    app.launch(server_name="127.0.0.1", server_port=args.port)
