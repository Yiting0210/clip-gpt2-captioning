"""Results manifest: ties every reported run to its code, configuration, data, environment and Slurm job.

    python -m src.manifest [--out results/manifest.json]      (repository root, on the machine that has runs/)

For every run folder with summary.json it records
    config       the merged config.yaml (as trained) and its SHA-256
    checkpoint   SHA-256 and size of best.pt (when the file is present)
    training     epochs run, best validation loss, trainable parameters
    slurm        the job whose log reports this run's training as done (logs/*.out), and the GPU in that log
    results      the results/<run>/*.json files with their SHA-256
plus, once for the whole manifest:
    code         git commit (if the folder is a git checkout) and SHA-256 of every file under src/, configs/, scripts/
    data         SHA-256 of captions.json, splits.json and the CLIP embedding file
    environment  python version and the versions of the main packages
The report cites numbers from results/metrics.csv; this file says where each of them came from.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import yaml

PACKAGES = ["torch", "transformers", "peft", "huggingface-hub", "pycocoevalcap", "diffusers", "numpy", "pandas"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "configs", "scripts"],
                               capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def code_hashes() -> dict[str, str]:
    files = [p for d in ("src", "configs", "scripts") for p in sorted(Path(d).rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts]
    return {p.as_posix(): sha256(p) for p in files}


def slurm_index(log_dir: Path) -> dict[str, dict]:
    """run name -> {job, gpu} for the job whose log shows that run's training finishing."""
    out = {}
    for log in sorted(log_dir.glob("*.out")):
        text = log.read_text(encoding="utf-8", errors="replace")
        gpu = next(iter(re.findall(r"^(Tesla [^,]+|NVIDIA [^,]+),", text, re.M)), None)
        job = log.stem.split("_")[-1]
        for run in re.findall(r"event=done .*?run_dir=runs/(\S+)", text):
            out[run] = {"job": job, "log": log.as_posix(), "gpu": gpu}
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/manifest.json")
    p.add_argument("--git-commit", help="commit of this code when the folder is not a git checkout (e.g. the cluster copy)")
    a = p.parse_args()

    jobs = slurm_index(Path("logs"))
    runs = {}
    for summ in sorted(Path("runs").glob("*/summary.json")):
        d = summ.parent
        cfg_path = d / "config.yaml"
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        s = json.loads(summ.read_text(encoding="utf-8"))
        ckpt = d / "best.pt"
        res_dir = Path("results") / d.name
        runs[d.name] = {
            "config": {k: cfg.get(k) for k in ("injection", "finetune", "prefix_length", "train_images")}
                      | {"seed": cfg.get("train", {}).get("seed"), "epochs_max": cfg.get("train", {}).get("epochs"),
                         "patience": cfg.get("train", {}).get("early_stopping_patience"),
                         "lr": cfg.get("train", {}).get("lr"), "gpt_lr": cfg.get("train", {}).get("gpt_lr")},
            "config_sha256": sha256(cfg_path) if cfg_path.exists() else None,
            "checkpoint": {"file": ckpt.as_posix(), "sha256": sha256(ckpt), "bytes": ckpt.stat().st_size}
                          if ckpt.exists() else None,
            "training": {k: s.get(k) for k in ("epochs_run", "best_val_loss", "trainable", "total", "mean_epoch_time_s",
                                               "peak_mem_gb", "device")},
            "slurm": jobs.get(d.name),
            "results": {f.as_posix(): sha256(f) for f in sorted(res_dir.glob("*.json"))} if res_dir.exists() else {},
        }
    data = {f: sha256(Path(f)) for f in ("data/processed/captions.json", "data/processed/splits.json")
            if Path(f).exists()}
    data |= {f.as_posix(): sha256(f) for f in sorted(Path("data/processed").glob("clip_*.pt"))}
    env = {"python": platform.python_version()}
    for pkg in PACKAGES:
        try:
            env[pkg] = version(pkg)
        except PackageNotFoundError:
            env[pkg] = None
    manifest = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "host": platform.node(),
                "code": {"git_commit": git_commit() or a.git_commit, "files": code_hashes()}, "data": data, "environment": env,
                "metrics_csv_sha256": sha256(Path("results/metrics.csv")) if Path("results/metrics.csv").exists() else None,
                "runs": runs}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    n_ck = sum(r["checkpoint"] is not None for r in runs.values())
    n_job = sum(r["slurm"] is not None for r in runs.values())
    print(f"wrote {a.out}: {len(runs)} runs ({n_ck} with checkpoint hash, {n_job} matched to a Slurm job)")


if __name__ == "__main__":
    sys.exit(main())
