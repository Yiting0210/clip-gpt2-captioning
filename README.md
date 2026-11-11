# clip-gpt2-captioning

Image captioning with a frozen **CLIP** image encoder and a **GPT-2** language decoder, trained on a 5,000-image
subset of COCO Captions. The project compares two ways of injecting the image into GPT-2 (prefix conditioning
and cross-attention), three fine-tuning strategies (frozen GPT-2, LoRA and full fine-tuning) and three decoding
strategies (greedy, beam search and nucleus sampling).

Course project for IE7615 (Northeastern University).

## How it works

```
image ──► CLIP ViT-B/32 (frozen) ──► 512-d embedding ──► mapping MLP ──► 10 prefix embeddings ─┐
                                                                                                ├─► GPT-2 ──► caption
                                         caption so far ──► GPT-2 token embeddings ─────────────┘
```

- **Prefix conditioning** (ClipCap): an MLP turns the CLIP embedding into 10 pseudo-token embeddings that are
  placed in front of the caption.
- **Cross-attention**: every GPT-2 block gets a new cross-attention layer that attends to the 50 CLIP patch features.
- Only the new weights train by default. LoRA adds low-rank adapters to GPT-2 self-attention; full fine-tuning
  updates every GPT-2 weight.

## Repository layout

```
src/data.py        download, clean and split COCO val2017; CLIP embeddings; PyTorch dataset
src/model.py       ClipCaptionModel (prefix / cross-attention × frozen / LoRA / full)
src/decode.py      greedy, beam search, nucleus sampling
src/train.py       training loop: --debug, --overfit-batch, --resume, checkpoints, JSONL logs
src/evaluate.py    BLEU, METEOR, ROUGE-L, CIDEr, CHAIR, gender-word error, CLIPScore, bootstrap intervals, baselines
src/attention.py   cross-attention heatmaps and a grounding score (attention inside the named object's boxes)
src/cycle.py       image -> caption -> image with Stable Diffusion v1.5, scored by CLIP image similarity
configs/           one YAML per experiment, all inheriting configs/base.yaml
notebooks/         01 data pipeline · 02 baseline captions + decoding · 03 evaluation
scripts/           Slurm jobs for the HPC cluster
tests/             fast unit tests on a tiny random GPT-2
results/           metrics.csv, generated captions, figures
```

## Setup

```bash
conda create -n captioning python=3.12 -y && conda activate captioning
pip install torch            # pick the CUDA or CPU build from pytorch.org
pip install -r requirements.txt
conda install -c conda-forge openjdk -y   # Java for METEOR and the PTB tokenizer
python -m pytest -q          # ~1 minute, no dataset needed
```

## Run

```bash
# 1. data (≈1 GB download) and CLIP embeddings
python -m src.data download
python -m src.data prepare
python -m src.data embed

# 2. sanity checks before a long run
python -m src.train --config configs/prefix_frozen.yaml --overfit-batch   # loss → ~0, captions memorised
python -m src.train --config configs/prefix_frozen.yaml --debug           # whole pipeline in a minute

# 3. train (locally, on Colab, or on the cluster)
python -m src.train --config configs/prefix_frozen.yaml
sbatch scripts/train.sbatch configs/prefix_lora.yaml                      # one job

# 4. evaluate
python -m src.evaluate --runs runs/prefix_frozen --decode greedy beam_w3 nucleus_p0.9_t0.7
python -m src.evaluate --runs runs/prefix_frozen --decode sweep
python -m src.evaluate --baseline retrieval
```

To try the pipeline on a laptop without the full download, add `--limit 64` to `download` and `prepare`.

## Running on the cluster (chained Slurm jobs)

`scripts/pipeline.sh` submits each config's training job and its evaluation job with
`--dependency=afterok` (evaluation starts only if training succeeded), then a report job with
`--dependency=afterany` that re-executes `notebooks/03_evaluation.ipynb` once everything has ended.

```bash
# stage 1, interactive GPU session: data, embeddings, sanity checks
srun --partition=courses-gpu --gres=gpu:v100-sxm2:1 --time=01:00:00 --pty /bin/bash
python -m src.data download && python -m src.data prepare && python -m src.data embed
python -m src.train --config configs/prefix_frozen.yaml --overfit-batch
python -m src.train --config configs/prefix_frozen.yaml --debug

# stage 2: main model + decoding sweep; check its captions before stage 3
bash scripts/pipeline.sh configs/prefix_frozen.yaml --sweep prefix_frozen

# stage 3: the other five experiments + baselines, all in parallel
bash scripts/pipeline.sh configs/prefix_lora.yaml configs/prefix_full.yaml \
     configs/xattn_frozen.yaml configs/xattn_lora.yaml configs/xattn_full.yaml --retrieval --blip2

DRY_RUN=1 bash scripts/pipeline.sh ...   # print the sbatch commands without submitting

# round 2: 3 seeds x 6 configs (40 epochs, early stopping), no-repeat decoding, decoding sweep,
# data-size runs, attention maps, caption -> image cycle, then the report notebook
bash scripts/round2.sh
squeue -u $USER                          # job states; a DependencyNeverSatisfied eval is cancelled automatically
```

Partition and module names in `scripts/*.sbatch` follow the Northeastern Explorer cluster (course partitions `courses-gpu` / `courses`); adjust them if yours differ.

## Experiments

| Config | Injection | Fine-tuning |
|---|---|---|
| `prefix_frozen` | prefix | GPT-2 frozen, mapping MLP only |
| `prefix_lora` | prefix | LoRA (r = 8) on self-attention |
| `prefix_full` | prefix | all GPT-2 weights (lr 2e-5) |
| `xattn_frozen` | cross-attention | GPT-2 frozen, new cross-attention layers train |
| `xattn_lora` | cross-attention | LoRA + cross-attention layers |
| `xattn_full` | cross-attention | all weights |

## Results

*To be filled in after Milestone 3.*

## References

- Mokady et al., *ClipCap: CLIP Prefix for Image Captioning*, 2021
- Radford et al., *Learning Transferable Visual Models From Natural Language Supervision*, 2021
- Hu et al., *LoRA: Low-Rank Adaptation of Large Language Models*, 2022
- Rohrbach et al., *Object Hallucination in Image Captioning*, 2018
