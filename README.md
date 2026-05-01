# Qwen UCF-Crime Labeling

This project is being set up for Qwen-based labeling on the UCF-Crime dataset.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

Place the UCF-Crime dataset under:

```text
data/ucf-crime/
```

The manifest builder expects class folders when available, for example:

```text
data/ucf-crime/Arson/*.mp4
data/ucf-crime/Normal/*.mp4
```

If your downloaded dataset uses a different layout, update `configs/ucf_crime.yaml`.

## Build Manifests

```bash
python3 scripts/prepare_ucf_crime.py
```

Outputs are written to:

```text
data/processed/ucf-crime/manifest.jsonl
data/processed/ucf-crime/manifest.csv
```
