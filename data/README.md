# Nahawi Training Data

This folder contains training data for Nahawi. Data files are not included in the repository (too large).

## Required Files

### QALB Real Data (Gold Standard)
```bash
# Download from remote server (if you have access)
scp -i nahawi.pem ubuntu@192.222.50.72:/home/ubuntu/nahawi/data/qalb_real_train.json ./
scp -i nahawi.pem ubuntu@192.222.50.72:/home/ubuntu/nahawi/data/qalb_real_dev.json ./
```

| File | Size | Description |
|------|------|-------------|
| qalb_real_train.json | 40MB | 36,771 training pairs |
| qalb_real_dev.json | 2.2MB | ~2,000 dev pairs |

### Hamza Augmentation
```bash
scp -i nahawi.pem ubuntu@192.222.50.72:/home/ubuntu/nahawi/data/hamza_augmentation.json ./
```

| File | Size | Description |
|------|------|-------------|
| hamza_augmentation.json | ~150MB | MSA corpus with hamza errors |

## Data Format

All JSON files use the following format:
```json
[
  {
    "source": "اعلنت الحكومه عن خطه جديده",
    "target": "أعلنت الحكومة عن خطة جديدة"
  },
  ...
]
```

## Training Data Composition (Punct-aware LoRA)

For the punct-aware LoRA training that achieves 78.5% F0.5:

| Source | Count | Ratio | Purpose |
|--------|-------|-------|---------|
| QALB x50 (with punct) | 1,838,550 | 86.5% | Teaches punctuation |
| Synthetic (no punct) | 150,000 | 7.1% | Teaches content errors |
| Hamza aug (no punct) | 137,972 | 6.5% | Teaches hamza patterns |
| **Total** | 2,126,522 | 100% | |

Key insight: Strip punctuation from synthetic data so the model learns punct patterns only from real QALB data.

## Preparation Script

Use `lora_experiment/prepare_punct_aware_data.py` to create the combined training dataset.
