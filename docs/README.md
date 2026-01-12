# Nahawi Documentation

Project documentation and reference materials.

## Contents

| Document | Description |
|----------|-------------|
| [LOCAL_INVENTORY.md](LOCAL_INVENTORY.md) | Complete inventory of local files (models, data, benchmarks) |

## Project Documentation

Main documentation files are in the project root:

| File | Description |
|------|-------------|
| [README.md](../README.md) | Project overview, quick start, performance |
| [CLAUDE.md](../CLAUDE.md) | Development context and technical details |

## Section READMEs

Each major directory has its own README:

| Directory | README |
|-----------|--------|
| [models/](../models/README.md) | Model weights, architecture, download instructions |
| [data/](../data/README.md) | Training data format and sources |
| [benchmark/](../benchmark/README.md) | FASIH benchmark and competitive analysis |
| [corpus/](../corpus/README.md) | Pretraining corpus |
| [scripts/](../scripts/README.md) | Production training scripts |
| [lora_experiment/](../lora_experiment/README.md) | Experimental training scripts |
| [web/](../web/README.md) | Web demo (FastAPI + React) |

## Key Technical Documents

### In CLAUDE.md
- Model architecture (124M params, 768d, 12 heads, 6+6 layers)
- LoRA configuration (rank=64, alpha=128)
- Training data strategy (punct-aware, 86.5% QALB)
- Evaluation methodology (F0.5 with punctuation)
- Performance history and lessons learned

### In benchmark/README.md
- FASIH benchmark design philosophy
- Competitive analysis vs Word/Google
- Error category definitions

## Quick Reference

### Current Best Model
```
Base: fasih_v15_model.pt (71.82% F0.5)
LoRA: best_lora.pt (+6.7 points → 78.52% F0.5)
Gap to SOTA: 4.1 points (82.63%)
```

### Key Insight
> QALB (86.5% of training) teaches punctuation.
> Stripped synthetic/hamza teaches content.
> This separation gained 6.7 F0.5 points.

### Model Loading
```python
from nahawi import NahawiModel
model = NahawiModel()
model.load()
corrected, corrections = model.correct("اعلنت الحكومه")
```

## Contributing

See the main README for contribution guidelines.
