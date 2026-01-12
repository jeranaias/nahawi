# Nahawi Local Repository Inventory

**Date**: January 10, 2025
**Status**: Ready for future training

---

## Models

### Base Model
| File | Size | Location | Status |
|------|------|----------|--------|
| fasih_v15_model.pt | 1.4GB | models/base/ | Downloaded |

### LoRA Weights (Punct-aware)
| File | Size | Location | Status |
|------|------|----------|--------|
| best_lora.pt | 30MB | models/punct_aware_lora/ | Downloaded |
| epoch_1.pt | 30MB | models/punct_aware_lora/ | Downloaded |
| epoch_2.pt | 30MB | models/punct_aware_lora/ | Downloaded |

### Tokenizer
| File | Size | Location | Status |
|------|------|----------|--------|
| nahawi_spm.model | 900KB | / (root) | Present |

---

## Training Data

### QALB Real Data (GOLD)
| File | Size | Location | Status |
|------|------|----------|--------|
| qalb_real_train.json | 40MB | data/ | Downloaded |
| qalb_real_dev.json | 2.2MB | data/ | Downloaded |

### Hamza Augmentation
| File | Size | Location | Status |
|------|------|----------|--------|
| hamza_augmentation.json | ~150MB | data/ | Downloaded |

---

## Pretraining Corpus

### MSA Corpus (for base model training)
| File | Size | Location | Status |
|------|------|----------|--------|
| msa_corpus_full.txt | 6.2GB | corpus/ | Present |

**Composition**:
- Arabic Wikipedia (clean)
- UN corpus
- News (Leipzig)
- Shamela

---

## Benchmarks

### FASIH Benchmark (Our custom benchmark)
| Version | Location | Samples |
|---------|----------|---------|
| v4.1 | benchmark/fasih_v4.1/ | Test + Dev |
| v4.2 | benchmark/fasih_v4.2/ | Test + Dev |
| v5 | benchmark/fasih_v5/ | Test + Dev |
| v5.1 | benchmark/fasih_v5.1/ | Test + Dev |

### Competitive Benchmarks
| File | Location |
|------|----------|
| competitive_test_set.json | benchmark/ |
| Analysis report | benchmark/competitive/ |

---

## Training Scripts

### LoRA Training Pipeline
| Script | Location | Purpose |
|--------|----------|---------|
| prepare_punct_aware_data.py | lora_experiment/ | Data prep |
| train_lora_punct_aware.py | lora_experiment/ | Training |

### Legacy Scripts (in archive)
- Various train_fasih_*.py
- Various generate_*.py
- Various eval_*.py

---

## Web Demo

| Component | Location | Status |
|-----------|----------|--------|
| Backend (FastAPI) | web/backend/ | Updated for LoRA |
| Frontend (React) | web/frontend/ | Ready |

---

## What's Ready for Next Training

### For LoRA Fine-tuning (quick experiments)
1. Base model (fasih_v15_model.pt)
2. QALB real data (train/dev)
3. Hamza augmentation
4. Training scripts

### For New Base Model Pretraining
1. MSA corpus (6.2GB)
2. Tokenizer (nahawi_spm.model)
3. Need to create: pretrain script

---

## Remote Server Files (Not Downloaded)

Still on remote server (192.222.50.72):
- Additional step checkpoints (step_*.pt)
- Synthetic training data (66M pairs, 19GB)
- V8/V9 training data variants
- Ultimate pretrain checkpoints (700M model, partial)

**To download later if needed**:
```bash
# Synthetic data (large - 19GB)
scp -i nahawi.pem ubuntu@192.222.50.72:/home/ubuntu/nahawi/data/ultimate/train_full.json ./data/

# Additional models
scp -i nahawi.pem ubuntu@192.222.50.72:/home/ubuntu/nahawi/gec_clean/epoch_11.pt ./models/
```

---

## Directory Structure

```
c:\nahawi\
├── README.md               # Main documentation
├── CLAUDE.md              # Dev context
├── nahawi_spm.model       # Tokenizer
│
├── nahawi/                # Python package
│   ├── __init__.py
│   └── model.py
│
├── models/
│   ├── base/
│   │   └── fasih_v15_model.pt
│   └── punct_aware_lora/
│       ├── best_lora.pt
│       ├── epoch_1.pt
│       └── epoch_2.pt
│
├── data/
│   ├── qalb_real_train.json
│   ├── qalb_real_dev.json
│   └── hamza_augmentation.json
│
├── corpus/
│   └── msa_corpus_full.txt    # 6.2GB MSA
│
├── benchmark/
│   ├── fasih_v4.1/
│   ├── fasih_v4.2/
│   ├── fasih_v5/
│   ├── fasih_v5.1/
│   └── competitive/
│
├── web/
│   ├── backend/
│   └── frontend/
│
├── scripts/               # Training scripts
├── docs/                  # Documentation
├── lora_experiment/       # LoRA training
└── archive/               # Old experiments
```

---

*Last updated: January 10, 2025*
