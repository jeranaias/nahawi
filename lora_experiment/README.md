# LoRA Experiment Scripts

Development and experimental scripts from the LoRA fine-tuning research.

## Overview

This directory contains the evolution of our LoRA training approach, from initial experiments to the final punct-aware solution that achieved **78.52% F0.5**.

## Key Discovery

**The Problem**: Base model achieved 87.63% F0.5 without punctuation but only 71.82% with punctuation. The 16-point gap was caused by learning wrong punctuation patterns from synthetic data.

**The Solution**: Train with punct-aware data where:
- QALB (86.5%) teaches correct punctuation patterns
- Synthetic/hamza data has punctuation stripped to teach content only

## Script Categories

### Data Preparation
| Script | Purpose |
|--------|---------|
| `sample_balanced_300k.py` | Early balanced sampling (1:4 ratio) |
| `sample_stratified_500k.py` | Stratified by error type |
| `generate_hamza_augmentation.py` | Generate hamza-specific training data |
| `merge_augmentation.py` | Combine training data sources |
| `prepare_punct_aware_data.py` | **Final**: Punct-aware data preparation |
| `extract_real_patterns.py` | Extract patterns from QALB |

### Training Scripts
| Script | Purpose |
|--------|---------|
| `train_lora.py` | Initial LoRA training |
| `train_lora_v2.py` | Improved with better hyperparams |
| `train_lora_hamza.py` | Hamza-focused LoRA |
| `train_lora_hamza_continue.py` | Continue hamza training |
| `train_lora_round3.py` | Third iteration |
| `train_lora_round4.py` | Fourth iteration |
| `train_lora_punct_aware.py` | **Final**: Punct-aware LoRA training |

### Punctuation Experiments
| Script | Purpose |
|--------|---------|
| `train_punct_classifier.py` | Separate punct classifier |
| `train_punct_classifier_v2.py` | Improved punct classifier |
| `train_qalb_punct_classifier.py` | QALB-only punct classifier |
| `train_qalb_punct_v2.py` | V2 QALB punct |
| `train_qalb_punct_v3.py` | V3 QALB punct |
| `punct_analysis.py` | Analyze punct patterns |
| `extract_punct_patterns.py` | Extract punct rules |
| `eval_punct_rules.py` | Evaluate rule-based punct |
| `debug_punct_pipeline.py` | Debug punct issues |

### Evaluation Scripts
| Script | Purpose |
|--------|---------|
| `eval_lora.py` | Evaluate LoRA checkpoints |
| `eval_hamza_lora.py` | Evaluate hamza LoRA |
| `eval_combined.py` | Evaluate combined system |
| `eval_two_pass.py` | Two-pass evaluation |
| `eval_two_pass_full.py` | Full two-pass eval |
| `eval_with_punct_classifier.py` | Eval with punct classifier |
| `eval_punct_fast.py` | Fast punct evaluation |
| `eval_punct_fix_inplace.py` | Punct fix in-place |
| `eval_round3.py` | Round 3 evaluation |

### Analysis Scripts
| Script | Purpose |
|--------|---------|
| `diagnose_punct.py` | Diagnose punct problems |
| `error_analysis.py` | Analyze error distribution |
| `error_analysis_fixed.py` | Fixed error analysis |
| `classify_error_fixed.py` | Error classification |
| `audit_hamza_other.py` | Audit hamza/other errors |
| `clean_fasih_benchmark.py` | Clean FASIH benchmark |
| `generate_targeted_fixes.py` | Generate targeted training data |

## Evolution Timeline

1. **Initial LoRA** (`train_lora.py`): Balanced data, 300K samples
2. **Hamza Focus** (`train_lora_hamza.py`): Hamza-heavy training
3. **Two-Pass** (`eval_two_pass.py`): Separate content + punct
4. **Punct Classifier** (`train_punct_classifier_v2.py`): Dedicated punct model
5. **Punct-Aware** (`train_lora_punct_aware.py`): **Winner** - single model, smart data

## Key Files for Production

For production use, see `scripts/` instead:
- `scripts/prepare_punct_aware_data.py`
- `scripts/train_lora_punct_aware.py`

## Results History

| Approach | F0.5 (wp) | F0.5 (np) | Notes |
|----------|-----------|-----------|-------|
| Base V15 | 71.82% | 87.63% | Starting point |
| + LoRA v1 | 65% | 85% | Worse with punct |
| + Hamza LoRA | 71.34% | 88.53% | Good content |
| + Punct classifier | 72.1% | - | Marginal gain |
| **+ Punct-aware LoRA** | **78.52%** | **91.95%** | **Best** |

## Running Experiments

Most scripts expect to run on the remote server with GPU:

```bash
ssh -i nahawi.pem ubuntu@192.222.50.72
cd /home/ubuntu/nahawi
python lora_experiment/train_lora_punct_aware.py
```

## Configuration

Experiments use these common settings:

```python
# Model architecture (fixed)
d_model: 768
nhead: 12
num_layers: 6 + 6
vocab_size: 32000

# LoRA config (tuned)
lora_rank: 64
lora_alpha: 128
lora_dropout: 0.05

# Training (tuned for GH200)
batch_size: 224
learning_rate: 3e-4
warmup_steps: 200
```

## See Also

- `scripts/` - Production-ready training scripts
- `models/README.md` - Model weights and checkpoints
- `benchmark/` - Evaluation benchmarks
- `CLAUDE.md` - Full project context
