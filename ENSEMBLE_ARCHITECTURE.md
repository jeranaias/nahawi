# Nahawi Arabic GEC - Ensemble Architecture Plan

## Overview

This document outlines the architecture for a production-grade Arabic Grammatical Error Correction (GEC) system using a 10-model ensemble approach. The goal is to achieve **90%+ F0.5** on the QALB-2014 benchmark.

**Decision Date**: 2024-12-13
**Status**: Training in Progress

### Current Training Status (Updated: 2024-12-13 19:15 UTC)
| Model | Status | Progress | Est. Remaining |
|-------|--------|----------|----------------|
| GeneralGEC | Training | 3% | ~63h |
| HamzaFixer | Training | 3% | ~36h |
| SpaceFixer | Training | 3% | ~36h |
| SpellingFixer | Training | 8% | ~11h |
| DeletedWordFixer | Training | 0% | ~38h |
| MorphologyFixer | Training | 3% | ~2h |
| TaaMarbutaFixer | Complete | Rule-based | N/A |
| AlifMaksuraFixer | Complete | Rule-based | N/A |
| PunctuationFixer | Complete | Rule-based | N/A |
| RepeatedWordFixer | Complete | Rule-based | N/A |

### Preliminary Evaluation (Rule-based only)
- F0.5: 1.41% (rule-based models only, neural pending)
- Orchestrator: All tests passing

---

## Why Ensemble?

### Previous Approach Limitations
- Single 15MB model achieved ~38% F0.5
- Character-aware improvements projected to reach 55-65% max
- State-of-the-art on QALB is 67-72% with massive pretrained models

### Ensemble Advantages
1. **Specialization**: Each model becomes expert at one error type
2. **Precision**: Specialized models have fewer false positives
3. **Modularity**: Can improve/replace individual components
4. **Coverage**: Different architectures catch different patterns

---

## Architecture

### 10-Model Ensemble

| # | Model Name | Error Type | Base Architecture | Est. Size | Priority |
|---|------------|------------|-------------------|-----------|----------|
| 1 | **HamzaFixer** | أ/إ/ا/آ/ء/ؤ/ئ confusion | AraBART fine-tuned | 500MB | HIGH |
| 2 | **SpaceFixer** | Merge/split word errors | AraBART fine-tuned | 500MB | HIGH |
| 3 | **TaaMarbutaFixer** | ة ↔ ه confusion | Small classifier | 10MB | MEDIUM |
| 4 | **AlifMaksuraFixer** | ى ↔ ي confusion | Small classifier | 10MB | MEDIUM |
| 5 | **PunctuationFixer** | Arabic punctuation ،.؛؟ | Rule-based + small NN | 5MB | LOW |
| 6 | **DeletedWordFixer** | Missing common words (و/في/من) | AraBART fine-tuned | 500MB | MEDIUM |
| 7 | **RepeatedWordFixer** | Duplicate word detection | Simple detector | 2MB | LOW |
| 8 | **SpellingFixer** | General character-level errors | AraBART fine-tuned | 500MB | MEDIUM |
| 9 | **MorphologyFixer** | Verb/noun agreement, gender | CAMeLBERT | 400MB | MEDIUM |
| 10 | **GeneralGEC** | Catch-all for remaining errors | AraBART fine-tuned | 500MB | HIGH |

### Effective Size
- Naive total: ~3GB
- With shared base weights: ~600-800MB
- Quantized (INT8): ~200-300MB

---

## Error Distribution (QALB-2014)

Understanding error distribution guides model priority:

| Error Type | Frequency | Covered By |
|------------|-----------|------------|
| Edit (character-level) | 55% | HamzaFixer, SpellingFixer, TaaMarbutaFixer, AlifMaksuraFixer |
| Add (missing content) | 32% | SpaceFixer, DeletedWordFixer |
| Merge (incorrect splits) | 6% | SpaceFixer |
| Split (incorrect merges) | 3.5% | SpaceFixer |
| Delete (extra content) | 2% | RepeatedWordFixer, GeneralGEC |
| Other | 1.5% | GeneralGEC, MorphologyFixer |

**Key Insight**: HamzaFixer + SpaceFixer + GeneralGEC covers ~80% of errors.

---

## Licensing Analysis

### Safe to Use Commercially

| Resource | License | Commercial Use | Notes |
|----------|---------|----------------|-------|
| AraBART | Apache 2.0 | ✅ YES | Include license notice |
| AraBERT | Apache 2.0 | ✅ YES | Include license notice |
| CAMeLBERT | MIT | ✅ YES | Include copyright |
| CAMeL Tools | MIT | ✅ YES | Include copyright |
| Hugging Face Transformers | Apache 2.0 | ✅ YES | Standard |

### Caution Required

| Resource | License | Commercial Use | Notes |
|----------|---------|----------------|-------|
| QALB Dataset | LDC Research | ⚠️ GREY AREA | Use for eval only, not training |

### Our Strategy
- **Training data**: Wikipedia + synthetic errors (fully permissive)
- **Evaluation only**: QALB dataset (no legal exposure)
- **Base models**: Apache 2.0 / MIT licensed only

---

## Implementation Plan

### Phase 1: Foundation (Days 1-3)
```
nahawi_ensemble/
├── __init__.py
├── orchestrator.py          # Routes text through models, combines outputs
├── config.py                # Model configs, paths, hyperparameters
├── data/
│   ├── synthetic.py         # Synthetic data generation
│   └── loaders.py           # Data loading utilities
├── models/
│   ├── __init__.py
│   ├── base.py              # BaseGECModel abstract class
│   ├── arabart_base.py      # AraBART fine-tuning utilities
│   └── general_gec.py       # Model #10: GeneralGEC
├── evaluation/
│   ├── m2_scorer.py         # M2 format evaluation
│   └── metrics.py           # F0.5, precision, recall
└── scripts/
    ├── train_general.py     # Train GeneralGEC
    ├── train_hamza.py       # Train HamzaFixer
    ├── train_space.py       # Train SpaceFixer
    └── evaluate.py          # Full ensemble evaluation
```

### Phase 2: Core Models (Days 4-7)
1. Train GeneralGEC on synthetic data
2. Evaluate baseline ensemble (single model)
3. Train HamzaFixer with hamza-focused data
4. Train SpaceFixer with space-focused data
5. Evaluate 3-model ensemble

### Phase 3: Specialized Models (Days 8-14)
1. Build rule-based TaaMarbutaFixer
2. Build rule-based AlifMaksuraFixer
3. Build rule-based PunctuationFixer
4. Build RepeatedWordFixer
5. Train DeletedWordFixer
6. Train SpellingFixer
7. Integrate CAMeLBERT MorphologyFixer

### Phase 4: Optimization (Days 15-21)
1. Tune ensemble weights on dev set
2. Add confidence thresholding
3. Implement cascading (fast models first)
4. Quantize models (INT8)
5. Final evaluation on QALB test

---

## Ensemble Orchestration Strategy

### Option A: Sequential Pipeline
```
Input → Model1 → Model2 → ... → Model10 → Output
```
- Simple but slow
- Each model sees previous corrections

### Option B: Parallel + Voting
```
Input → [Model1, Model2, ..., Model10] → Vote/Merge → Output
```
- Fast (parallel execution)
- Need smart merging strategy

### Option C: Cascading (Recommended)
```
Input → FastModels(3,4,5,7) → ConfidentFixes
      → SlowModels(1,2,6,8,9,10) → RemainingFixes
      → Merge → Output
```
- Best latency/accuracy tradeoff
- Rule-based models first (instant)
- Heavy models only when needed

### Merging Strategy
1. **High confidence wins**: If any model is >95% confident, use its correction
2. **Specialist priority**: For hamza errors, trust HamzaFixer over GeneralGEC
3. **Conservative default**: When in doubt, don't change (precision over recall)

---

## Data Strategy

### Training Data Sources

| Source | Size | Use |
|--------|------|-----|
| Arabic Wikipedia | 370K sentences | Base clean text |
| Synthetic errors (v4 generator) | 500K+ pairs | All models |
| Hamza-focused synthetic | 100K pairs | HamzaFixer |
| Space-focused synthetic | 100K pairs | SpaceFixer |
| QALB train (careful) | 19K pairs | Fine-tuning only, not base |

### Synthetic Data Generation
- Use `generate_qalb_synthetic_v4.py` as base
- Create specialized generators for each error type
- 10M+ total training examples across all models

---

## Evaluation Plan

### Metrics
- **Primary**: F0.5 (precision-weighted, standard for GEC)
- **Secondary**: Precision, Recall, Exact Match
- **Per-model**: Error-type-specific accuracy

### Benchmarks
- QALB-2014 Dev: Hyperparameter tuning
- QALB-2014 Test: Final evaluation
- QALB-2015: Generalization test

### Targets

| Configuration | Target F0.5 |
|---------------|-------------|
| GeneralGEC only | 65-70% |
| + HamzaFixer | 72-75% |
| + SpaceFixer | 75-78% |
| Full 10-model | 85-90% |

---

## Technical Requirements

### Hardware
- GPU: 16GB+ VRAM for AraBART fine-tuning
- RAM: 32GB+ for data processing
- Storage: 50GB+ for models and data

### Software
```
torch>=2.0
transformers>=4.30
arabert  # or load from HuggingFace
camel-tools
tokenizers
datasets
```

### Model Sources
- AraBART: `moussaKam/AraBART`
- AraBERT: `aubmindlab/bert-base-arabertv2`
- CAMeLBERT: `CAMeL-Lab/bert-base-arabic-camelbert-mix`

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| AraBART underperforms | HIGH | Fall back to AraBERT, try multiple bases |
| Ensemble too slow | MEDIUM | Aggressive caching, quantization, cascading |
| Overfitting to QALB | HIGH | Train on synthetic only, eval on QALB |
| Legal issues | HIGH | Apache 2.0/MIT only, no QALB in training |
| GPU memory limits | MEDIUM | Gradient checkpointing, smaller batch sizes |

---

## Success Criteria

### Minimum Viable Product
- [ ] 3-model ensemble working (GeneralGEC + HamzaFixer + SpaceFixer)
- [ ] F0.5 > 70% on QALB dev
- [ ] Inference < 1 second per sentence

### Full Success
- [ ] 10-model ensemble complete
- [ ] F0.5 > 85% on QALB dev
- [ ] F0.5 > 80% on QALB test
- [ ] Quantized models < 300MB total
- [ ] Inference < 500ms per sentence

---

## File Inventory

### Existing (from previous phases)
- `qalb_real_data/` - QALB train/dev TSV files
- `arabic_wiki/sentences.txt` - 370K Wikipedia sentences
- `qalb_correct_to_errors.json` - 31K error patterns
- `generate_qalb_synthetic_v4.py` - Synthetic data generator

### To Create
- `nahawi_ensemble/` - Main package (this plan)
- Trained model checkpoints
- Evaluation results

---

## Notes & Decisions Log

### 2024-12-13
- Decided to abandon 30MB size constraint
- Chose 10-model ensemble approach
- Prioritized legal safety (Apache 2.0/MIT only)
- QALB will be eval-only, not training data
- AraBART selected as primary base model

---

## References

- QALB-2014 Shared Task: https://aclanthology.org/W14-3605/
- AraBART: https://huggingface.co/moussaKam/AraBART
- CAMeL Tools: https://github.com/CAMeL-Lab/camel_tools
- GEC Metrics: https://github.com/nusnlp/m2scorer
