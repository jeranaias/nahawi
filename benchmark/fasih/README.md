# FASIH Arabic GEC Benchmark

**FASIH** (فصيح - "eloquent") is a world-class benchmark for Arabic Grammatical Error Correction built **100% from real MSA corpus** with zero synthetic or QALB data.

## Version 2.0 - Pure MSA

This version removes all QALB-sourced samples to ensure:
- **Zero training contamination** - No overlap with QALB train data
- **Pure MSA quality** - All samples from Wikipedia, UN, News corpus
- **Expert curation** - Manual preposition examples verified by linguists

## Benchmark Suite

```
fasih/
├── core/           # Orthographic + preposition errors
├── full/           # Complete error coverage
├── identity/       # Correct text (false positive testing)
└── rubric.json     # Category definitions & metrics
```

### FASIH-Core (820 test + 135 dev)

**Purpose**: Orthographic and preposition error correction

| Category | Test Samples | Description |
|----------|--------------|-------------|
| hamza | 124 | أ/إ/آ/ء placement |
| taa_marbuta | 123 | ة vs ه confusion |
| alif_maqsura | 125 | ى vs ي at word end |
| alif_madda | 130 | آ vs اا |
| dad_za | 132 | ض vs ظ confusion |
| dal_thal | 131 | د vs ذ confusion |
| missing_prep | 30 | Required preposition missing |
| wrong_prep | 25 | Incorrect preposition choice |
| **Total** | **820** | |

**Source**: 100% MSA Corpus (Wikipedia, UN, News) + Manual Curation

### FASIH-Full (877 test + 143 dev)

**Purpose**: Complete error coverage with more preposition samples

| Category | Test Samples |
|----------|--------------|
| hamza | 125 |
| taa_marbuta | 120 |
| alif_maqsura | 129 |
| alif_madda | 133 |
| dad_za | 132 |
| dal_thal | 125 |
| missing_prep | 88 |
| wrong_prep | 25 |
| **Total** | **877** |

**Source**: 100% MSA Corpus + Manual Curation (QALB-free)

### FASIH-Identity (500 samples)

**Purpose**: False positive testing - models should return these unchanged

| Length | Samples |
|--------|---------|
| Short (5-10 words) | ~25 |
| Medium (10-20 words) | ~300 |
| Long (20+ words) | ~175 |
| **Total** | **500** |

**Source**: Clean MSA corpus sentences (verified correct)

## Quality Assurance

| Metric | Value |
|--------|-------|
| QALB samples | 0 |
| Synthetic samples | 0 |
| Verified samples | 2,197 |
| Manual preposition samples | 75 |

## File Format

### Error samples (core/, full/)
```json
{
  "id": "core-hamza-0001",
  "source": "اعلنت الحكومة عن خطة جديدة",
  "target": "أعلنت الحكومة عن خطة جديدة",
  "category": "hamza",
  "verified": true,
  "source_corpus": "wikipedia"
}
```

### Identity samples
```json
{
  "id": "identity-0001",
  "source": "تعد المملكة العربية السعودية من أكبر الدول العربية مساحة",
  "target": "تعد المملكة العربية السعودية من أكبر الدول العربية مساحة",
  "verified": true
}
```

## Evaluation

### Primary Metric: F0.5

```
F0.5 = (1.25 × Precision × Recall) / (0.25 × Precision + Recall)
```

Precision-weighted because **not introducing new errors** matters more than catching all errors.

### Thresholds

| Grade | F0.5 Score |
|-------|------------|
| **PASS** | ≥ 90% |
| **WORK** | ≥ 80% |
| **FAIL** | < 80% |

### False Positive Target

Identity samples should have **< 5%** modification rate.

## Usage

```python
import json

# Load benchmark
with open('fasih/core/test.json', encoding='utf-8') as f:
    test_data = json.load(f)

# Evaluate
correct = 0
for sample in test_data:
    prediction = your_model.correct(sample['source'])
    if prediction == sample['target']:
        correct += 1

accuracy = correct / len(test_data)
print(f"Accuracy: {accuracy:.2%}")
```

## Category Definitions

### Orthographic

| Category | Arabic | Example |
|----------|--------|---------|
| `hamza` | همزة | اعلان → إعلان |
| `taa_marbuta` | تاء مربوطة | المدرسه → المدرسة |
| `alif_maqsura` | ألف مقصورة | علي → على |
| `alif_madda` | ألف المد | الان → الآن |
| `dad_za` | ض/ظ | نضر → نظر |
| `dal_thal` | د/ذ | هدا → هذا |

### Prepositions

| Category | Arabic | Example |
|----------|--------|---------|
| `missing_prep` | حرف جر مفقود | يبحث المعلومات → يبحث عن المعلومات |
| `wrong_prep` | حرف جر خاطئ | يعتمد من → يعتمد على |

## Version History

| Version | Changes |
|---------|---------|
| 2.0.0 | Removed all QALB samples, 100% MSA corpus |
| 1.0.0 | Initial release with QALB grammar samples |

## Citation

```bibtex
@dataset{fasih2025,
  title={FASIH: A Pure MSA Arabic GEC Benchmark},
  author={Nahawi Team},
  year={2025},
  version={2.0.0},
  url={https://github.com/nahawi/benchmark}
}
```

## License

CC-BY-4.0 - Free for research and commercial use with attribution.
