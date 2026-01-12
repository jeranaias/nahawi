# FASIH v3.0 - Exhaustive Arabic GEC Benchmark

## Overview

FASIH v3.0 is the most comprehensive Arabic Grammatical Error Correction benchmark available, covering **18 error categories** with **2,251 total samples**.

## Categories

| Category | Test | Description |
|----------|------|-------------|
| **Orthographic** | | |
| hamza | 105 | أ/إ/آ/ء placement |
| hamza_wasl | 127 | همزة الوصل (استخدام not إستخدام) |
| taa_marbuta | 104 | ة vs ه confusion |
| alif_maqsura | 106 | ى vs ي at word end |
| alif_madda | 110 | آ vs اا |
| dad_za | 112 | ض vs ظ confusion |
| dal_thal | 111 | د vs ذ confusion |
| tanwin | 85 | تنوين spelling |
| **Spacing** | | |
| space_join | 85 | Words incorrectly joined (عبدالله → عبد الله) |
| space_split | 85 | Words incorrectly split |
| **Agreement** | | |
| gender_agreement | 76 | Noun-adjective gender mismatch |
| number_agreement | 25 | Singular/plural mismatch |
| verb_agreement | 27 | Subject-verb agreement |
| definiteness | 85 | Article agreement (ال) |
| **Prepositions** | | |
| missing_prep | 110 | Required preposition missing |
| wrong_prep | 45 | Incorrect preposition choice |
| **Typos** | | |
| repeated_char | 42 | Doubled character |
| missing_char | 42 | Missing character |

## Files

```
v3/
├── test.json     # 1,482 test samples
├── dev.json      # 269 dev samples
├── rubric.json   # Category definitions
└── README.md
```

Plus: `../identity/test.json` - 500 correct sentences for false positive testing

## Sample Format

```json
{
  "id": "fasih-hamza_wasl-0001",
  "source": "إستخدام التكنولوجيا في التعليم",
  "target": "استخدام التكنولوجيا في التعليم",
  "category": "hamza_wasl",
  "verified": true,
  "source_corpus": "msa"
}
```

## Data Sources

| Source | Samples | Categories |
|--------|---------|------------|
| MSA Corpus (hunted) | ~1,200 | Orthographic, spacing, definiteness |
| Manual curation | ~150 | Agreement, prepositions |
| Generated (controlled) | ~100 | Typos |

## Evaluation

Primary metric: **F0.5** (precision-weighted)

```
F0.5 = (1.25 × P × R) / (0.25 × P + R)
```

Thresholds:
- **PASS**: F0.5 ≥ 90%
- **WORK**: F0.5 ≥ 80%
- **FAIL**: F0.5 < 80%

## Usage

```python
import json

# Load test set
test = json.load(open('fasih/v3/test.json', encoding='utf-8'))

# Evaluate
from collections import defaultdict
results = defaultdict(lambda: {'correct': 0, 'total': 0})

for sample in test:
    pred = your_model.correct(sample['source'])
    cat = sample['category']
    results[cat]['total'] += 1
    if pred == sample['target']:
        results[cat]['correct'] += 1

# Report
for cat, r in sorted(results.items()):
    acc = r['correct'] / r['total'] * 100
    print(f"{cat}: {acc:.1f}%")
```

## Version History

| Version | Changes |
|---------|---------|
| 3.0.0 | Added 10 new categories (hamza_wasl, space_*, agreement, typos) |
| 2.0.0 | Removed QALB, 100% MSA corpus |
| 1.0.0 | Initial release |
