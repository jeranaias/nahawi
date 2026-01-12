# FASIH v4.1 Benchmark (فصيح)

**FASIH** (فصيح - "eloquent" in Arabic) is a comprehensive Arabic Grammatical Error Correction benchmark designed to evaluate **real grammatical errors** found in formal Modern Standard Arabic writing.

## Why FASIH?

QALB (Qatar Arabic Language Bank) has significant limitations:
- **38% punctuation errors** (arbitrary comma placement)
- **39% dialectal content** (forum comments, not MSA)
- **Inconsistent annotations** (same errors annotated differently)

FASIH addresses these issues by focusing on **pure MSA grammatical errors** extracted from Wikipedia.

## Benchmark Statistics

| Split | Samples | Source |
|-------|---------|--------|
| Test | 2,210 | Wikipedia MSA |
| Dev | 500 | Wikipedia MSA |

## Error Categories (13 total)

### Orthographic (4)
1. **hamza** - Hamza placement (أ/إ/آ/ء/ؤ/ئ)
2. **alif_maqsura** - Alif maqsura vs ya (ى/ي)
3. **taa_marbuta** - Taa marbuta vs ha (ة/ه)
4. **definiteness** - Article agreement (ال)

### Letter Confusion (4)
5. **letter_confusion_س_ص** - Seen vs Sad
6. **letter_confusion_ض_ظ** - Dad vs Za
7. **letter_confusion_د_ذ** - Dal vs Dhal
8. **letter_confusion_ت_ط** - Ta vs Ta emphatic

### Morphological (3)
9. **verb_conjugation** - Verb form errors
10. **number_agreement** - Singular/dual/plural
11. **gender_agreement** - Masculine/feminine agreement

### Syntactic (2)
12. **missing_preposition** - Missing required prepositions
13. **wrong_preposition** - Incorrect preposition usage

## File Format

### fasih_test.json / fasih_dev.json
```json
[
  {
    "source": "ذهب الي المدرسه",
    "target": "ذهب إلى المدرسة",
    "error_type": "alif_maqsura"
  },
  ...
]
```

### fasih_rubric.json
Contains the official category definitions and evaluation thresholds.

## Evaluation Criteria

- **PASS**: F0.5 >= 90%
- **WORK**: F0.5 >= 80%
- **FAIL**: F0.5 < 80%

## Current Best Results (V5 + Rules Hybrid)

| Category | F0.5 | Status |
|----------|------|--------|
| Overall | **94.7%** | **11/13 PASS** |

## Usage

```python
import json

# Load test data
with open('fasih_test.json') as f:
    test_data = json.load(f)

print(f"Test samples: {len(test_data)}")

# Evaluate your model
for sample in test_data:
    source = sample['source']
    target = sample['target']
    error_type = sample['error_type']

    # Your model prediction
    prediction = your_model.correct(source)

    # Compare prediction vs target
```

## Citation

```bibtex
@dataset{fasih2025,
  title={FASIH: A Comprehensive Arabic Grammar Benchmark},
  author={[Author]},
  year={2025},
  note={2,210 test samples, 13 error categories}
}
```

## License

Research use permitted. Commercial licensing available upon request.
