# FASIH v4.2 - Synthetic Arabic GEC Benchmark

## Overview

FASIH (Fluency Assessment System for Intuitive Arabic Handling) v4.2 is a synthetic benchmark
for evaluating Arabic Grammatical Error Correction (GEC) systems.

## Key Features

- **16 error categories** covering orthography, spelling, morphology, and syntax
- **1,600 total samples** (100 per category) with 80/20 test/dev split
- **High vocabulary diversity** (811 unique words from 4K+ Arabic roots)
- **Sentence variety** including questions, conditionals, and passive voice
- **Proper nouns** (names like أحمد, محمد, cities like القاهرة, الرياض)
- **Zero semantic implausibility** (no "hot plans" or "cold systems")
- **Strict holdout** (no overlap with training data)

## Error Categories

| Category | Description | Test | Dev |
|----------|-------------|------|-----|
| alif_madda | Alif Madda | 80 | 20 |
| alif_maqsura | Alif Maqsura | 80 | 20 |
| common_misspelling | Common Misspelling | 80 | 20 |
| definiteness | Definiteness | 80 | 20 |
| gender_agreement | Gender Agreement | 80 | 20 |
| hamza | Hamza | 80 | 20 |
| letter_dad_za | Letter Dad Za | 80 | 20 |
| letter_dal_thal | Letter Dal Thal | 80 | 20 |
| letter_sst | Letter Sst | 80 | 20 |
| mixed_all | Mixed All | 80 | 20 |
| mixed_orthographic | Mixed Orthographic | 80 | 20 |
| number_agreement | Number Agreement | 80 | 20 |
| stress_test | Stress Test | 80 | 20 |
| taa_marbuta | Taa Marbuta | 80 | 20 |
| verb_form | Verb Form | 80 | 20 |
| wrong_prep | Wrong Prep | 80 | 20 |

## Statistics

- **Test set**: 1280 samples
- **Dev set**: 320 samples
- **Total**: 1600 samples

## File Structure

```
fasih_v4.2/
├── fasih_test.json    # Test set (80%)
├── fasih_dev.json     # Dev set (20%)
├── fasih_rubric.json  # Category definitions and stats
└── README.md          # This file
```

## Sample Format

```json
{
  "source": "دعمت المؤسسه برنامج شراكه في مجال العدل",
  "target": "دعمت المؤسسة برنامج شراكة في مجال العدل",
  "error_type": "taa_marbuta",
  "correction": "المؤسسه → المؤسسة, شراكه → شراكة"
}
```

## Target Performance

- **Baseline**: ~60-70% F0.5
- **Good**: 80-85% F0.5
- **Production-ready**: 90%+ F0.5

## Version History

- **v4.2** (Jan 2025): V7 Ultimate synthetic data with comprehensive improvements
- **v4.1** (Jan 2025): Wikipedia-extracted real-world errors
- **v4.0** (Dec 2024): Initial release
