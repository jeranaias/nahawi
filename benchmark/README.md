# Nahawi Benchmark Suite

This directory contains the evaluation benchmarks and competitive analysis for the Nahawi Arabic GEC system.

## Directory Structure

```
benchmark/
├── README.md                           # This file
├── fasih_v4.1/                         # FASIH benchmark (primary)
│   ├── fasih_test.json                 # Test set (433 samples)
│   ├── fasih_dev.json                  # Dev set (96 samples)
│   ├── fasih_rubric.json               # Evaluation rubric & error types
│   └── README.md                       # FASIH documentation
├── competitive/                        # Word/Google comparison
│   ├── competitive_test_set_v4.json    # 100 sentences, 520 errors
│   ├── test_sentences.txt              # Plain text for testing
│   ├── test_sentences_annotated.txt    # With error annotations
│   ├── scoring_sheet.tsv               # Manual testing scoresheet
│   ├── run_competitive_benchmark.py    # Automated Nahawi runner
│   ├── COMPETITIVE_TEST_PROTOCOL.md    # Testing methodology
│   ├── COMPETITIVE_ANALYSIS_REPORT.md  # Full results
│   ├── screenshots_word/               # MS Word testing evidence (5 images)
│   └── screenshots_google/             # Google Docs testing evidence (3 images)
└── results/                            # Benchmark results
    └── nahawi_competitive_results.json # Nahawi output
```

## Quick Links

- [FASIH Benchmark](fasih_v4.1/README.md) - Our primary evaluation benchmark
- [Competitive Analysis](competitive/COMPETITIVE_ANALYSIS_REPORT.md) - Nahawi vs Word vs Google
- [Test Protocol](competitive/COMPETITIVE_TEST_PROTOCOL.md) - How we tested

## Results Summary

### FASIH v4.1 (13 Error Categories)

| Metric | Nahawi V5 |
|--------|-----------|
| Overall F0.5 | **95.1%** |
| Categories PASS (>90%) | 12/13 |
| Categories WORK (<90%) | 1/13 |

### Competitive Comparison (100 Sentences, 520 Errors)

| System | Detection | Correction | Improvement |
|--------|-----------|------------|-------------|
| **Nahawi V5** | 95.2% | **90.0%** | Baseline |
| Microsoft Word | 36.0% | 28.8% | 3.1x behind |
| Google Docs | 30.0% | 24.0% | 3.8x behind |

**Key Finding:** Word and Google detect **0%** of grammar errors (gender agreement, verb conjugation, prepositions, etc.) - they only catch spelling.

## Running Benchmarks

### FASIH Evaluation

```bash
# Full FASIH benchmark
python evaluation/eval_hybrid.py --benchmark fasih

# Quick evaluation (first 100 samples)
python evaluation/eval_hybrid.py --benchmark fasih --max-samples 100
```

### Competitive Benchmark

```bash
# Run Nahawi on competitive test set
cd benchmark
python run_competitive_benchmark.py
```

## Benchmark Design Philosophy

### Why We Created FASIH

Existing benchmarks like QALB have significant issues:
- 40% punctuation insertion (not grammar)
- 15% dialect normalization (different task)
- Inconsistent annotations
- Social media quality text

FASIH focuses on **real MSA grammar errors** that professional Arabic writers make:
- Hamza confusion (أ/إ/ا)
- Taa marbuta (ة vs ه)
- Alif maqsura (ى vs ي)
- Gender/number agreement
- Verb conjugation
- Preposition usage
- Definiteness

### Why We Test Against Word/Google

Commercial grammar checkers set user expectations. Our competitive testing demonstrates:
1. **Gap in the market** - No commercial solution handles Arabic grammar
2. **Technical superiority** - Nahawi catches 3x more errors
3. **Expert-proof results** - All 13 FASIH error types tested

## Error Categories

| Category | Types | Description |
|----------|-------|-------------|
| **Orthography** | hamza, taa_marbuta, alif_maqsura | Arabic script conventions |
| **Spelling** | letter confusions (د/ذ, ض/ظ) | Similar letter substitutions |
| **Morphology** | gender_agreement, number_agreement | Noun-adjective agreement |
| **Syntax** | missing_preposition, wrong_preposition | Prepositional usage |
| **Verb** | verb_conjugation | Subject-verb agreement |
| **Article** | definiteness | Tanween/definite article usage |

## Citation

If you use these benchmarks in your research:

```bibtex
@software{nahawi_benchmark2025,
  title = {FASIH: A Benchmark for Arabic Grammatical Error Correction},
  author = {Nahawi Team},
  year = {2025},
  url = {https://github.com/nahawi/benchmark}
}
```

## License

The benchmark data is released under CC-BY-4.0 for research use.
