# FASIH-Core: Orthographic Error Benchmark

High-quality orthographic error samples extracted from real Arabic Wikipedia text.

## Statistics

| Category | Test | Dev | Total |
|----------|------|-----|-------|
| alif_madda | 103 | ~20 | ~123 |
| dad_za | 108 | ~20 | ~128 |
| dal_thal | 95 | ~18 | ~113 |
| alif_maqsura | 80 | ~15 | ~95 |
| hamza | 61 | ~12 | ~73 |
| taa_marbuta | 60 | ~12 | ~72 |
| **Total** | **507** | **91** | **598** |

## Why "Core"?

These 6 orthographic categories:
1. Are objectively verifiable (rule-based)
2. Come from real Wikipedia edit histories
3. Have no ambiguity in what's correct
4. Are the foundation of Arabic writing accuracy

## Sample

```json
{
  "id": "core-hamza-0015",
  "source": "في اكتوبر 2021 خسرت مباراة الميدالية البرونزية في بطولة العالم",
  "target": "في أكتوبر 2021 خسرت مباراة الميدالية البرونزية في بطولة العالم",
  "category": "hamza",
  "correction": "اكتوبر → أكتوبر",
  "source_corpus": "wikipedia",
  "difficulty": "easy"
}
```

## Error Types

### hamza (همزة)
Missing or incorrect hamza on alif.
- اعلان → إعلان (beginning of word)
- مسألة → مسئلة (middle of word)
- قراءة → قرائة (on seat)

### taa_marbuta (تاء مربوطة)
Confusion between ة and ه at word end.
- المدرسه → المدرسة
- الحكومه → الحكومة

### alif_maqsura (ألف مقصورة)
Confusion between ى and ي at word end.
- علي → على (preposition)
- إلي → إلى (preposition)
- حتي → حتى

### alif_madda (ألف المد)
Missing madda on alif.
- الان → الآن
- اخر → آخر
- القران → القرآن

### dad_za (ض/ظ)
Confusion between emphatic consonants.
- نضر → نظر
- ضهر → ظهر
- حضور → حظور

### dal_thal (د/ذ)
Confusion between dental consonants.
- هدا → هذا
- ادا → إذا
- كدلك → كذلك

## Usage

```python
import json

with open('test.json') as f:
    core_test = json.load(f)

print(f"Core test samples: {len(core_test)}")

# Evaluate by category
from collections import defaultdict
by_cat = defaultdict(list)
for s in core_test:
    by_cat[s['category']].append(s)

for cat, samples in by_cat.items():
    print(f"{cat}: {len(samples)}")
```

## Expected Performance

A production-ready Arabic GEC model should achieve:
- **Overall**: ≥95% accuracy
- **Per-category**: ≥90% accuracy

These are straightforward orthographic rules that native speakers learn in elementary school.
