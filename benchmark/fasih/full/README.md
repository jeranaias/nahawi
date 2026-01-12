# FASIH-Full: Complete Grammar Benchmark

Comprehensive Arabic GEC benchmark covering orthography, morphology, and syntax.

## Statistics

| Category | Test | Dev | Source |
|----------|------|-----|--------|
| **Orthographic (Core)** | | | |
| alif_madda | 103 | ~20 | Wikipedia |
| dad_za | 108 | ~20 | Wikipedia |
| dal_thal | 95 | ~18 | Wikipedia |
| alif_maqsura | 80 | ~15 | Wikipedia |
| hamza | 61 | ~12 | Wikipedia |
| taa_marbuta | 60 | ~12 | Wikipedia |
| **Grammar** | | | |
| gender_agreement | 120 | 30 | QALB |
| number_agreement | 120 | 30 | QALB |
| verb_agreement | 120 | 30 | QALB |
| wrong_prep | 120 | 30 | QALB |
| definiteness | 120 | 30 | QALB |
| spelling | 99 | 25 | QALB |
| punctuation | 120 | 30 | QALB |
| **Total** | **~1,327** | **~297** | |

## What's Different from Core?

FASIH-Full adds **grammar and syntax** categories that require deeper linguistic understanding:

### Morphological Agreement

**gender_agreement**: Adjectives must match noun gender
```
المدينة الكبير → المدينة الكبيرة
(the city the-big.MASC → the city the-big.FEM)
```

**number_agreement**: Adjectives must match noun number
```
الطلاب الناجح → الطلاب الناجحون
(the students the-successful.SING → the students the-successful.PL)
```

**verb_agreement**: Verbs must agree with subject
```
الطالبات درسوا → الطالبات درسن
(the female students studied.MASC.PL → studied.FEM.PL)
```

### Syntactic Errors

**wrong_prep**: Incorrect preposition choice
```
يعتمد من المصادر → يعتمد على المصادر
(depends from sources → depends on sources)
```

**definiteness**: Article agreement in idafa/adjective constructions
```
كتاب الجديد → الكتاب الجديد
(book the-new → the-book the-new)
```

### Other

**spelling**: Common spelling mistakes beyond orthography
**punctuation**: Punctuation placement errors

## Source: QALB

Grammar samples are extracted from the Qatar Arabic Language Bank (QALB) 2014 dataset, which contains real user errors from native Arabic speakers.

**Extraction process**:
1. Parse 36,771 QALB source-target pairs
2. Classify error type by linguistic rules
3. Filter for clean single-error examples
4. Balance categories (max 150 per type)

## Sample

```json
{
  "id": "full-gender_agreement-0042",
  "source": "المنظمة الدولي للهجرة أعلنت عن برنامج جديد",
  "target": "المنظمة الدولية للهجرة أعلنت عن برنامج جديد",
  "category": "gender_agreement",
  "correction": "الدولي → الدولية",
  "source_corpus": "qalb",
  "difficulty": "medium"
}
```

## Why Grammar is Harder

| Aspect | Orthographic | Grammar |
|--------|--------------|---------|
| Scope | Single character | Word/phrase |
| Context | Local | Sentence-wide |
| Rules | Deterministic | Context-dependent |
| Detection | Pattern match | Parsing required |

Grammar errors require understanding sentence structure, not just character patterns.

## Expected Performance

| Category Type | Target F0.5 |
|---------------|-------------|
| Orthographic | ≥95% |
| Morphological | ≥85% |
| Syntactic | ≥80% |
| Overall | ≥90% |

## Usage

```python
import json
from collections import Counter

with open('test.json') as f:
    full_test = json.load(f)

# Category distribution
cats = Counter(s['category'] for s in full_test)
for cat, count in cats.most_common():
    print(f"{cat}: {count}")

# Filter by difficulty
hard = [s for s in full_test if s.get('difficulty') == 'hard']
print(f"Hard samples: {len(hard)}")
```
