# FASIH-Identity: False Positive Testing

Correct Arabic sentences that models should leave **unchanged**.

## Purpose

GEC models can be overly aggressive, "fixing" text that was already correct. This causes:

1. **User frustration** - correct writing gets modified
2. **Loss of meaning** - valid stylistic choices overwritten
3. **Trust erosion** - users stop trusting the tool

FASIH-Identity tests whether your model knows when to **do nothing**.

## Statistics

| Category | Samples | Word Range |
|----------|---------|------------|
| short | 23 | 5-10 words |
| medium | 297 | 10-20 words |
| long | 180 | 20+ words |
| **Total** | **500** | |

## Source

Random samples from our 5.9GB MSA corpus (Wikipedia, UN, news, Shamela), filtered for:
- Complete sentences (not fragments)
- Proper punctuation
- No obvious errors
- Diverse topics and styles

## Format

```json
{
  "id": "identity-0042",
  "text": "تعتبر الطاقة المتجددة من أهم مصادر الطاقة في العصر الحديث",
  "word_count": 10,
  "category": "medium",
  "source_corpus": "msa"
}
```

## Evaluation

### Simple Check
```python
import json

with open('test.json') as f:
    identity = json.load(f)

false_positives = 0
for sample in identity:
    original = sample['text']
    corrected = your_model.correct(original)

    if corrected != original:
        false_positives += 1
        print(f"FP: {original[:50]}...")
        print(f"  → {corrected[:50]}...")

fp_rate = false_positives / len(identity)
print(f"False Positive Rate: {fp_rate:.2%}")
```

### Target Metrics

| Metric | Target |
|--------|--------|
| False Positive Rate | < 5% |
| Identity Preservation | > 95% |

A model that "corrects" more than 5% of correct text is too aggressive.

## Why This Matters

### The "Grammarly Problem"

Aggressive grammar tools often:
- Change `مما` to `ما` (both valid)
- "Fix" regional spelling variants
- Modify author's stylistic choices
- Insert unnecessary punctuation

### Real-World Impact

Users stop using tools that:
- Change their correct writing
- Don't understand context
- Apply rules blindly

## Tricky Cases (Future Work)

We plan to add a "tricky" subset with:

```
على علي أن يدرس
(The preposition على + the name علي - both correct)

في في الماء سمك
(في as verb "there is" + في as preposition - both correct)
```

These test whether models understand context vs. pattern-matching.

## Usage Tips

1. **Always test identity** alongside error correction
2. **Report FP rate** in your evaluations
3. **Tune confidence thresholds** to reduce FPs
4. **Consider "suggest only" mode** for borderline cases
