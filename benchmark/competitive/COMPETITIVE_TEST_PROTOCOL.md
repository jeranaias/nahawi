# Competitive Test Protocol V4: Nahawi vs Word vs Google Docs

## Objective
Rigorous head-to-head comparison of Arabic grammar correction capabilities across three systems on 100 professional-length MSA sentences with **ALL 13 FASIH ERROR TYPES**.

---

## Test Set Statistics (V4)

| Metric | Value |
|--------|-------|
| Total sentences | 100 |
| Average words/sentence | **19.8** |
| Total errors | **520** |
| Error types | **13** |
| Error range per sentence | 1-10 |

### Error Distribution (Matching FASIH v4.1)

| Error Type | Count | Percentage | Category |
|------------|-------|------------|----------|
| taa_marbuta (ة/ه) | 80 | 15.4% | Orthography |
| hamza (أ/إ/ء/آ) | 78 | 15.0% | Orthography |
| alif_maqsura (ى/ي) | 75 | 14.4% | Orthography |
| gender_agreement | 45 | 8.7% | Morphology |
| letter_confusion_د_ذ | 42 | 8.1% | Spelling |
| missing_preposition | 40 | 7.7% | Syntax |
| wrong_preposition | 35 | 6.7% | Syntax |
| verb_conjugation | 35 | 6.7% | Verb |
| number_agreement | 30 | 5.8% | Morphology |
| definiteness | 28 | 5.4% | Article |
| letter_confusion_ض_ظ | 18 | 3.5% | Spelling |
| letter_confusion_س_ص | 10 | 1.9% | Spelling |
| letter_confusion_ت_ط | 4 | 0.8% | Spelling |

### Error Categories Summary

| Category | Error Types | Count | % |
|----------|-------------|-------|---|
| **Orthography** | hamza, taa_marbuta, alif_maqsura | 233 | 44.8% |
| **Spelling** | د_ذ, ض_ظ, س_ص, ت_ط confusions | 74 | 14.2% |
| **Morphology** | gender_agreement, number_agreement | 75 | 14.4% |
| **Syntax** | missing_preposition, wrong_preposition | 75 | 14.4% |
| **Verb** | verb_conjugation | 35 | 6.7% |
| **Article** | definiteness | 28 | 5.4% |

---

## Test Systems

| System | Version | How to Test |
|--------|---------|-------------|
| **Nahawi** | V5 Hybrid | Run `run_competitive_benchmark.py` |
| **Microsoft Word** | Latest (Office 365) | Arabic proofing tools enabled |
| **Google Docs** | Web version | Language set to Arabic |

---

## Methodology

### Setup Requirements

**Microsoft Word:**
1. Open Word with new blank document
2. Go to File > Options > Language
3. Ensure "Arabic" is installed as proofing language
4. Enable: Review > Spelling & Grammar > Check Document

**Google Docs:**
1. Open new Google Doc
2. Go to Tools > Spelling and grammar > Show spelling suggestions
3. Ensure document language is Arabic (File > Language > Arabic)

### Testing Procedure

For each sentence:

1. **Copy** the source sentence from `test_sentences.txt`
2. **Paste** into Word (fresh paragraph)
3. **Wait** 2-3 seconds for proofing to run
4. **Record** in scoring_sheet.tsv:
   - Word_Flagged: Number of errors underlined
   - Word_Fixed: Number of correct suggestions offered
   - Word_FP: False positives (correct text flagged)
5. **Repeat** for Google Docs
6. **Compare** to Nahawi result and gold target

### Scoring Criteria

For each error in the sentence:

| Score | Meaning |
|-------|------------|
| **2** | Correctly identified AND correctly fixed |
| **1** | Correctly identified but wrong/no fix suggested |
| **0** | Not identified at all |
| **-1** | False positive (flagged correct text as error) |

---

## Files for Testing

| File | Purpose |
|------|---------|
| `test_sentences.txt` | Plain text, one sentence per line - COPY FROM HERE |
| `test_sentences_annotated.txt` | Sentences + expected corrections + error types |
| `scoring_sheet.tsv` | Tab-separated for Excel/Sheets import |
| `competitive_test_set_v4.json` | Full JSON with all metadata |
| `run_competitive_benchmark.py` | Automated Nahawi benchmark runner |

---

## Error Type Examples

### Orthography
- **hamza**: انشاء -> إنشاء, اكثر -> أكثر
- **taa_marbuta**: الحكومه -> الحكومة
- **alif_maqsura**: العربى -> العربي

### Spelling (Letter Confusion)
- **د/ذ**: هدا -> هذا, ادا -> إذا
- **ض/ظ**: نضر -> نظر, معضم -> معظم
- **س/ص**: (rarely tested)
- **ت/ط**: (rarely tested)

### Morphology
- **gender_agreement**: المدينه الكبير -> المدينة الكبيرة
- **number_agreement**: الموظفين يحتاجوا -> الموظفون يحتاجون

### Syntax
- **missing_preposition**: يبحث المعلومات -> يبحث عن المعلومات
- **wrong_preposition**: اشترك بالمؤتمر -> اشترك في المؤتمر

### Verb
- **verb_conjugation**: المهندسات صمموا -> المهندسات صممن

### Article
- **definiteness**: تقرير شامل -> تقريرا شاملا

---

## Expected Outcomes

Based on FASIH v4.1 benchmark performance:

**Nahawi V5 Hybrid:**
- 95.1% F0.5 overall
- 12/13 error categories PASS
- Strong on: orthography, spelling, morphology
- Expected correction rate: 90%+

**Microsoft Word Arabic Proofing:**
- Strong on: Obvious spelling errors, some hamza
- Weak on: Gender agreement, verb conjugation, prepositions
- Expected correction rate: 30-40%

**Google Docs Arabic:**
- Strong on: Basic spell check
- Weak on: Grammar beyond spelling
- Expected correction rate: 20-30%

---

## Calculation Method

After testing:

```
Correction Rate = (Errors Correctly Fixed) / (Total Errors)
Detection Rate = (Errors Flagged) / (Total Errors)
False Positive Rate = (False Positives) / (Clean Items)
```

---

## Output Deliverable

After testing, you'll have:

```
| System | Errors Caught | Correct Fixes | False Positives | Correction Rate |
|--------|---------------|---------------|-----------------|--------------------|
| Nahawi | X/520         | Y/520         | Z               | Y/520 = A%         |
| Word   | X/520         | Y/520         | Z               | B%                 |
| Google | X/520         | Y/520         | Z               | C%                 |
```

The killer slide becomes:
> "On 100 professional Arabic sentences with 520 grammar errors across 13 error types, Nahawi achieved X% correction rate while Word achieved Y% and Google achieved Z%."

---

## Quick Test Option

To do a quick sanity check before full testing:
- Test sentences 1-10 only (56 errors across all categories)
- Should take ~15-20 minutes per system
- If results are dramatically different, the full test is worth it

---

## Why V4 is Comprehensive

Previous versions (V2, V3) had limited error types:
- V2: Only 3 types (taa_marbuta, hamza, alif_maqsura)
- V3: 8 types (added morphology, verb, syntax basics)
- **V4: All 13 FASIH error types** - matches the benchmark we measure against

This makes V4 **expert-proof** - critics cannot claim cherry-picking.
