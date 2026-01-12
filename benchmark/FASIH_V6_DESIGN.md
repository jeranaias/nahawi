# FASIH Benchmark Redesign: Core + Full + Identity

## Vision

Create the **definitive Arabic GEC benchmark** that the world can use - no synthetic templates, no mislabeled categories, no ambiguity.

---

## Architecture

```
FASIH/
├── core/                    # Orthographic errors from real text
│   ├── test.json           # 1,000 samples
│   ├── dev.json            # 250 samples
│   └── README.md
│
├── full/                    # Core + Grammar/Syntax from real text
│   ├── test.json           # 2,000 samples
│   ├── dev.json            # 500 samples
│   └── README.md
│
├── identity/                # Correct sentences (false positive testing)
│   ├── test.json           # 500 samples
│   └── README.md
│
├── competitive/             # Word/Google comparison (keep existing)
│   └── ...
│
├── README.md               # Main documentation
└── rubric.json             # Category definitions
```

---

## FASIH-Core (Orthographic Errors)

**Source**: Real MSA corpus + existing v5.1 real samples

**Categories** (6 types, ~165 samples each = 1,000 total):

| Category | Description | Example |
|----------|-------------|---------|
| `hamza` | Hamza placement errors | اعلان → إعلان |
| `taa_marbuta` | ة vs ه confusion | الحكومه → الحكومة |
| `alif_maqsura` | ى vs ي confusion | علي → على |
| `alif_madda` | آ vs ا confusion | الان → الآن |
| `dal_thal` | د vs ذ confusion | هدا → هذا |
| `dad_za` | ض vs ظ confusion | نضر → نظر |

**Quality criteria**:
- Real sentences from Wikipedia/news/UN
- Single error type per sentence (clean signal)
- Natural, fluent Arabic (not templates)
- Diverse vocabulary and domains

---

## FASIH-Full (Complete Grammar)

**Includes**: All of Core + these additional categories

**Additional Categories** (8 types, ~125 samples each = 1,000 more):

| Category | Description | Example |
|----------|-------------|---------|
| `gender_agreement` | Noun-adjective gender | المدينة الكبير → المدينة الكبيرة |
| `number_agreement` | Noun-adjective number | الطلاب الناجح → الطلاب الناجحون |
| `verb_agreement` | Subject-verb agreement | الطالبات درسوا → الطالبات درسن |
| `missing_prep` | Missing preposition | يبحث المعلومات → يبحث عن المعلومات |
| `wrong_prep` | Incorrect preposition | اشترك بالمؤتمر → اشترك في المؤتمر |
| `definiteness` | Article agreement | كتاب الجديد → الكتاب الجديد |
| `spelling` | Common misspellings | الإقتصاد → الاقتصاد |
| `punctuation` | Punctuation errors | (from QALB patterns) |

**Source strategy**:
- Extract from QALB (36K real error pairs)
- Mine MSA corpus for patterns
- NO synthetic generation

---

## FASIH-Identity (False Positive Testing)

**Purpose**: Test that models don't "fix" correct text

**Source**: Random samples from MSA corpus that are grammatically correct

**Categories**:
| Type | Samples | Purpose |
|------|---------|---------|
| Short sentences (5-10 words) | 150 | Basic false positive |
| Medium sentences (10-20 words) | 200 | Typical length |
| Long sentences (20+ words) | 100 | Complex structures |
| Tricky correct (looks like error) | 50 | Edge cases |

**Examples of "tricky correct"**:
- "على علي أن يدرس" (على the preposition + علي the name)
- "في في الماء" (في as verb + في as preposition)

---

## Data Extraction Plan

### Step 1: Extract Core from MSA Corpus

```python
# Find sentences with common error patterns
# Then manually verify and create pairs

patterns = {
    'hamza': ['اعلان', 'اكثر', 'اخر', 'اول', 'امام'],
    'taa_marbuta': ['ه$' in feminine words],
    'alif_maqsura': ['علي', 'الي', 'حتي', 'متي'],
    ...
}
```

### Step 2: Extract Grammar from QALB

```python
# Parse QALB pairs and classify error types
# Filter for clean single-error examples

for pair in qalb_data:
    error_type = classify_error(pair['source'], pair['target'])
    if error_type in grammar_categories:
        add_to_full(pair, error_type)
```

### Step 3: Extract Identity from MSA Corpus

```python
# Random sample of correct sentences
# Verify they have no errors

identity_samples = random.sample(msa_corpus, 1000)
# Manual verification pass
```

---

## Category Definitions (rubric.json)

```json
{
  "version": "1.0",
  "categories": {
    "hamza": {
      "description": "Incorrect or missing hamza (أ إ آ ء ؤ ئ)",
      "examples": ["اعلان → إعلان", "مسائل → مسائل"],
      "common_confusions": ["ا↔أ", "ا↔إ", "ا↔آ"]
    },
    "taa_marbuta": {
      "description": "Confusion between taa marbuta (ة) and haa (ه)",
      "examples": ["المدرسه → المدرسة"],
      "note": "Only in feminine nouns/adjectives"
    },
    "gender_agreement": {
      "description": "Noun-adjective gender mismatch",
      "examples": ["الشركة الكبير → الشركة الكبيرة"],
      "note": "Adjective must match noun gender"
    },
    "wrong_prep": {
      "description": "Semantically incorrect preposition choice",
      "examples": ["يعتمد من → يعتمد على"],
      "note": "NOT alif_maqsura errors like علي→على"
    }
  }
}
```

---

## Naming Convention

**OLD (confusing)**:
- fasih_v4.1, fasih_v4.2, fasih_v5, fasih_v5.1

**NEW (clear)**:
- `fasih-core` - Orthographic only, highest quality
- `fasih-full` - Complete grammar coverage
- `fasih-identity` - False positive testing

No version numbers in names. Version tracked in rubric.json.

---

## File Format

```json
{
  "id": "core-hamza-0001",
  "source": "اعلنت الحكومة عن خطة جديدة",
  "target": "أعلنت الحكومة عن خطة جديدة",
  "category": "hamza",
  "correction": "اعلنت → أعلنت",
  "source_corpus": "wikipedia",
  "difficulty": "easy"
}
```

**Fields**:
- `id`: Unique identifier
- `source`: Text with error
- `target`: Corrected text
- `category`: Error type
- `correction`: Specific fix (for analysis)
- `source_corpus`: Origin (wikipedia/un/news/qalb)
- `difficulty`: easy/medium/hard

---

## Quality Assurance

1. **No synthetic templates** - All from real text
2. **Single error per sentence** - Clean signal for per-category eval
3. **Diverse domains** - Politics, science, culture, sports, etc.
4. **Native speaker review** - At least 10% spot-checked
5. **Deduplication** - No repeated sentences
6. **Length variety** - Short, medium, long sentences

---

## Migration Plan

1. **Keep existing benchmarks** in `benchmark/legacy/`
2. **Build new structure** in `benchmark/fasih/`
3. **Update evaluation scripts** to use new format
4. **Update README** with new documentation

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Core samples | 1,000+ |
| Full samples | 2,000+ |
| Identity samples | 500+ |
| Categories | 14 (6 ortho + 8 grammar) |
| Real text ratio | 100% (no synthetic) |
| Source diversity | 4+ corpora |

---

## Timeline

1. **Phase 1**: Extract Core orthographic samples from existing v5.1 real data
2. **Phase 2**: Extract grammar samples from QALB with proper labeling
3. **Phase 3**: Build Identity set from MSA corpus
4. **Phase 4**: Consolidate, deduplicate, quality check
5. **Phase 5**: Update documentation and evaluation scripts
