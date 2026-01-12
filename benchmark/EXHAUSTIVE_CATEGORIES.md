# Exhaustive Arabic GEC Error Categories

## Current Coverage (8 categories)

| Category | Arabic | Status | Samples |
|----------|--------|--------|---------|
| hamza | همزة | Done | 124 |
| taa_marbuta | تاء مربوطة | Done | 123 |
| alif_maqsura | ألف مقصورة | Done | 125 |
| alif_madda | ألف المد | Done | 130 |
| dad_za | ض/ظ | Done | 132 |
| dal_thal | د/ذ | Done | 131 |
| missing_prep | حرف جر مفقود | Done | 88 |
| wrong_prep | حرف جر خاطئ | Done | 25 |

## Missing Categories

### Tier 1: Critical (Must Have)

| Category | Arabic | Description | Example | Difficulty |
|----------|--------|-------------|---------|------------|
| hamza_wasl | همزة الوصل | Wasl vs Qat' hamza | إستخدام → استخدام | Medium |
| gender_agreement | مطابقة الجنس | Noun-adj gender | المدينة الكبير → المدينة الكبيرة | Hard |
| number_agreement | مطابقة العدد | Singular/plural | الطلاب الناجح → الطلاب الناجحون | Hard |
| definiteness | التعريف | Article agreement | كتاب الجديد → الكتاب الجديد | Medium |
| verb_form | صيغة الفعل | Wrong verb form | يكتبون الطالبات → تكتب الطالبات | Hard |

### Tier 2: Important (Should Have)

| Category | Arabic | Description | Example | Difficulty |
|----------|--------|-------------|---------|------------|
| ta_marbouta_alif | تاء + ألف | ة vs اة ending | الحياه → الحياة | Easy |
| tanwin | تنوين | Tanwin spelling | مثلاً → مثلا | Easy |
| shadda_alif | شدة + ألف | لّا vs لا | إلا → إلّا | Medium |
| nun_tanwin | نون/تنوين | Final ن vs tanwin | مسلمين → مسلمون | Medium |
| idafa | الإضافة | Construct state | الجامعة الطلاب → طلاب الجامعة | Hard |

### Tier 3: Common Confusions

| Category | Arabic | Description | Example | Difficulty |
|----------|--------|-------------|---------|------------|
| sin_sad | س/ص | س vs ص confusion | مسطرة → مسترة (rare) | Easy |
| ta_tta | ت/ط | ت vs ط confusion | مطار → متار (rare) | Easy |
| tha_za | ث/ذ | ث vs ذ confusion | ثلاثة → ذلاذة (rare) | Easy |
| ya_alif | ي/ا | Final ي vs ا | فتا → فتى | Easy |
| waw_alif | و/ا | Long vowel confusion | داود → داوود | Easy |

### Tier 4: Spelling & Typos

| Category | Arabic | Description | Example | Difficulty |
|----------|--------|-------------|---------|------------|
| space_join | فصل الكلمات | Word splitting | إن شاء الله → إنشاء الله | Medium |
| space_split | وصل الكلمات | Word joining | عبدالله → عبد الله | Medium |
| repeated_char | تكرار حرف | Doubled letter | الكتااب → الكتاب | Easy |
| missing_char | حرف مفقود | Missing letter | الكتب → الكتاب | Easy |
| swapped_char | حرف مقلوب | Transposed letters | الكتبا → الكتاب | Easy |
| keyboard_adj | خطأ لوحة | Adjacent key error | المدرسث → المدرسة | Easy |

### Tier 5: Syntax & Structure

| Category | Arabic | Description | Example | Difficulty |
|----------|--------|-------------|---------|------------|
| word_order | ترتيب الكلمات | Wrong word order | الكبير البيت → البيت الكبير | Hard |
| extra_word | كلمة زائدة | Unnecessary word | ذهب هو إلى → ذهب إلى | Medium |
| missing_word | كلمة مفقودة | Missing word | ذهب المدرسة → ذهب إلى المدرسة | Medium |
| wrong_conj | حرف عطف | Wrong conjunction | ذهب إلى ولكن → ذهب إلى و | Medium |

### Tier 6: Semantic & Style

| Category | Arabic | Description | Example | Difficulty |
|----------|--------|-------------|---------|------------|
| punctuation | علامات الترقيم | Punctuation errors | Missing/wrong punct | Easy |
| diacritic | تشكيل | Wrong diacritics | كَتَبَ vs كُتِبَ | Hard |

## Priority Order for Implementation

### Phase 1: Low-Hanging Fruit (Can hunt from corpus)
1. **hamza_wasl** - Very common, searchable patterns (إست، إنت، إبت)
2. **tanwin** - Searchable (words ending in اً)
3. **space_join/split** - Common compounds (عبدالله، إن شاء الله)
4. **repeated_char** - Can generate from clean text
5. **punctuation** - Abundant in corpus

### Phase 2: Pattern-Based (Can generate from templates)
1. **gender_agreement** - Find noun+adj pairs, flip gender
2. **number_agreement** - Find noun+adj pairs, flip number
3. **definiteness** - Find الX الY patterns, remove one ال
4. **verb_form** - Find subject+verb, mismatch agreement

### Phase 3: Expert Curation (Manual examples needed)
1. **word_order** - Requires linguistic expertise
2. **idafa** - Complex construct state rules
3. **missing_word** - Context-dependent

## Recommended Final Structure

```
FASIH v3.0
├── Orthographic (12 categories, ~1500 samples)
│   ├── hamza (150)
│   ├── hamza_wasl (150)      # NEW
│   ├── taa_marbuta (150)
│   ├── alif_maqsura (150)
│   ├── alif_madda (150)
│   ├── dad_za (150)
│   ├── dal_thal (150)
│   ├── tanwin (100)          # NEW
│   ├── space_join (100)      # NEW
│   ├── space_split (100)     # NEW
│   ├── repeated_char (50)    # NEW
│   └── missing_char (50)     # NEW
│
├── Agreement (4 categories, ~400 samples)
│   ├── gender_agreement (100)
│   ├── number_agreement (100)
│   ├── definiteness (100)
│   └── verb_form (100)
│
├── Prepositions (2 categories, ~200 samples)
│   ├── missing_prep (100)
│   ├── wrong_prep (100)
│
├── Syntax (2 categories, ~100 samples)
│   ├── word_order (50)
│   └── extra_word (50)
│
└── Identity (500 samples)

TOTAL: 20 categories, ~2700 samples
```

## Huntable Patterns in MSA Corpus

### hamza_wasl (همزة الوصل)
```
# Wrong (with hamza): إستخدام، إنتقال، إبتداء، إستمرار
# Correct (wasl): استخدام، انتقال، ابتداء، استمرار

PATTERNS = {
    'استخدام': ['إستخدام'],
    'استمرار': ['إستمرار'],
    'انتخابات': ['إنتخابات'],
    'انتقال': ['إنتقال'],
    'ابتداء': ['إبتداء'],
    'اجتماع': ['إجتماع'],
    'اقتصاد': ['إقتصاد'],
    'افتتاح': ['إفتتاح'],
}
```

### space_join (Should be separate)
```
# Wrong (joined): إنشاءالله، عبدالرحمن، مابين
# Correct (split): إن شاء الله، عبد الرحمن، ما بين

PATTERNS = {
    'إن شاء الله': ['إنشاءالله', 'انشاءالله', 'ان شاء الله'],
    'عبد الله': ['عبدالله'],
    'عبد الرحمن': ['عبدالرحمن'],
    'ما بين': ['مابين'],
    'في ما': ['فيما'],  # context-dependent
}
```

### space_split (Should be joined)
```
# Wrong (split): لا كن، إ ذا، هـ ذا
# Correct (joined): لكن، إذا، هذا

# Less common - usually typos
```

### gender_agreement
```
# Find patterns: NOUN_FEM + ADJ_MASC or NOUN_MASC + ADJ_FEM
# Use morphological analyzer to detect

# Example sentences to corrupt:
# المدينة الكبيرة → المدينة الكبير
# البيت الجميل → البيت الجميلة
```
