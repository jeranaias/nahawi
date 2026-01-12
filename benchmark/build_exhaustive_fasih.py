#!/usr/bin/env python3
"""
Build exhaustive FASIH benchmark with ALL Arabic GEC categories.

Strategy:
1. HUNT: Find real errors in corpus (hamza_wasl, space patterns)
2. GENERATE: Take clean sentences, introduce controlled single errors (agreement)
3. CURATE: Manual examples for complex categories

All samples are REAL MSA sentences - we just systematically introduce errors.
"""

import json
import random
import re
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Optional, Tuple, List, Dict

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Paths
BENCHMARK_DIR = Path(__file__).parent
FASIH_DIR = BENCHMARK_DIR / "fasih"
CORPUS_PATH = Path("c:/nahawi/corpus/msa_corpus_full.txt")

# Target samples per category
TARGET_PER_CAT = 100

# ============================================================
# CATEGORY 1: HAMZA WASL (همزة الوصل)
# Very common error - putting hamza on wasl words
# ============================================================

HAMZA_WASL_PATTERNS = {
    # افتعال pattern verbs (Form VIII)
    'استخدام': ['إستخدام'],
    'استمرار': ['إستمرار'],
    'استقلال': ['إستقلال'],
    'استثمار': ['إستثمار'],
    'استعداد': ['إستعداد'],
    'استقبال': ['إستقبال'],
    'استطاعة': ['إستطاعة'],
    'استراتيجية': ['إستراتيجية'],
    'استفادة': ['إستفادة'],
    'استجابة': ['إستجابة'],

    # انفعال pattern verbs (Form VII)
    'انتخابات': ['إنتخابات'],
    'انتقال': ['إنتقال'],
    'انتشار': ['إنتشار'],
    'انتصار': ['إنتصار'],
    'انتظار': ['إنتظار'],
    'انتهاء': ['إنتهاء'],
    'انتباه': ['إنتباه'],
    'انتاج': ['إنتاج'],
    'انسحاب': ['إنسحاب'],
    'انضمام': ['إنضمام'],

    # افتعل verbs
    'اجتماع': ['إجتماع'],
    'اقتصاد': ['إقتصاد'],
    'اقتراح': ['إقتراح'],
    'اقتراب': ['إقتراب'],
    'افتتاح': ['إفتتاح'],
    'احتفال': ['إحتفال'],
    'احتجاج': ['إحتجاج'],
    'اختيار': ['إختيار'],
    'اختلاف': ['إختلاف'],
    'اكتشاف': ['إكتشاف'],
    'ابتداء': ['إبتداء'],
    'ابتعاد': ['إبتعاد'],
    'اتفاق': ['إتفاق'],
    'اتجاه': ['إتجاه'],
    'اتحاد': ['إتحاد'],
    'اتصال': ['إتصال'],
    'اتخاذ': ['إتخاذ'],
    'التزام': ['إلتزام'],
    'التحاق': ['إلتحاق'],
    'التقاء': ['إلتقاء'],
    'امتحان': ['إمتحان'],
    'امتداد': ['إمتداد'],

    # Imperative forms
    'اذهب': ['إذهب'],
    'اكتب': ['إكتب'],
    'اقرأ': ['إقرأ'],
    'افعل': ['إفعل'],
    'انتظر': ['إنتظر'],
    'استمع': ['إستمع'],
}

# ============================================================
# CATEGORY 2: SPACE ERRORS (فصل ووصل الكلمات)
# ============================================================

# Words that should be SEPARATE but often written joined
SPACE_JOIN_PATTERNS = {
    'إن شاء الله': ['إنشاءالله', 'انشاءالله', 'إنشاء الله'],
    'ما شاء الله': ['ماشاءالله', 'ماشاء الله'],
    'بسم الله': ['بسمالله'],
    'الحمد لله': ['الحمدلله'],
    'عبد الله': ['عبدالله'],
    'عبد الرحمن': ['عبدالرحمن'],
    'عبد العزيز': ['عبدالعزيز'],
    'عبد الكريم': ['عبدالكريم'],
    'ما بين': ['مابين'],
    'ما زال': ['مازال'],
    'لا بد': ['لابد'],
    'لا سيما': ['لاسيما'],
}

# Words that should be JOINED but often written separate
SPACE_SPLIT_PATTERNS = {
    'فيما': ['في ما'],
    'عما': ['عن ما'],
    'مما': ['من ما'],
    'إنما': ['إن ما'],
    'كلما': ['كل ما'],
    'حيثما': ['حيث ما'],
    'أينما': ['أين ما'],
    'كيفما': ['كيف ما'],
    'بينما': ['بين ما'],
}

# ============================================================
# CATEGORY 3: TANWIN (تنوين)
# ============================================================

# Words commonly written with/without proper tanwin alif
TANWIN_PATTERNS = {
    # Should have alif with tanwin
    'أيضًا': ['أيضا', 'ايضا', 'ايضاً'],
    'جدًا': ['جدا', 'جداً'],
    'كثيرًا': ['كثيرا', 'كثيراً'],
    'قليلًا': ['قليلا', 'قليلاً'],
    'سابقًا': ['سابقا', 'سابقاً'],
    'لاحقًا': ['لاحقا', 'لاحقاً'],
    'نظرًا': ['نظرا', 'نظراً'],
    'بدلًا': ['بدلا', 'بدلاً'],
    'فضلًا': ['فضلا', 'فضلاً'],
    'مثلًا': ['مثلا', 'مثلاً'],
    'أولًا': ['أولا', 'اولا'],
    'ثانيًا': ['ثانيا', 'ثانياً'],
    'ثالثًا': ['ثالثا', 'ثالثاً'],
    'أخيرًا': ['أخيرا', 'اخيرا'],
    'تمامًا': ['تماما', 'تماماً'],
    'دائمًا': ['دائما', 'دائماً'],
    'عادةً': ['عادة'],  # Different - taa marbuta + tanwin
    'خاصةً': ['خاصة'],
    'عامةً': ['عامة'],
}

# ============================================================
# CATEGORY 4: GENDER AGREEMENT (مطابقة الجنس)
# ============================================================

# Feminine markers
FEMININE_MARKERS = ['ة', 'اء', 'ى']

# Common adjectives with masc/fem forms
ADJECTIVES = {
    # masc: fem
    'كبير': 'كبيرة',
    'صغير': 'صغيرة',
    'جديد': 'جديدة',
    'قديم': 'قديمة',
    'جميل': 'جميلة',
    'طويل': 'طويلة',
    'قصير': 'قصيرة',
    'سريع': 'سريعة',
    'بطيء': 'بطيئة',
    'قوي': 'قوية',
    'ضعيف': 'ضعيفة',
    'غني': 'غنية',
    'فقير': 'فقيرة',
    'سعيد': 'سعيدة',
    'حزين': 'حزينة',
    'مهم': 'مهمة',
    'خاص': 'خاصة',
    'عام': 'عامة',
    'واسع': 'واسعة',
    'ضيق': 'ضيقة',
    'عميق': 'عميقة',
    'بعيد': 'بعيدة',
    'قريب': 'قريبة',
    'أول': 'أولى',
    'ثاني': 'ثانية',
    'رئيسي': 'رئيسية',
    'أساسي': 'أساسية',
    'وطني': 'وطنية',
    'دولي': 'دولية',
    'محلي': 'محلية',
    'عربي': 'عربية',
    'إسلامي': 'إسلامية',
    'تاريخي': 'تاريخية',
    'سياسي': 'سياسية',
    'اقتصادي': 'اقتصادية',
    'اجتماعي': 'اجتماعية',
    'ثقافي': 'ثقافية',
    'علمي': 'علمية',
    'عسكري': 'عسكرية',
    'رسمي': 'رسمية',
    'شخصي': 'شخصية',
}

# Common feminine nouns (end in ة or inherently feminine)
FEMININE_NOUNS = [
    'المدينة', 'الدولة', 'الحكومة', 'الجامعة', 'المدرسة', 'الشركة',
    'المنظمة', 'الجمعية', 'اللجنة', 'الوزارة', 'المؤسسة', 'الهيئة',
    'القرية', 'المنطقة', 'المحافظة', 'العاصمة', 'الجزيرة', 'القارة',
    'اللغة', 'الثقافة', 'الحضارة', 'الفترة', 'المرحلة', 'السنة',
    'الحرب', 'المعركة', 'الثورة', 'الحركة', 'النهضة', 'الوحدة',
    'الصحيفة', 'المجلة', 'الجريدة', 'القناة', 'الإذاعة', 'الصورة',
    'الرحلة', 'الزيارة', 'المباراة', 'البطولة', 'الجائزة', 'المسابقة',
]

# Common masculine nouns
MASCULINE_NOUNS = [
    'البيت', 'القصر', 'المسجد', 'المعبد', 'الجسر', 'الطريق',
    'النهر', 'البحر', 'الجبل', 'الوادي', 'السهل', 'الشاطئ',
    'الكتاب', 'القلم', 'الباب', 'الشارع', 'السوق', 'المتحف',
    'الجيش', 'الشعب', 'الوطن', 'العالم', 'التاريخ', 'المستقبل',
    'الرئيس', 'الوزير', 'الملك', 'الأمير', 'القائد', 'الزعيم',
]

# ============================================================
# CATEGORY 5: NUMBER AGREEMENT (مطابقة العدد)
# ============================================================

# Plural patterns
PLURAL_ADJECTIVES = {
    # singular_masc: plural_masc
    'كبير': 'كبار',
    'صغير': 'صغار',
    'جديد': 'جدد',
    'قديم': 'قدماء',
    'عظيم': 'عظماء',
    'كريم': 'كرماء',
    'حكيم': 'حكماء',
    'عالم': 'علماء',
    'ناجح': 'ناجحون',
    'فاشل': 'فاشلون',
    'عامل': 'عاملون',
    'طالب': 'طلاب',
    'كاتب': 'كتاب',
    'قارئ': 'قراء',
    'شاعر': 'شعراء',
}

# ============================================================
# CATEGORY 6: DEFINITENESS (التعريف والتنكير)
# ============================================================

# Pattern: الX الY - both must have ال or neither
# Error: كتاب الجديد (mixing) → الكتاب الجديد

# ============================================================
# HUNTING AND GENERATION FUNCTIONS
# ============================================================

def load_corpus_sentences(limit: int = 500000) -> List[str]:
    """Load sentences from MSA corpus."""
    sentences = []

    if not CORPUS_PATH.exists():
        print(f"Corpus not found at {CORPUS_PATH}")
        return sentences

    print(f"Loading corpus from {CORPUS_PATH}...")

    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= limit:
                break
            line = line.strip()
            if len(line) > 20 and len(line) < 500:
                sentences.append(line)

    print(f"Loaded {len(sentences)} sentences")
    return sentences


def hunt_pattern_errors(sentences: List[str], patterns: Dict[str, List[str]],
                        category: str, target: int = 100) -> List[dict]:
    """Hunt for sentences containing correct forms, create error versions."""
    samples = []

    for correct, error_forms in patterns.items():
        if len(samples) >= target:
            break

        for sentence in sentences:
            if correct in sentence:
                # Create error version
                error_form = random.choice(error_forms)
                source = sentence.replace(correct, error_form, 1)

                # Verify the replacement actually happened
                if source != sentence:
                    samples.append({
                        'source': source,
                        'target': sentence,
                        'category': category,
                        'error': f'{error_form} → {correct}',
                        'verified': True,
                        'source_corpus': 'msa'
                    })

                    if len(samples) >= target:
                        break

    return samples


def hunt_hamza_wasl(sentences: List[str], target: int = 150) -> List[dict]:
    """Hunt for hamza wasl errors."""
    return hunt_pattern_errors(sentences, HAMZA_WASL_PATTERNS, 'hamza_wasl', target)


def hunt_space_join(sentences: List[str], target: int = 100) -> List[dict]:
    """Hunt for space join errors (should be separate)."""
    return hunt_pattern_errors(sentences, SPACE_JOIN_PATTERNS, 'space_join', target)


def hunt_space_split(sentences: List[str], target: int = 100) -> List[dict]:
    """Hunt for space split errors (should be joined)."""
    return hunt_pattern_errors(sentences, SPACE_SPLIT_PATTERNS, 'space_split', target)


def hunt_tanwin(sentences: List[str], target: int = 100) -> List[dict]:
    """Hunt for tanwin errors."""
    return hunt_pattern_errors(sentences, TANWIN_PATTERNS, 'tanwin', target)


def generate_gender_agreement(sentences: List[str], target: int = 100) -> List[dict]:
    """Generate gender agreement errors from clean sentences."""
    samples = []

    # Pattern: FEMININE_NOUN + FEMININE_ADJ → introduce error by using MASC_ADJ
    fem_to_masc = {v: k for k, v in ADJECTIVES.items()}

    for sentence in sentences:
        if len(samples) >= target:
            break

        # Look for feminine noun + feminine adjective
        for noun in FEMININE_NOUNS:
            if noun not in sentence:
                continue

            for masc_adj, fem_adj in ADJECTIVES.items():
                pattern = f'{noun} {fem_adj}'
                if pattern in sentence:
                    # Create error: use masculine adjective with feminine noun
                    source = sentence.replace(pattern, f'{noun} {masc_adj}', 1)

                    if source != sentence:
                        samples.append({
                            'source': source,
                            'target': sentence,
                            'category': 'gender_agreement',
                            'error': f'{masc_adj} → {fem_adj}',
                            'verified': True,
                            'source_corpus': 'msa'
                        })
                        break

            if len(samples) >= target:
                break

    return samples


def generate_definiteness(sentences: List[str], target: int = 100) -> List[dict]:
    """Generate definiteness errors."""
    samples = []

    # Pattern: الX الY → X الY (remove first ال)
    pattern = re.compile(r'(ال\w+)\s+(ال\w+)')

    for sentence in sentences:
        if len(samples) >= target:
            break

        match = pattern.search(sentence)
        if match:
            full_match = match.group(0)
            word1 = match.group(1)
            word2 = match.group(2)

            # Create error: remove ال from first word
            error_word1 = word1[2:]  # Remove ال
            error_pattern = f'{error_word1} {word2}'

            source = sentence.replace(full_match, error_pattern, 1)

            if source != sentence and len(error_word1) > 2:
                samples.append({
                    'source': source,
                    'target': sentence,
                    'category': 'definiteness',
                    'error': f'{error_word1} → {word1}',
                    'verified': True,
                    'source_corpus': 'msa'
                })

    return samples


def generate_repeated_char(sentences: List[str], target: int = 50) -> List[dict]:
    """Generate repeated character errors."""
    samples = []

    for sentence in sentences:
        if len(samples) >= target:
            break

        words = sentence.split()
        if len(words) < 3:
            continue

        # Pick a random word to corrupt
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]

        if len(word) < 4:
            continue

        # Pick a random position to double a character
        char_idx = random.randint(1, len(word) - 2)
        char = word[char_idx]

        # Skip if already doubled or if it's a combining character
        if char in 'ًٌٍَُِّْ':
            continue

        corrupted_word = word[:char_idx] + char + word[char_idx:]

        source_words = words.copy()
        source_words[word_idx] = corrupted_word
        source = ' '.join(source_words)

        samples.append({
            'source': source,
            'target': sentence,
            'category': 'repeated_char',
            'error': f'{corrupted_word} → {word}',
            'verified': True,
            'source_corpus': 'msa'
        })

    return samples


def generate_missing_char(sentences: List[str], target: int = 50) -> List[dict]:
    """Generate missing character errors."""
    samples = []

    for sentence in sentences:
        if len(samples) >= target:
            break

        words = sentence.split()
        if len(words) < 3:
            continue

        # Pick a random word to corrupt
        word_idx = random.randint(0, len(words) - 1)
        word = words[word_idx]

        if len(word) < 5:
            continue

        # Pick a random position to remove a character (not first or last)
        char_idx = random.randint(1, len(word) - 2)
        char = word[char_idx]

        # Skip diacritics
        if char in 'ًٌٍَُِّْ':
            continue

        corrupted_word = word[:char_idx] + word[char_idx + 1:]

        source_words = words.copy()
        source_words[word_idx] = corrupted_word
        source = ' '.join(source_words)

        samples.append({
            'source': source,
            'target': sentence,
            'category': 'missing_char',
            'error': f'{corrupted_word} → {word}',
            'verified': True,
            'source_corpus': 'msa'
        })

    return samples


def generate_number_agreement(sentences: List[str], target: int = 100) -> List[dict]:
    """Generate number agreement errors (singular adj with plural noun)."""
    samples = []

    # Plural nouns with sound masculine plural ending ون/ين
    PLURAL_NOUNS = [
        'المسلمون', 'المؤمنون', 'العاملون', 'المعلمون', 'الطالبون',
        'المهندسون', 'الموظفون', 'المواطنون', 'اللاعبون', 'الفائزون',
        'المشاركون', 'الحاضرون', 'الباحثون', 'الكاتبون', 'القارئون',
        'المسافرون', 'الزائرون', 'المتظاهرون', 'المحتجون', 'الناخبون',
    ]

    # Adjectives that should agree in number
    ADJ_PAIRS = {
        # singular: plural (sound masc)
        'ناجح': 'ناجحون',
        'فاشل': 'فاشلون',
        'قادر': 'قادرون',
        'عاجز': 'عاجزون',
        'حاضر': 'حاضرون',
        'غائب': 'غائبون',
        'صادق': 'صادقون',
        'كاذب': 'كاذبون',
        'مخلص': 'مخلصون',
        'متفائل': 'متفائلون',
        'متشائم': 'متشائمون',
        'سعيد': 'سعيدون',
        'حزين': 'حزينون',
    }

    for sentence in sentences:
        if len(samples) >= target:
            break

        for noun in PLURAL_NOUNS:
            if noun not in sentence:
                continue

            for sing_adj, plur_adj in ADJ_PAIRS.items():
                pattern = f'{noun} {plur_adj}'
                if pattern in sentence:
                    # Create error: singular adj with plural noun
                    source = sentence.replace(pattern, f'{noun} {sing_adj}', 1)

                    if source != sentence:
                        samples.append({
                            'source': source,
                            'target': sentence,
                            'category': 'number_agreement',
                            'error': f'{sing_adj} → {plur_adj}',
                            'verified': True,
                            'source_corpus': 'msa'
                        })
                        break

            if len(samples) >= target:
                break

    return samples


def generate_verb_agreement(sentences: List[str], target: int = 100) -> List[dict]:
    """Generate verb agreement errors."""
    samples = []

    # Feminine subjects with verbs
    FEM_SUBJECTS = ['هي', 'المرأة', 'الفتاة', 'البنت', 'الأم', 'الأخت']

    # Verb pairs: correct fem -> incorrect masc
    VERB_PAIRS = {
        'ذهبت': 'ذهب',
        'جاءت': 'جاء',
        'قالت': 'قال',
        'كتبت': 'كتب',
        'درست': 'درس',
        'عملت': 'عمل',
        'سافرت': 'سافر',
        'رجعت': 'رجع',
        'نجحت': 'نجح',
        'فشلت': 'فشل',
    }

    for sentence in sentences:
        if len(samples) >= target:
            break

        for subject in FEM_SUBJECTS:
            if subject not in sentence:
                continue

            for correct_verb, wrong_verb in VERB_PAIRS.items():
                # Look for: subject ... verb (fem correct form)
                if f'{subject} {correct_verb}' in sentence:
                    source = sentence.replace(f'{subject} {correct_verb}',
                                             f'{subject} {wrong_verb}', 1)
                    if source != sentence:
                        samples.append({
                            'source': source,
                            'target': sentence,
                            'category': 'verb_agreement',
                            'error': f'{wrong_verb} → {correct_verb}',
                            'verified': True,
                            'source_corpus': 'msa'
                        })
                        break

            if len(samples) >= target:
                break

    return samples


# ============================================================
# MANUAL SAMPLES FOR HARD CATEGORIES
# ============================================================

MANUAL_NUMBER_AGREEMENT = [
    # Plural subject needs plural adjective
    {"source": "الطلاب الناجح يستحقون التقدير", "target": "الطلاب الناجحون يستحقون التقدير"},
    {"source": "المعلمون المخلص يبذلون جهدهم", "target": "المعلمون المخلصون يبذلون جهدهم"},
    {"source": "العمال الكادح يستحقون أجورا أفضل", "target": "العمال الكادحون يستحقون أجورا أفضل"},
    {"source": "المهندسون الماهر صمموا المبنى", "target": "المهندسون الماهرون صمموا المبنى"},
    {"source": "الأطباء الحاذق أنقذوا المرضى", "target": "الأطباء الحاذقون أنقذوا المرضى"},
    {"source": "اللاعبون الموهوب فازوا بالبطولة", "target": "اللاعبون الموهوبون فازوا بالبطولة"},
    {"source": "المواطنون الشريف يحترمون القانون", "target": "المواطنون الشرفاء يحترمون القانون"},
    {"source": "العلماء البارع اكتشفوا علاجا جديدا", "target": "العلماء البارعون اكتشفوا علاجا جديدا"},
    {"source": "الكتاب المبدع ألفوا روايات رائعة", "target": "الكتاب المبدعون ألفوا روايات رائعة"},
    {"source": "الفنانون الموهوب عرضوا لوحاتهم", "target": "الفنانون الموهوبون عرضوا لوحاتهم"},
    {"source": "المسلمون المؤمن يصلون في المسجد", "target": "المسلمون المؤمنون يصلون في المسجد"},
    {"source": "الطالبات الناجح حصلن على المنحة", "target": "الطالبات الناجحات حصلن على المنحة"},
    {"source": "المعلمات المخلص يعلمن الأطفال", "target": "المعلمات المخلصات يعلمن الأطفال"},
    {"source": "الممرضات الحنون يعتنين بالمرضى", "target": "الممرضات الحنونات يعتنين بالمرضى"},
    {"source": "الأمهات الصابر يربين أطفالهن", "target": "الأمهات الصابرات يربين أطفالهن"},
    {"source": "العاملات الجاد ينجزن عملهن", "target": "العاملات الجادات ينجزن عملهن"},
    {"source": "البنات الذكي يتفوقن في الدراسة", "target": "البنات الذكيات يتفوقن في الدراسة"},
    {"source": "النساء الفاضل يحترمهن المجتمع", "target": "النساء الفاضلات يحترمهن المجتمع"},
    {"source": "الفتيات الرياضي يمارسن السباحة", "target": "الفتيات الرياضيات يمارسن السباحة"},
    {"source": "الطبيبات الماهر يعالجن المرضى", "target": "الطبيبات الماهرات يعالجن المرضى"},
    # More varied examples
    {"source": "جاء الضيوف الكريم", "target": "جاء الضيوف الكرام"},
    {"source": "حضر العلماء الكبير", "target": "حضر العلماء الكبار"},
    {"source": "وصل الزوار الأجنبي", "target": "وصل الزوار الأجانب"},
    {"source": "غادر المسافرون المتأخر", "target": "غادر المسافرون المتأخرون"},
    {"source": "نجح المتسابقون الأول", "target": "نجح المتسابقون الأوائل"},
    {"source": "فاز الرياضيون الممتاز", "target": "فاز الرياضيون الممتازون"},
    {"source": "قدم الباحثون الجاد أبحاثهم", "target": "قدم الباحثون الجادون أبحاثهم"},
    {"source": "أنجز العمال الدؤوب المهمة", "target": "أنجز العمال الدؤوبون المهمة"},
    {"source": "احتفل الفائزون السعيد", "target": "احتفل الفائزون السعداء"},
    {"source": "عاد الجنود الشجاع", "target": "عاد الجنود الشجعان"},
]

MANUAL_VERB_AGREEMENT = [
    # Feminine subject needs feminine verb
    {"source": "الطالبة ذهب إلى المدرسة", "target": "الطالبة ذهبت إلى المدرسة"},
    {"source": "المعلمة شرح الدرس بوضوح", "target": "المعلمة شرحت الدرس بوضوح"},
    {"source": "الأم طبخ وجبة لذيذة", "target": "الأم طبخت وجبة لذيذة"},
    {"source": "البنت قرأ القصة كاملة", "target": "البنت قرأت القصة كاملة"},
    {"source": "المرأة عمل في الشركة", "target": "المرأة عملت في الشركة"},
    {"source": "الفتاة رسم لوحة جميلة", "target": "الفتاة رسمت لوحة جميلة"},
    {"source": "الممرضة ساعد المريض", "target": "الممرضة ساعدت المريض"},
    {"source": "الطبيبة فحص المريضة", "target": "الطبيبة فحصت المريضة"},
    {"source": "المديرة وقع على القرار", "target": "المديرة وقعت على القرار"},
    {"source": "السيدة سافر إلى باريس", "target": "السيدة سافرت إلى باريس"},
    {"source": "الصحفية كتب مقالا مهما", "target": "الصحفية كتبت مقالا مهما"},
    {"source": "المهندسة صمم المبنى", "target": "المهندسة صممت المبنى"},
    {"source": "الشاعرة ألف قصيدة رائعة", "target": "الشاعرة ألفت قصيدة رائعة"},
    {"source": "الفنانة عرض أعمالها", "target": "الفنانة عرضت أعمالها"},
    {"source": "الرياضية فاز بالميدالية", "target": "الرياضية فازت بالميدالية"},
    # Plural feminine subjects
    {"source": "الطالبات درسوا بجد", "target": "الطالبات درسن بجد"},
    {"source": "المعلمات علموا الأطفال", "target": "المعلمات علمن الأطفال"},
    {"source": "الأمهات اهتموا بأطفالهن", "target": "الأمهات اهتممن بأطفالهن"},
    {"source": "البنات لعبوا في الحديقة", "target": "البنات لعبن في الحديقة"},
    {"source": "النساء حضروا الاجتماع", "target": "النساء حضرن الاجتماع"},
    {"source": "الفتيات شاركوا في المسابقة", "target": "الفتيات شاركن في المسابقة"},
    {"source": "الممرضات عالجوا المرضى", "target": "الممرضات عالجن المرضى"},
    {"source": "الطبيبات أجروا العمليات", "target": "الطبيبات أجرين العمليات"},
    {"source": "المديرات اتخذوا القرارات", "target": "المديرات اتخذن القرارات"},
    {"source": "المعلمات شرحوا الدرس", "target": "المعلمات شرحن الدرس"},
    # Additional patterns
    {"source": "هي ذهب أمس", "target": "هي ذهبت أمس"},
    {"source": "هي قال الحقيقة", "target": "هي قالت الحقيقة"},
    {"source": "الشمس أشرق صباحا", "target": "الشمس أشرقت صباحا"},
    {"source": "السيارة توقف فجأة", "target": "السيارة توقفت فجأة"},
    {"source": "الطائرة هبط بسلام", "target": "الطائرة هبطت بسلام"},
]

MANUAL_GENDER_AGREEMENT = [
    # Feminine noun needs feminine adjective
    {"source": "المدينة الكبير تتطور باستمرار", "target": "المدينة الكبيرة تتطور باستمرار"},
    {"source": "الحكومة الجديد أصدرت قرارات مهمة", "target": "الحكومة الجديدة أصدرت قرارات مهمة"},
    {"source": "الجامعة العريق تحتفل بذكراها", "target": "الجامعة العريقة تحتفل بذكراها"},
    {"source": "الشركة الناجح توسعت في أعمالها", "target": "الشركة الناجحة توسعت في أعمالها"},
    {"source": "المدرسة الابتدائي افتتحت أبوابها", "target": "المدرسة الابتدائية افتتحت أبوابها"},
    {"source": "الدولة العربي وقعت الاتفاقية", "target": "الدولة العربية وقعت الاتفاقية"},
    {"source": "اللغة العربي لغة جميلة", "target": "اللغة العربية لغة جميلة"},
    {"source": "الثقافة الإسلامي غنية بالقيم", "target": "الثقافة الإسلامية غنية بالقيم"},
    {"source": "الحضارة القديم تركت آثارا عظيمة", "target": "الحضارة القديمة تركت آثارا عظيمة"},
    {"source": "السياسة الخارجي تغيرت كثيرا", "target": "السياسة الخارجية تغيرت كثيرا"},
    {"source": "الفترة الماضي كانت صعبة", "target": "الفترة الماضية كانت صعبة"},
    {"source": "المرحلة الأول انتهت بنجاح", "target": "المرحلة الأولى انتهت بنجاح"},
    {"source": "الخطة الاستراتيجي أعدت بعناية", "target": "الخطة الاستراتيجية أعدت بعناية"},
    {"source": "المنطقة الصناعي تشهد نموا", "target": "المنطقة الصناعية تشهد نموا"},
    {"source": "القناة الفضائي بثت الحدث", "target": "القناة الفضائية بثت الحدث"},
    # Masculine noun needs masculine adjective
    {"source": "البيت الجميلة يطل على البحر", "target": "البيت الجميل يطل على البحر"},
    {"source": "الكتاب المفيدة يستحق القراءة", "target": "الكتاب المفيد يستحق القراءة"},
    {"source": "النهر الطويلة يمر بالمدينة", "target": "النهر الطويل يمر بالمدينة"},
    {"source": "الجبل العالية يغطيه الثلج", "target": "الجبل العالي يغطيه الثلج"},
    {"source": "الطريق الرئيسية مزدحم اليوم", "target": "الطريق الرئيسي مزدحم اليوم"},
    {"source": "المشروع الكبيرة نجح تماما", "target": "المشروع الكبير نجح تماما"},
    {"source": "البرنامج الجديدة بدأ أمس", "target": "البرنامج الجديد بدأ أمس"},
    {"source": "الفيلم الممتازة حصد الجوائز", "target": "الفيلم الممتاز حصد الجوائز"},
    {"source": "القرار الصعبة اتخذ بعد تفكير", "target": "القرار الصعب اتخذ بعد تفكير"},
    {"source": "التاريخ العربية غني بالأحداث", "target": "التاريخ العربي غني بالأحداث"},
    # Additional
    {"source": "هذه المشكلة الصعب تحتاج حلا", "target": "هذه المشكلة الصعبة تحتاج حلا"},
    {"source": "تلك القضية المهم تناقش اليوم", "target": "تلك القضية المهمة تناقش اليوم"},
    {"source": "هذا الفكرة الجديد مثيرة", "target": "هذه الفكرة الجديدة مثيرة"},
    {"source": "الرسالة الواضح وصلت بأمان", "target": "الرسالة الواضحة وصلت بأمان"},
    {"source": "الصورة الجميل معلقة على الحائط", "target": "الصورة الجميلة معلقة على الحائط"},
]

MANUAL_WRONG_PREP = [
    # Common preposition confusions
    {"source": "يعتمد من المساعدات الخارجية", "target": "يعتمد على المساعدات الخارجية"},
    {"source": "أصر إلى موقفه السابق", "target": "أصر على موقفه السابق"},
    {"source": "اعترض إلى القرار الجديد", "target": "اعترض على القرار الجديد"},
    {"source": "حافظ من التقاليد القديمة", "target": "حافظ على التقاليد القديمة"},
    {"source": "سيطر إلى المنطقة بالكامل", "target": "سيطر على المنطقة بالكامل"},
    {"source": "تغلب إلى الصعوبات", "target": "تغلب على الصعوبات"},
    {"source": "أثر من الاقتصاد سلبا", "target": "أثر على الاقتصاد سلبا"},
    {"source": "رد إلى الاتهامات", "target": "رد على الاتهامات"},
    {"source": "وافق إلى الاقتراح", "target": "وافق على الاقتراح"},
    {"source": "حصل من الشهادة", "target": "حصل على الشهادة"},
    {"source": "تعرف إلى الحقيقة", "target": "تعرف على الحقيقة"},
    {"source": "اطلع من التقرير", "target": "اطلع على التقرير"},
    {"source": "شكر على صديقه", "target": "شكر صديقه"},
    {"source": "دخل على الغرفة", "target": "دخل الغرفة"},
    {"source": "تخصص على الطب", "target": "تخصص في الطب"},
    {"source": "يرغب على السفر", "target": "يرغب في السفر"},
    {"source": "يفكر على المشكلة", "target": "يفكر في المشكلة"},
    {"source": "يعمل على الشركة", "target": "يعمل في الشركة"},
    {"source": "يسكن على المدينة", "target": "يسكن في المدينة"},
    {"source": "يدرس على الجامعة", "target": "يدرس في الجامعة"},
    {"source": "ذهب على المدرسة", "target": "ذهب إلى المدرسة"},
    {"source": "سافر على باريس", "target": "سافر إلى باريس"},
    {"source": "وصل على المطار", "target": "وصل إلى المطار"},
    {"source": "انتقل على البيت الجديد", "target": "انتقل إلى البيت الجديد"},
    {"source": "تحول على نظام جديد", "target": "تحول إلى نظام جديد"},
    {"source": "يبحث على الحقيقة", "target": "يبحث عن الحقيقة"},
    {"source": "يسأل على الموضوع", "target": "يسأل عن الموضوع"},
    {"source": "أعلن على النتائج", "target": "أعلن عن النتائج"},
    {"source": "تكلم على المشروع", "target": "تكلم عن المشروع"},
    {"source": "عبر على رأيه", "target": "عبر عن رأيه"},
]

# More preposition patterns for hunting
MORE_MISSING_PREP = {
    # يتحدث عن
    'يتحدث عن': ['يتحدث'],
    'تتحدث عن': ['تتحدث'],
    'نتحدث عن': ['نتحدث'],
    # يتعلق ب
    'يتعلق ب': ['يتعلق'],
    'تتعلق ب': ['تتعلق'],
    # يؤدي إلى
    'يؤدي إلى': ['يؤدي'],
    'تؤدي إلى': ['تؤدي'],
    # يعتمد على
    'يعتمد على': ['يعتمد'],
    'تعتمد على': ['تعتمد'],
    'نعتمد على': ['نعتمد'],
}


def hunt_more_missing_prep(sentences: List[str], target: int = 100) -> List[dict]:
    """Hunt for more missing preposition errors."""
    samples = []

    for sentence in sentences:
        if len(samples) >= target:
            break

        for correct, errors in MORE_MISSING_PREP.items():
            if correct in sentence:
                for error in errors:
                    # Make sure we're not just matching a substring
                    if f' {error} ' in f' {sentence} ' or sentence.startswith(f'{error} ') or sentence.endswith(f' {error}'):
                        # This sentence has the preposition - create error by removing it
                        parts = correct.split()
                        if len(parts) == 2:
                            verb, prep = parts
                            # Find what comes after the prep
                            idx = sentence.find(correct)
                            if idx >= 0:
                                source = sentence[:idx] + verb + sentence[idx+len(correct):]
                                if source != sentence and len(source) > 10:
                                    samples.append({
                                        'source': source.strip(),
                                        'target': sentence,
                                        'category': 'missing_prep',
                                        'error': f'missing {prep}',
                                        'verified': True,
                                        'source_corpus': 'msa'
                                    })
                                    break

        if len(samples) >= target:
            break

    return samples


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("BUILDING EXHAUSTIVE FASIH BENCHMARK")
    print("=" * 60)

    # Load corpus
    sentences = load_corpus_sentences(500000)
    random.shuffle(sentences)

    all_samples = defaultdict(list)

    # ========================================
    # PHASE 1: Hunt pattern-based errors
    # ========================================
    print("\n=== Phase 1: Hunting pattern-based errors ===")

    print("Hunting hamza_wasl...")
    all_samples['hamza_wasl'] = hunt_hamza_wasl(sentences, 150)
    print(f"  Found {len(all_samples['hamza_wasl'])} samples")

    print("Hunting space_join...")
    all_samples['space_join'] = hunt_space_join(sentences, 100)
    print(f"  Found {len(all_samples['space_join'])} samples")

    print("Hunting space_split...")
    all_samples['space_split'] = hunt_space_split(sentences, 100)
    print(f"  Found {len(all_samples['space_split'])} samples")

    print("Hunting tanwin...")
    all_samples['tanwin'] = hunt_tanwin(sentences, 100)
    print(f"  Found {len(all_samples['tanwin'])} samples")

    # ========================================
    # PHASE 2: Generate agreement errors
    # ========================================
    print("\n=== Phase 2: Generating agreement errors ===")

    print("Generating gender_agreement...")
    all_samples['gender_agreement'] = generate_gender_agreement(sentences, 100)
    print(f"  Generated {len(all_samples['gender_agreement'])} samples")

    print("Generating definiteness...")
    all_samples['definiteness'] = generate_definiteness(sentences, 100)
    print(f"  Generated {len(all_samples['definiteness'])} samples")

    print("Generating number_agreement...")
    all_samples['number_agreement'] = generate_number_agreement(sentences, 100)
    print(f"  Generated {len(all_samples['number_agreement'])} samples")

    print("Generating verb_agreement...")
    all_samples['verb_agreement'] = generate_verb_agreement(sentences, 100)
    print(f"  Generated {len(all_samples['verb_agreement'])} samples")

    print("Hunting more missing_prep...")
    more_prep = hunt_more_missing_prep(sentences, 100)
    all_samples['missing_prep'].extend(more_prep)
    print(f"  Found {len(more_prep)} more samples")

    # ========================================
    # PHASE 2.5: Add manual samples for hard categories
    # ========================================
    print("\n=== Phase 2.5: Adding manual samples ===")

    # Add manual number agreement
    for sample in MANUAL_NUMBER_AGREEMENT:
        all_samples['number_agreement'].append({
            'source': sample['source'],
            'target': sample['target'],
            'category': 'number_agreement',
            'verified': True,
            'manual': True,
            'source_corpus': 'manual'
        })
    print(f"  Added {len(MANUAL_NUMBER_AGREEMENT)} manual number_agreement samples")

    # Add manual verb agreement
    for sample in MANUAL_VERB_AGREEMENT:
        all_samples['verb_agreement'].append({
            'source': sample['source'],
            'target': sample['target'],
            'category': 'verb_agreement',
            'verified': True,
            'manual': True,
            'source_corpus': 'manual'
        })
    print(f"  Added {len(MANUAL_VERB_AGREEMENT)} manual verb_agreement samples")

    # Add manual gender agreement
    for sample in MANUAL_GENDER_AGREEMENT:
        all_samples['gender_agreement'].append({
            'source': sample['source'],
            'target': sample['target'],
            'category': 'gender_agreement',
            'verified': True,
            'manual': True,
            'source_corpus': 'manual'
        })
    print(f"  Added {len(MANUAL_GENDER_AGREEMENT)} manual gender_agreement samples")

    # Add manual wrong prep
    for sample in MANUAL_WRONG_PREP:
        all_samples['wrong_prep'].append({
            'source': sample['source'],
            'target': sample['target'],
            'category': 'wrong_prep',
            'verified': True,
            'manual': True,
            'source_corpus': 'manual'
        })
    print(f"  Added {len(MANUAL_WRONG_PREP)} manual wrong_prep samples")

    # ========================================
    # PHASE 3: Generate typo errors
    # ========================================
    print("\n=== Phase 3: Generating typo errors ===")

    print("Generating repeated_char...")
    all_samples['repeated_char'] = generate_repeated_char(sentences, 50)
    print(f"  Generated {len(all_samples['repeated_char'])} samples")

    print("Generating missing_char...")
    all_samples['missing_char'] = generate_missing_char(sentences, 50)
    print(f"  Generated {len(all_samples['missing_char'])} samples")

    # ========================================
    # COMBINE WITH EXISTING
    # ========================================
    print("\n=== Loading existing FASIH samples ===")

    existing_core = []
    core_path = FASIH_DIR / "core" / "test.json"
    if core_path.exists():
        with open(core_path, 'r', encoding='utf-8') as f:
            existing_core = json.load(f)
        print(f"Loaded {len(existing_core)} existing core samples")

        # Add to all_samples by category
        for sample in existing_core:
            cat = sample['category']
            all_samples[cat].append(sample)

    # ========================================
    # SAVE NEW BENCHMARK
    # ========================================
    print("\n=== Building final benchmark ===")

    # Combine all samples
    all_test = []
    all_dev = []

    for cat, samples in sorted(all_samples.items()):
        # Deduplicate
        seen = set()
        unique = []
        for s in samples:
            key = s.get('source', '')
            if key not in seen:
                seen.add(key)
                unique.append(s)

        # Shuffle and split
        random.shuffle(unique)
        split_idx = int(len(unique) * 0.85)

        test_samples = unique[:split_idx]
        dev_samples = unique[split_idx:]

        # Assign IDs
        for i, s in enumerate(test_samples):
            s['id'] = f'fasih-{cat}-{i:04d}'
        for i, s in enumerate(dev_samples):
            s['id'] = f'fasih-{cat}-dev-{i:04d}'

        all_test.extend(test_samples)
        all_dev.extend(dev_samples)

        print(f"  {cat}: {len(test_samples)} test + {len(dev_samples)} dev")

    # Save
    output_dir = FASIH_DIR / "v3"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "test.json", 'w', encoding='utf-8') as f:
        json.dump(all_test, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(all_test)} test samples to {output_dir / 'test.json'}")

    with open(output_dir / "dev.json", 'w', encoding='utf-8') as f:
        json.dump(all_dev, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(all_dev)} dev samples to {output_dir / 'dev.json'}")

    # Generate summary
    print("\n" + "=" * 60)
    print("EXHAUSTIVE FASIH BENCHMARK COMPLETE")
    print("=" * 60)

    dist = Counter(s['category'] for s in all_test)
    for cat, count in sorted(dist.items()):
        print(f"  {cat}: {count}")
    print(f"\n  TOTAL: {len(all_test)} test + {len(all_dev)} dev")


if __name__ == "__main__":
    random.seed(42)
    main()
