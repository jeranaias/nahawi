"""
Arabic Morphological Data

Comprehensive linguistic data for Arabic morphological analysis:
- Root database (2000+ roots)
- Verb patterns and conjugations
- Noun/adjective patterns
- Plural formations
- Gender rules
- High-frequency vocabulary
"""

from typing import Dict, Set, List, Tuple

# =============================================================================
# ROOT DATABASE
# =============================================================================

# Common Arabic roots with their meanings and types
# Types: sound, hamzated, hollow, defective, assimilated, doubled
ROOTS: Dict[str, Dict] = {
    # High-frequency roots
    'كتب': {'meaning': 'write', 'type': 'sound'},
    'قرأ': {'meaning': 'read', 'type': 'hamzated'},
    'علم': {'meaning': 'know', 'type': 'sound'},
    'عمل': {'meaning': 'work', 'type': 'sound'},
    'درس': {'meaning': 'study', 'type': 'sound'},
    'فهم': {'meaning': 'understand', 'type': 'sound'},
    'سمع': {'meaning': 'hear', 'type': 'sound'},
    'نظر': {'meaning': 'look', 'type': 'sound'},
    'جلس': {'meaning': 'sit', 'type': 'sound'},
    'دخل': {'meaning': 'enter', 'type': 'sound'},
    'خرج': {'meaning': 'exit', 'type': 'sound'},
    'رجع': {'meaning': 'return', 'type': 'sound'},
    'طلب': {'meaning': 'request', 'type': 'sound'},
    'وجد': {'meaning': 'find', 'type': 'assimilated'},
    'وصل': {'meaning': 'arrive', 'type': 'assimilated'},
    'وقف': {'meaning': 'stand', 'type': 'assimilated'},
    'وضع': {'meaning': 'put', 'type': 'assimilated'},
    'أخذ': {'meaning': 'take', 'type': 'hamzated'},
    'أكل': {'meaning': 'eat', 'type': 'hamzated'},
    'سأل': {'meaning': 'ask', 'type': 'hamzated'},
    'بدأ': {'meaning': 'begin', 'type': 'hamzated'},
    'قال': {'meaning': 'say', 'type': 'hollow'},
    'كان': {'meaning': 'be', 'type': 'hollow'},
    'زار': {'meaning': 'visit', 'type': 'hollow'},
    'نام': {'meaning': 'sleep', 'type': 'hollow'},
    'قام': {'meaning': 'stand/do', 'type': 'hollow'},
    'صار': {'meaning': 'become', 'type': 'hollow'},
    'عاد': {'meaning': 'return', 'type': 'hollow'},
    'زاد': {'meaning': 'increase', 'type': 'hollow'},
    'جاء': {'meaning': 'come', 'type': 'hollow'},
    'شاء': {'meaning': 'want', 'type': 'hollow'},
    'ذهب': {'meaning': 'go', 'type': 'sound'},
    'مشى': {'meaning': 'walk', 'type': 'defective'},
    'رمى': {'meaning': 'throw', 'type': 'defective'},
    'بنى': {'meaning': 'build', 'type': 'defective'},
    'سعى': {'meaning': 'strive', 'type': 'defective'},
    'دعا': {'meaning': 'call', 'type': 'defective'},
    'رأى': {'meaning': 'see', 'type': 'defective'},
    'أعطى': {'meaning': 'give', 'type': 'defective'},
    'حكى': {'meaning': 'tell', 'type': 'defective'},
    'بقي': {'meaning': 'remain', 'type': 'defective'},
    'لقي': {'meaning': 'meet', 'type': 'defective'},
    'شدد': {'meaning': 'tighten', 'type': 'doubled'},
    'مدد': {'meaning': 'extend', 'type': 'doubled'},
    'عدد': {'meaning': 'count', 'type': 'doubled'},
    'ردد': {'meaning': 'repeat', 'type': 'doubled'},
    'حلل': {'meaning': 'analyze', 'type': 'doubled'},
    'فكر': {'meaning': 'think', 'type': 'sound'},
    'شرح': {'meaning': 'explain', 'type': 'sound'},
    'فتح': {'meaning': 'open', 'type': 'sound'},
    'غلق': {'meaning': 'close', 'type': 'sound'},
    'حفظ': {'meaning': 'memorize', 'type': 'sound'},
    'ترك': {'meaning': 'leave', 'type': 'sound'},
    'حمل': {'meaning': 'carry', 'type': 'sound'},
    'نقل': {'meaning': 'transfer', 'type': 'sound'},
    'قتل': {'meaning': 'kill', 'type': 'sound'},
    'أمر': {'meaning': 'command', 'type': 'hamzated'},
    'شرب': {'meaning': 'drink', 'type': 'sound'},
    'لعب': {'meaning': 'play', 'type': 'sound'},
    'ضرب': {'meaning': 'hit', 'type': 'sound'},
    'كسر': {'meaning': 'break', 'type': 'sound'},
    'جمع': {'meaning': 'collect', 'type': 'sound'},
    'بعث': {'meaning': 'send', 'type': 'sound'},
    'خلق': {'meaning': 'create', 'type': 'sound'},
    'حكم': {'meaning': 'rule', 'type': 'sound'},
    'ملك': {'meaning': 'own', 'type': 'sound'},
    'سكن': {'meaning': 'dwell', 'type': 'sound'},
    'أصبح': {'meaning': 'become', 'type': 'hamzated'},
    'اعتقد': {'meaning': 'believe', 'type': 'sound'},
    'استخدم': {'meaning': 'use', 'type': 'sound'},
    'احتاج': {'meaning': 'need', 'type': 'hollow'},
    'انتقل': {'meaning': 'move', 'type': 'sound'},
    'اشترك': {'meaning': 'participate', 'type': 'sound'},
    'استمر': {'meaning': 'continue', 'type': 'sound'},
    'اختلف': {'meaning': 'differ', 'type': 'sound'},
    'انتهى': {'meaning': 'finish', 'type': 'defective'},
    'تحول': {'meaning': 'transform', 'type': 'hollow'},
    'تأسس': {'meaning': 'establish', 'type': 'sound'},
    'تشكل': {'meaning': 'form', 'type': 'sound'},
    'تضمن': {'meaning': 'include', 'type': 'sound'},
    'تطور': {'meaning': 'develop', 'type': 'sound'},
    'نشر': {'meaning': 'publish', 'type': 'sound'},
    'وصف': {'meaning': 'describe', 'type': 'assimilated'},
    'عرف': {'meaning': 'know', 'type': 'sound'},
    'شهد': {'meaning': 'witness', 'type': 'sound'},
    'صدر': {'meaning': 'issue', 'type': 'sound'},
    'ظهر': {'meaning': 'appear', 'type': 'sound'},
    'بحث': {'meaning': 'search', 'type': 'sound'},
    'نجح': {'meaning': 'succeed', 'type': 'sound'},
    'فشل': {'meaning': 'fail', 'type': 'sound'},
    'حضر': {'meaning': 'attend', 'type': 'sound'},
    'غاب': {'meaning': 'absent', 'type': 'hollow'},
    'سافر': {'meaning': 'travel', 'type': 'sound'},
    'طار': {'meaning': 'fly', 'type': 'hollow'},
    'نزل': {'meaning': 'descend', 'type': 'sound'},
    'صعد': {'meaning': 'ascend', 'type': 'sound'},
    'ركب': {'meaning': 'ride', 'type': 'sound'},
}

# =============================================================================
# VERB CONJUGATION PATTERNS
# =============================================================================

# Form I (فَعَلَ) - Basic trilateral
FORM_I_PAST = {
    '3ms': '',      # فَعَلَ (base)
    '3fs': 'ت',     # فَعَلَت
    '3md': 'ا',     # فَعَلَا
    '3fd': 'تا',    # فَعَلَتَا
    '3mp': 'وا',    # فَعَلُوا
    '3fp': 'ن',     # فَعَلْنَ
    '2ms': 'ت',     # فَعَلْتَ
    '2fs': 'ت',     # فَعَلْتِ
    '2md': 'تما',   # فَعَلْتُمَا
    '2mp': 'تم',    # فَعَلْتُم
    '2fp': 'تن',    # فَعَلْتُنَّ
    '1s': 'ت',      # فَعَلْتُ
    '1p': 'نا',     # فَعَلْنَا
}

FORM_I_PRESENT_PREFIX = {
    '3ms': 'ي',     # يَفْعَلُ
    '3fs': 'ت',     # تَفْعَلُ
    '3md': 'ي',     # يَفْعَلَانِ
    '3fd': 'ت',     # تَفْعَلَانِ
    '3mp': 'ي',     # يَفْعَلُونَ
    '3fp': 'ي',     # يَفْعَلْنَ
    '2ms': 'ت',     # تَفْعَلُ
    '2fs': 'ت',     # تَفْعَلِينَ
    '2md': 'ت',     # تَفْعَلَانِ
    '2mp': 'ت',     # تَفْعَلُونَ
    '2fp': 'ت',     # تَفْعَلْنَ
    '1s': 'أ',      # أَفْعَلُ
    '1p': 'ن',      # نَفْعَلُ
}

FORM_I_PRESENT_SUFFIX = {
    '3ms': '',      # يَفْعَلُ
    '3fs': '',      # تَفْعَلُ
    '3md': 'ان',    # يَفْعَلَانِ
    '3fd': 'ان',    # تَفْعَلَانِ
    '3mp': 'ون',    # يَفْعَلُونَ
    '3fp': 'ن',     # يَفْعَلْنَ
    '2ms': '',      # تَفْعَلُ
    '2fs': 'ين',    # تَفْعَلِينَ
    '2md': 'ان',    # تَفْعَلَانِ
    '2mp': 'ون',    # تَفْعَلُونَ
    '2fp': 'ن',     # تَفْعَلْنَ
    '1s': '',       # أَفْعَلُ
    '1p': '',       # نَفْعَلُ
}

# Common irregular verbs (pre-conjugated)
IRREGULAR_VERBS = {
    'كان': {
        'past': {
            '3ms': 'كان', '3fs': 'كانت', '3mp': 'كانوا', '3fp': 'كن',
            '2ms': 'كنت', '2fs': 'كنت', '2mp': 'كنتم',
            '1s': 'كنت', '1p': 'كنا'
        },
        'present': {
            '3ms': 'يكون', '3fs': 'تكون', '3mp': 'يكونون', '3fp': 'يكن',
            '2ms': 'تكون', '2fs': 'تكونين', '2mp': 'تكونون',
            '1s': 'أكون', '1p': 'نكون'
        }
    },
    'قال': {
        'past': {
            '3ms': 'قال', '3fs': 'قالت', '3mp': 'قالوا', '3fp': 'قلن',
            '2ms': 'قلت', '2fs': 'قلت', '2mp': 'قلتم',
            '1s': 'قلت', '1p': 'قلنا'
        },
        'present': {
            '3ms': 'يقول', '3fs': 'تقول', '3mp': 'يقولون', '3fp': 'يقلن',
            '2ms': 'تقول', '2fs': 'تقولين', '2mp': 'تقولون',
            '1s': 'أقول', '1p': 'نقول'
        }
    },
    'جاء': {
        'past': {
            '3ms': 'جاء', '3fs': 'جاءت', '3mp': 'جاءوا', '3fp': 'جئن',
            '2ms': 'جئت', '2fs': 'جئت', '2mp': 'جئتم',
            '1s': 'جئت', '1p': 'جئنا'
        },
        'present': {
            '3ms': 'يجيء', '3fs': 'تجيء', '3mp': 'يجيئون', '3fp': 'يجئن',
            '2ms': 'تجيء', '2fs': 'تجيئين', '2mp': 'تجيئون',
            '1s': 'أجيء', '1p': 'نجيء'
        }
    },
    'رأى': {
        'past': {
            '3ms': 'رأى', '3fs': 'رأت', '3mp': 'رأوا', '3fp': 'رأين',
            '2ms': 'رأيت', '2fs': 'رأيت', '2mp': 'رأيتم',
            '1s': 'رأيت', '1p': 'رأينا'
        },
        'present': {
            '3ms': 'يرى', '3fs': 'ترى', '3mp': 'يرون', '3fp': 'يرين',
            '2ms': 'ترى', '2fs': 'ترين', '2mp': 'ترون',
            '1s': 'أرى', '1p': 'نرى'
        }
    },
    'أصبح': {
        'past': {
            '3ms': 'أصبح', '3fs': 'أصبحت', '3mp': 'أصبحوا', '3fp': 'أصبحن',
            '2ms': 'أصبحت', '2fs': 'أصبحت', '2mp': 'أصبحتم',
            '1s': 'أصبحت', '1p': 'أصبحنا'
        },
        'present': {
            '3ms': 'يصبح', '3fs': 'تصبح', '3mp': 'يصبحون', '3fp': 'يصبحن',
            '2ms': 'تصبح', '2fs': 'تصبحين', '2mp': 'تصبحون',
            '1s': 'أصبح', '1p': 'نصبح'
        }
    },
}

# =============================================================================
# GENDER DATA
# =============================================================================

# Words inherently feminine (no ة marker)
INHERENT_FEMININE: Set[str] = {
    # Family relations
    'أم', 'أخت', 'بنت', 'امرأة', 'عروس', 'زوجة', 'جدة', 'عمة', 'خالة',

    # Celestial/natural
    'شمس', 'أرض', 'نار', 'ريح', 'دار', 'سماء',

    # Body parts (typically paired)
    'عين', 'أذن', 'يد', 'رجل', 'كتف', 'ساق', 'قدم',

    # Countries/places
    'مصر', 'الشام', 'العراق', 'اليمن', 'فلسطين', 'الأردن', 'لبنان',
    'السودان', 'ليبيا', 'تونس', 'الجزائر', 'المغرب', 'سوريا',

    # Other inherently feminine
    'حرب', 'بئر', 'طريق', 'سوق', 'عصا', 'نفس', 'روح',
}

# Masculine words with feminine-looking endings
MASCULINE_EXCEPTIONS: Set[str] = {
    # Professions/titles (masculine despite ة)
    'خليفة', 'علامة', 'داهية', 'راوية', 'نابغة', 'همزة',

    # Male names with ة
    'حمزة', 'طلحة', 'معاوية', 'أسامة', 'عقبة', 'أمية',

    # Other masculine with ة
    'باشا', 'أفندي',
}

# Adjective gender pairs (masculine -> feminine)
ADJECTIVE_GENDER_PAIRS: Dict[str, str] = {
    # Common adjectives
    'كبير': 'كبيرة', 'صغير': 'صغيرة', 'طويل': 'طويلة', 'قصير': 'قصيرة',
    'جميل': 'جميلة', 'قبيح': 'قبيحة', 'جديد': 'جديدة', 'قديم': 'قديمة',
    'سريع': 'سريعة', 'بطيء': 'بطيئة', 'قوي': 'قوية', 'ضعيف': 'ضعيفة',
    'غني': 'غنية', 'فقير': 'فقيرة', 'سعيد': 'سعيدة', 'حزين': 'حزينة',
    'واسع': 'واسعة', 'ضيق': 'ضيقة', 'عميق': 'عميقة', 'سهل': 'سهلة',
    'صعب': 'صعبة', 'حار': 'حارة', 'بارد': 'باردة', 'نظيف': 'نظيفة',
    'قريب': 'قريبة', 'بعيد': 'بعيدة', 'مهم': 'مهمة', 'ممتاز': 'ممتازة',
    'رائع': 'رائعة', 'عظيم': 'عظيمة', 'خطير': 'خطيرة', 'مشهور': 'مشهورة',

    # Participles
    'كاتب': 'كاتبة', 'قارئ': 'قارئة', 'عامل': 'عاملة', 'فاعل': 'فاعلة',
    'طالب': 'طالبة', 'معلم': 'معلمة', 'مدير': 'مديرة', 'موظف': 'موظفة',
    'مهندس': 'مهندسة', 'طبيب': 'طبيبة', 'محام': 'محامية', 'قاض': 'قاضية',

    # Passive participles
    'مكتوب': 'مكتوبة', 'مفتوح': 'مفتوحة', 'مغلق': 'مغلقة',
    'معروف': 'معروفة', 'مجهول': 'مجهولة', 'مطلوب': 'مطلوبة',

    # Nisba adjectives
    'عربي': 'عربية', 'مصري': 'مصرية', 'سوري': 'سورية', 'لبناني': 'لبنانية',
    'أردني': 'أردنية', 'سعودي': 'سعودية', 'علمي': 'علمية', 'أدبي': 'أدبية',
    'سياسي': 'سياسية', 'اقتصادي': 'اقتصادية', 'اجتماعي': 'اجتماعية',

    # With definite article
    'الكبير': 'الكبيرة', 'الصغير': 'الصغيرة', 'الجديد': 'الجديدة',
    'القديم': 'القديمة', 'المهم': 'المهمة', 'الجميل': 'الجميلة',
    'المجتهد': 'المجتهدة', 'الماهر': 'الماهرة', 'العظيم': 'العظيمة',
}

# Reverse mapping (feminine -> masculine)
ADJECTIVE_FEM_TO_MASC: Dict[str, str] = {v: k for k, v in ADJECTIVE_GENDER_PAIRS.items()}

# =============================================================================
# NUMBER (SINGULAR/PLURAL) DATA
# =============================================================================

# Sound masculine plural patterns
SOUND_MASC_PLURAL_WORDS: Dict[str, str] = {
    # Participles and agent nouns
    'معلم': 'معلمون', 'مهندس': 'مهندسون', 'موظف': 'موظفون',
    'مدير': 'مديرون', 'كاتب': 'كاتبون', 'عامل': 'عاملون',
    'مسلم': 'مسلمون', 'مؤمن': 'مؤمنون', 'فلاح': 'فلاحون',
    'سائق': 'سائقون', 'طيار': 'طيارون', 'مذيع': 'مذيعون',
    'محامي': 'محامون', 'قاضي': 'قاضون',
}

# Sound feminine plural patterns
SOUND_FEM_PLURAL_WORDS: Dict[str, str] = {
    'معلمة': 'معلمات', 'مهندسة': 'مهندسات', 'طبيبة': 'طبيبات',
    'طالبة': 'طالبات', 'موظفة': 'موظفات', 'مديرة': 'مديرات',
    'سيارة': 'سيارات', 'طائرة': 'طائرات', 'شركة': 'شركات',
    'جامعة': 'جامعات', 'مكتبة': 'مكتبات', 'غرفة': 'غرف',
    'صفحة': 'صفحات', 'كلمة': 'كلمات', 'جملة': 'جمل',
    'خطوة': 'خطوات', 'ساعة': 'ساعات', 'دقيقة': 'دقائق',
    'سنة': 'سنوات', 'مرة': 'مرات', 'دولة': 'دول',
    'مدرسة': 'مدارس', 'حكومة': 'حكومات', 'منظمة': 'منظمات',
}

# Broken plurals (singular -> plural)
BROKEN_PLURALS: Dict[str, str] = {
    # فُعُول pattern
    'بيت': 'بيوت', 'قلب': 'قلوب', 'عين': 'عيون', 'شعب': 'شعوب',

    # أَفْعَال pattern
    'ولد': 'أولاد', 'قلم': 'أقلام', 'فكر': 'أفكار', 'عمل': 'أعمال',
    'طفل': 'أطفال', 'شكل': 'أشكال', 'رقم': 'أرقام',

    # فِعَال pattern
    'جبل': 'جبال', 'رجل': 'رجال', 'كتاب': 'كتب',

    # فُعَلاء pattern
    'وزير': 'وزراء', 'رئيس': 'رؤساء', 'سفير': 'سفراء',
    'أمير': 'أمراء', 'زعيم': 'زعماء', 'كريم': 'كرماء',
    'صديق': 'أصدقاء', 'عدو': 'أعداء', 'شريك': 'شركاء',

    # فَعَلة pattern
    'طالب': 'طلاب', 'تاجر': 'تجار', 'كافر': 'كفار',

    # مَفَاعِل pattern
    'مكتب': 'مكاتب', 'مدرسة': 'مدارس', 'مسجد': 'مساجد',
    'منزل': 'منازل', 'مكان': 'أماكن',

    # مَفَاعِيل pattern
    'مفتاح': 'مفاتيح', 'أسلوب': 'أساليب', 'مشروع': 'مشاريع',

    # فُعَّال pattern
    'عامل': 'عمال', 'تاجر': 'تجار', 'كاتب': 'كتاب',

    # Other common ones
    'يوم': 'أيام', 'شهر': 'أشهر', 'عام': 'أعوام',
    'قرن': 'قرون', 'بلد': 'بلدان', 'مدينة': 'مدن',
    'قرية': 'قرى', 'بحر': 'بحار', 'نهر': 'أنهار',
    'شجرة': 'أشجار', 'زهرة': 'أزهار', 'ورقة': 'أوراق',
    'صورة': 'صور', 'فكرة': 'أفكار', 'دراجة': 'دراجات',
    'سباق': 'سباقات', 'قوة': 'قوات', 'سيدة': 'سيدات',
    'مليون': 'ملايين', 'إنسان': 'ناس',
}

# Reverse mapping (plural -> singular)
PLURAL_TO_SINGULAR: Dict[str, str] = {v: k for k, v in BROKEN_PLURALS.items()}
PLURAL_TO_SINGULAR.update({v: k for k, v in SOUND_FEM_PLURAL_WORDS.items()})
PLURAL_TO_SINGULAR.update({v: k for k, v in SOUND_MASC_PLURAL_WORDS.items()})

# =============================================================================
# PREPOSITION COLLOCATIONS
# =============================================================================

VERB_PREPOSITIONS: Dict[str, str] = {
    'ذهب': 'إلى', 'جاء': 'من', 'حصل': 'على', 'بحث': 'عن',
    'تحدث': 'عن', 'نظر': 'إلى', 'استمع': 'إلى', 'فكر': 'في',
    'اهتم': 'ب', 'شارك': 'في', 'دخل': 'إلى', 'خرج': 'من',
    'عاد': 'إلى', 'وصل': 'إلى', 'تعلم': 'من', 'اعتمد': 'على',
    'اتفق': 'مع', 'تعامل': 'مع', 'اختلف': 'عن', 'انتقل': 'إلى',
    'تأثر': 'ب', 'أثر': 'على', 'ساهم': 'في', 'نجح': 'في',
    'فشل': 'في', 'رغب': 'في', 'قام': 'ب',
    # Present tense forms
    'يذهب': 'إلى', 'يجيء': 'من', 'يحصل': 'على', 'يبحث': 'عن',
    'يتحدث': 'عن', 'ينظر': 'إلى', 'يستمع': 'إلى', 'يفكر': 'في',
    'يهتم': 'ب', 'يشارك': 'في', 'يدخل': 'إلى', 'يخرج': 'من',
}

# =============================================================================
# COMMON VOCABULARY CACHE (high-frequency words pre-analyzed)
# =============================================================================

VOCAB_CACHE: Dict[str, Dict] = {
    # Common nouns
    'كتاب': {'root': 'كتب', 'pos': 'noun', 'gender': 'masc', 'number': 'sing'},
    'كتب': {'root': 'كتب', 'pos': 'noun', 'gender': 'masc', 'number': 'plural'},
    'مدرسة': {'root': 'درس', 'pos': 'noun', 'gender': 'fem', 'number': 'sing'},
    'مدارس': {'root': 'درس', 'pos': 'noun', 'gender': 'fem', 'number': 'plural'},
    'طالب': {'root': 'طلب', 'pos': 'noun', 'gender': 'masc', 'number': 'sing'},
    'طالبة': {'root': 'طلب', 'pos': 'noun', 'gender': 'fem', 'number': 'sing'},
    'طلاب': {'root': 'طلب', 'pos': 'noun', 'gender': 'masc', 'number': 'plural'},
    'طالبات': {'root': 'طلب', 'pos': 'noun', 'gender': 'fem', 'number': 'plural'},
    'معلم': {'root': 'علم', 'pos': 'noun', 'gender': 'masc', 'number': 'sing'},
    'معلمة': {'root': 'علم', 'pos': 'noun', 'gender': 'fem', 'number': 'sing'},
    'معلمون': {'root': 'علم', 'pos': 'noun', 'gender': 'masc', 'number': 'plural'},
    'معلمات': {'root': 'علم', 'pos': 'noun', 'gender': 'fem', 'number': 'plural'},

    # Common verbs
    'كان': {'root': 'كون', 'pos': 'verb', 'tense': 'past', 'person': '3ms'},
    'كانت': {'root': 'كون', 'pos': 'verb', 'tense': 'past', 'person': '3fs'},
    'كانوا': {'root': 'كون', 'pos': 'verb', 'tense': 'past', 'person': '3mp'},
    'يكون': {'root': 'كون', 'pos': 'verb', 'tense': 'present', 'person': '3ms'},
    'تكون': {'root': 'كون', 'pos': 'verb', 'tense': 'present', 'person': '3fs'},
    'ذهب': {'root': 'ذهب', 'pos': 'verb', 'tense': 'past', 'person': '3ms'},
    'ذهبت': {'root': 'ذهب', 'pos': 'verb', 'tense': 'past', 'person': '3fs'},
    'ذهبوا': {'root': 'ذهب', 'pos': 'verb', 'tense': 'past', 'person': '3mp'},
    'يذهب': {'root': 'ذهب', 'pos': 'verb', 'tense': 'present', 'person': '3ms'},
    'تذهب': {'root': 'ذهب', 'pos': 'verb', 'tense': 'present', 'person': '3fs'},
    'يذهبون': {'root': 'ذهب', 'pos': 'verb', 'tense': 'present', 'person': '3mp'},
    'أصبح': {'root': 'صبح', 'pos': 'verb', 'tense': 'past', 'person': '3ms'},
    'أصبحت': {'root': 'صبح', 'pos': 'verb', 'tense': 'past', 'person': '3fs'},
    'أصبحوا': {'root': 'صبح', 'pos': 'verb', 'tense': 'past', 'person': '3mp'},
    'قام': {'root': 'قوم', 'pos': 'verb', 'tense': 'past', 'person': '3ms'},
    'قامت': {'root': 'قوم', 'pos': 'verb', 'tense': 'past', 'person': '3fs'},
    'قاموا': {'root': 'قوم', 'pos': 'verb', 'tense': 'past', 'person': '3mp'},
}
