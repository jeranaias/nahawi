#!/usr/bin/env python3
"""
Arabic Verb Forms (أوزان الفعل)

The 10 derived forms of the Arabic triliteral verb:
Form I   (فَعَلَ)      - Basic meaning
Form II  (فَعَّلَ)     - Intensive, causative, denominative
Form III (فَاعَلَ)     - Reciprocal, attempt
Form IV  (أَفْعَلَ)    - Causative, transitive
Form V   (تَفَعَّلَ)   - Reflexive of II, gradual
Form VI  (تَفَاعَلَ)   - Reciprocal reflexive, pretense
Form VII (اِنْفَعَلَ)  - Passive, reflexive, inchoative
Form VIII (اِفْتَعَلَ) - Reflexive, middle voice
Form IX  (اِفْعَلَّ)   - Colors and physical defects (rare)
Form X   (اِسْتَفْعَلَ) - Requestative, considerative

Each form has:
- Past tense pattern (ماضي)
- Present tense pattern (مضارع)
- Active participle (اسم الفاعل)
- Passive participle (اسم المفعول)
- Verbal noun patterns (مصدر)
"""

from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Optional, Dict


class VerbForm(Enum):
    """Arabic verb forms I-X."""
    I = 1
    II = 2
    III = 3
    IV = 4
    V = 5
    VI = 6
    VII = 7
    VIII = 8
    IX = 9
    X = 10


@dataclass
class FormPattern:
    """Pattern for a verb form."""
    form: VerbForm
    name_ar: str
    name_en: str
    past_pattern: str       # Using ف-ع-ل as template
    present_pattern: str    # يَفْعَلُ pattern
    active_participle: str  # اسم الفاعل
    passive_participle: str # اسم المفعول
    verbal_nouns: List[str] # مصدر (can have multiple patterns)
    description: str


# Verb Form Patterns
# Using ف (fa), ع (ain), ل (lam) as the three root letters
# Vowel markers: a=fatha, i=kasra, u=damma, aa=long a, etc.

VERB_FORMS: Dict[VerbForm, FormPattern] = {
    VerbForm.I: FormPattern(
        form=VerbForm.I,
        name_ar="فَعَلَ",
        name_en="Form I",
        past_pattern="فَعَلَ",
        present_pattern="يَفْعَلُ",  # Most common, but varies (يَفْعِلُ، يَفْعُلُ)
        active_participle="فَاعِل",
        passive_participle="مَفْعُول",
        verbal_nouns=["فَعْل", "فِعَال", "فُعُول", "فَعَلَان"],  # Variable
        description="Basic form with primary meaning"
    ),

    VerbForm.II: FormPattern(
        form=VerbForm.II,
        name_ar="فَعَّلَ",
        name_en="Form II",
        past_pattern="فَعَّلَ",      # Doubled middle letter
        present_pattern="يُفَعِّلُ",
        active_participle="مُفَعِّل",
        passive_participle="مُفَعَّل",
        verbal_nouns=["تَفْعِيل", "تَفْعِلَة"],
        description="Intensive, causative, or denominative"
    ),

    VerbForm.III: FormPattern(
        form=VerbForm.III,
        name_ar="فَاعَلَ",
        name_en="Form III",
        past_pattern="فَاعَلَ",      # Long a after first letter
        present_pattern="يُفَاعِلُ",
        active_participle="مُفَاعِل",
        passive_participle="مُفَاعَل",
        verbal_nouns=["مُفَاعَلَة", "فِعَال"],
        description="Reciprocal action, attempt"
    ),

    VerbForm.IV: FormPattern(
        form=VerbForm.IV,
        name_ar="أَفْعَلَ",
        name_en="Form IV",
        past_pattern="أَفْعَلَ",     # Hamza prefix
        present_pattern="يُفْعِلُ",
        active_participle="مُفْعِل",
        passive_participle="مُفْعَل",
        verbal_nouns=["إِفْعَال"],
        description="Causative, transitive"
    ),

    VerbForm.V: FormPattern(
        form=VerbForm.V,
        name_ar="تَفَعَّلَ",
        name_en="Form V",
        past_pattern="تَفَعَّلَ",    # ت prefix + doubled middle
        present_pattern="يَتَفَعَّلُ",
        active_participle="مُتَفَعِّل",
        passive_participle="مُتَفَعَّل",
        verbal_nouns=["تَفَعُّل"],
        description="Reflexive of II, gradual action"
    ),

    VerbForm.VI: FormPattern(
        form=VerbForm.VI,
        name_ar="تَفَاعَلَ",
        name_en="Form VI",
        past_pattern="تَفَاعَلَ",    # ت prefix + long a
        present_pattern="يَتَفَاعَلُ",
        active_participle="مُتَفَاعِل",
        passive_participle="مُتَفَاعَل",
        verbal_nouns=["تَفَاعُل"],
        description="Reciprocal reflexive, pretense"
    ),

    VerbForm.VII: FormPattern(
        form=VerbForm.VII,
        name_ar="اِنْفَعَلَ",
        name_en="Form VII",
        past_pattern="اِنْفَعَلَ",   # ان prefix
        present_pattern="يَنْفَعِلُ",
        active_participle="مُنْفَعِل",
        passive_participle="—",  # No passive
        verbal_nouns=["اِنْفِعَال"],
        description="Passive, reflexive, inchoative"
    ),

    VerbForm.VIII: FormPattern(
        form=VerbForm.VIII,
        name_ar="اِفْتَعَلَ",
        name_en="Form VIII",
        past_pattern="اِفْتَعَلَ",   # ا prefix + ت infix
        present_pattern="يَفْتَعِلُ",
        active_participle="مُفْتَعِل",
        passive_participle="مُفْتَعَل",
        verbal_nouns=["اِفْتِعَال"],
        description="Reflexive, middle voice"
    ),

    VerbForm.IX: FormPattern(
        form=VerbForm.IX,
        name_ar="اِفْعَلَّ",
        name_en="Form IX",
        past_pattern="اِفْعَلَّ",    # Doubled final + sukun on middle
        present_pattern="يَفْعَلُّ",
        active_participle="مُفْعَلّ",
        passive_participle="—",  # Rarely used
        verbal_nouns=["اِفْعِلَال"],
        description="Colors and physical defects (rare)"
    ),

    VerbForm.X: FormPattern(
        form=VerbForm.X,
        name_ar="اِسْتَفْعَلَ",
        name_en="Form X",
        past_pattern="اِسْتَفْعَلَ",  # است prefix
        present_pattern="يَسْتَفْعِلُ",
        active_participle="مُسْتَفْعِل",
        passive_participle="مُسْتَفْعَل",
        verbal_nouns=["اِسْتِفْعَال"],
        description="Requestative, considerative"
    ),
}


# ============================================
# FORM I VOWEL PATTERNS
# ============================================

# Form I has variable vowel patterns in past and present
# The pattern is indicated as (past_middle, present_middle)
# a = fatha, i = kasra, u = damma

class FormIPattern(Enum):
    """Form I vowel patterns (past-present)."""
    FA_AL_A = "a-a"    # فَعَلَ - يَفْعَلُ (e.g., فَتَحَ - يَفْتَحُ)
    FA_AL_I = "a-i"    # فَعَلَ - يَفْعِلُ (e.g., ضَرَبَ - يَضْرِبُ)
    FA_AL_U = "a-u"    # فَعَلَ - يَفْعُلُ (e.g., نَصَرَ - يَنْصُرُ)
    FA_IL_A = "i-a"    # فَعِلَ - يَفْعَلُ (e.g., عَلِمَ - يَعْلَمُ)
    FA_UL_U = "u-u"    # فَعُلَ - يَفْعُلُ (e.g., كَرُمَ - يَكْرُمُ)


# Common Form I patterns for specific roots
FORM_I_PATTERNS = {
    # a-a pattern (فَعَلَ - يَفْعَلُ)
    'فتح': FormIPattern.FA_AL_A,
    'منع': FormIPattern.FA_AL_A,
    'قطع': FormIPattern.FA_AL_A,

    # a-i pattern (فَعَلَ - يَفْعِلُ) - most common
    'ضرب': FormIPattern.FA_AL_I,
    'كتب': FormIPattern.FA_AL_I,
    'جلس': FormIPattern.FA_AL_I,
    'نزل': FormIPattern.FA_AL_I,
    'دخل': FormIPattern.FA_AL_I,
    'خرج': FormIPattern.FA_AL_I,

    # a-u pattern (فَعَلَ - يَفْعُلُ)
    'نصر': FormIPattern.FA_AL_U,
    'كتب': FormIPattern.FA_AL_U,  # Also valid
    'قتل': FormIPattern.FA_AL_U,

    # i-a pattern (فَعِلَ - يَفْعَلُ)
    'علم': FormIPattern.FA_IL_A,
    'فهم': FormIPattern.FA_IL_A,
    'شرب': FormIPattern.FA_IL_A,
    'سمع': FormIPattern.FA_IL_A,

    # u-u pattern (فَعُلَ - يَفْعُلُ) - stative verbs
    'كرم': FormIPattern.FA_UL_U,
    'حسن': FormIPattern.FA_UL_U,
    'كبر': FormIPattern.FA_UL_U,
}


# ============================================
# QUADRILITERAL FORMS
# ============================================

@dataclass
class QuadFormPattern:
    """Pattern for quadriliteral verb forms."""
    form: int  # I, II, III, IV for quadriliterals
    name_ar: str
    past_pattern: str
    present_pattern: str
    active_participle: str
    passive_participle: str
    verbal_noun: str


QUADRILITERAL_FORMS = {
    1: QuadFormPattern(
        form=1,
        name_ar="فَعْلَلَ",
        past_pattern="فَعْلَلَ",
        present_pattern="يُفَعْلِلُ",
        active_participle="مُفَعْلِل",
        passive_participle="مُفَعْلَل",
        verbal_noun="فَعْلَلَة"
    ),
    2: QuadFormPattern(
        form=2,
        name_ar="تَفَعْلَلَ",
        past_pattern="تَفَعْلَلَ",
        present_pattern="يَتَفَعْلَلُ",
        active_participle="مُتَفَعْلِل",
        passive_participle="مُتَفَعْلَل",
        verbal_noun="تَفَعْلُل"
    ),
}


def get_form_info(form: VerbForm) -> FormPattern:
    """Get information about a verb form."""
    return VERB_FORMS[form]


def list_all_forms() -> List[FormPattern]:
    """List all verb form patterns."""
    return list(VERB_FORMS.values())


# ============================================
# TESTING
# ============================================

if __name__ == '__main__':
    print("="*70)
    print("ARABIC VERB FORMS (أوزان الفعل)")
    print("="*70)

    for form, pattern in VERB_FORMS.items():
        print(f"\n{pattern.name_en} ({pattern.name_ar}):")
        print(f"  Past:    {pattern.past_pattern}")
        print(f"  Present: {pattern.present_pattern}")
        print(f"  Active:  {pattern.active_participle}")
        print(f"  Passive: {pattern.passive_participle}")
        print(f"  Masdar:  {', '.join(pattern.verbal_nouns)}")
        print(f"  Meaning: {pattern.description}")
