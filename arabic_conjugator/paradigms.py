#!/usr/bin/env python3
"""
Arabic Verb Conjugation Paradigms

This module contains the conjugation tables for all persons, numbers, and genders
in past, present, and imperative tenses.

Paradigm structure:
- Person: 1st (متكلم), 2nd (مخاطب), 3rd (غائب)
- Number: singular (مفرد), dual (مثنى), plural (جمع)
- Gender: masculine (مذكر), feminine (مؤنث)
- Tense: past (ماضي), present (مضارع), imperative (أمر)
- Mood (present): indicative (مرفوع), subjunctive (منصوب), jussive (مجزوم)
- Voice: active (معلوم), passive (مجهول)
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
from enum import Enum, auto


class Person(Enum):
    FIRST = 1   # متكلم
    SECOND = 2  # مخاطب
    THIRD = 3   # غائب


class Number(Enum):
    SINGULAR = 1  # مفرد
    DUAL = 2      # مثنى
    PLURAL = 3    # جمع


class Gender(Enum):
    MASCULINE = 1  # مذكر
    FEMININE = 2   # مؤنث
    COMMON = 3     # مشترك (for 1st person)


class Tense(Enum):
    PAST = 1      # ماضي
    PRESENT = 2   # مضارع
    IMPERATIVE = 3  # أمر


class Mood(Enum):
    INDICATIVE = 1   # مرفوع
    SUBJUNCTIVE = 2  # منصوب
    JUSSIVE = 3      # مجزوم


class Voice(Enum):
    ACTIVE = 1   # معلوم
    PASSIVE = 2  # مجهول


@dataclass
class ConjugationSlot:
    """A single conjugation slot in the paradigm."""
    person: Person
    number: Number
    gender: Gender
    prefix: str
    suffix: str
    name_ar: str
    name_en: str


# ============================================
# PAST TENSE PARADIGM (الماضي)
# ============================================

# Past tense suffixes - attached to the stem فَعَلـ
PAST_PARADIGM: Dict[Tuple[Person, Number, Gender], ConjugationSlot] = {
    # Third person (الغائب)
    (Person.THIRD, Number.SINGULAR, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.SINGULAR, Gender.MASCULINE,
        "", "", "هو", "3ms"  # فَعَلَ
    ),
    (Person.THIRD, Number.SINGULAR, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.SINGULAR, Gender.FEMININE,
        "", "ت", "هي", "3fs"  # فَعَلَت
    ),
    (Person.THIRD, Number.DUAL, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.DUAL, Gender.MASCULINE,
        "", "ا", "هما", "3md"  # فَعَلَا
    ),
    (Person.THIRD, Number.DUAL, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.DUAL, Gender.FEMININE,
        "", "تَا", "هما", "3fd"  # فَعَلَتَا
    ),
    (Person.THIRD, Number.PLURAL, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.PLURAL, Gender.MASCULINE,
        "", "وا", "هم", "3mp"  # فَعَلُوا
    ),
    (Person.THIRD, Number.PLURAL, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.PLURAL, Gender.FEMININE,
        "", "نَ", "هن", "3fp"  # فَعَلْنَ
    ),

    # Second person (المخاطب)
    (Person.SECOND, Number.SINGULAR, Gender.MASCULINE): ConjugationSlot(
        Person.SECOND, Number.SINGULAR, Gender.MASCULINE,
        "", "تَ", "أنتَ", "2ms"  # فَعَلْتَ
    ),
    (Person.SECOND, Number.SINGULAR, Gender.FEMININE): ConjugationSlot(
        Person.SECOND, Number.SINGULAR, Gender.FEMININE,
        "", "تِ", "أنتِ", "2fs"  # فَعَلْتِ
    ),
    (Person.SECOND, Number.DUAL, Gender.COMMON): ConjugationSlot(
        Person.SECOND, Number.DUAL, Gender.COMMON,
        "", "تُمَا", "أنتما", "2d"  # فَعَلْتُمَا
    ),
    (Person.SECOND, Number.PLURAL, Gender.MASCULINE): ConjugationSlot(
        Person.SECOND, Number.PLURAL, Gender.MASCULINE,
        "", "تُم", "أنتم", "2mp"  # فَعَلْتُم
    ),
    (Person.SECOND, Number.PLURAL, Gender.FEMININE): ConjugationSlot(
        Person.SECOND, Number.PLURAL, Gender.FEMININE,
        "", "تُنَّ", "أنتن", "2fp"  # فَعَلْتُنَّ
    ),

    # First person (المتكلم)
    (Person.FIRST, Number.SINGULAR, Gender.COMMON): ConjugationSlot(
        Person.FIRST, Number.SINGULAR, Gender.COMMON,
        "", "تُ", "أنا", "1s"  # فَعَلْتُ
    ),
    (Person.FIRST, Number.PLURAL, Gender.COMMON): ConjugationSlot(
        Person.FIRST, Number.PLURAL, Gender.COMMON,
        "", "نَا", "نحن", "1p"  # فَعَلْنَا
    ),
}


# ============================================
# PRESENT TENSE PARADIGM (المضارع)
# ============================================

# Present tense has prefix + stem + suffix
# Prefixes: أ، ن، ت، ي
# Stem: ـفعلـ (varies by mood)
# Suffixes vary by person/number/gender and mood

# INDICATIVE (المرفوع) - default mood
PRESENT_INDICATIVE: Dict[Tuple[Person, Number, Gender], ConjugationSlot] = {
    # Third person
    (Person.THIRD, Number.SINGULAR, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.SINGULAR, Gender.MASCULINE,
        "ي", "", "هو", "3ms"  # يَفْعَلُ
    ),
    (Person.THIRD, Number.SINGULAR, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.SINGULAR, Gender.FEMININE,
        "ت", "", "هي", "3fs"  # تَفْعَلُ
    ),
    (Person.THIRD, Number.DUAL, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.DUAL, Gender.MASCULINE,
        "ي", "ان", "هما", "3md"  # يَفْعَلَان
    ),
    (Person.THIRD, Number.DUAL, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.DUAL, Gender.FEMININE,
        "ت", "ان", "هما", "3fd"  # تَفْعَلَان
    ),
    (Person.THIRD, Number.PLURAL, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.PLURAL, Gender.MASCULINE,
        "ي", "ون", "هم", "3mp"  # يَفْعَلُون
    ),
    (Person.THIRD, Number.PLURAL, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.PLURAL, Gender.FEMININE,
        "ي", "نَ", "هن", "3fp"  # يَفْعَلْنَ
    ),

    # Second person
    (Person.SECOND, Number.SINGULAR, Gender.MASCULINE): ConjugationSlot(
        Person.SECOND, Number.SINGULAR, Gender.MASCULINE,
        "ت", "", "أنتَ", "2ms"  # تَفْعَلُ
    ),
    (Person.SECOND, Number.SINGULAR, Gender.FEMININE): ConjugationSlot(
        Person.SECOND, Number.SINGULAR, Gender.FEMININE,
        "ت", "ين", "أنتِ", "2fs"  # تَفْعَلِين
    ),
    (Person.SECOND, Number.DUAL, Gender.COMMON): ConjugationSlot(
        Person.SECOND, Number.DUAL, Gender.COMMON,
        "ت", "ان", "أنتما", "2d"  # تَفْعَلَان
    ),
    (Person.SECOND, Number.PLURAL, Gender.MASCULINE): ConjugationSlot(
        Person.SECOND, Number.PLURAL, Gender.MASCULINE,
        "ت", "ون", "أنتم", "2mp"  # تَفْعَلُون
    ),
    (Person.SECOND, Number.PLURAL, Gender.FEMININE): ConjugationSlot(
        Person.SECOND, Number.PLURAL, Gender.FEMININE,
        "ت", "نَ", "أنتن", "2fp"  # تَفْعَلْنَ
    ),

    # First person
    (Person.FIRST, Number.SINGULAR, Gender.COMMON): ConjugationSlot(
        Person.FIRST, Number.SINGULAR, Gender.COMMON,
        "أ", "", "أنا", "1s"  # أَفْعَلُ
    ),
    (Person.FIRST, Number.PLURAL, Gender.COMMON): ConjugationSlot(
        Person.FIRST, Number.PLURAL, Gender.COMMON,
        "ن", "", "نحن", "1p"  # نَفْعَلُ
    ),
}

# SUBJUNCTIVE (المنصوب) - after أن، لن، كي، حتى، etc.
# Changes: ون → وا, ان → ا, ين → ي, final damma → fatha
PRESENT_SUBJUNCTIVE: Dict[Tuple[Person, Number, Gender], ConjugationSlot] = {
    # Third person
    (Person.THIRD, Number.SINGULAR, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.SINGULAR, Gender.MASCULINE,
        "ي", "", "هو", "3ms"  # يَفْعَلَ
    ),
    (Person.THIRD, Number.SINGULAR, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.SINGULAR, Gender.FEMININE,
        "ت", "", "هي", "3fs"  # تَفْعَلَ
    ),
    (Person.THIRD, Number.DUAL, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.DUAL, Gender.MASCULINE,
        "ي", "ا", "هما", "3md"  # يَفْعَلَا
    ),
    (Person.THIRD, Number.DUAL, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.DUAL, Gender.FEMININE,
        "ت", "ا", "هما", "3fd"  # تَفْعَلَا
    ),
    (Person.THIRD, Number.PLURAL, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.PLURAL, Gender.MASCULINE,
        "ي", "وا", "هم", "3mp"  # يَفْعَلُوا (not يفعلون!)
    ),
    (Person.THIRD, Number.PLURAL, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.PLURAL, Gender.FEMININE,
        "ي", "نَ", "هن", "3fp"  # يَفْعَلْنَ (same)
    ),

    # Second person
    (Person.SECOND, Number.SINGULAR, Gender.MASCULINE): ConjugationSlot(
        Person.SECOND, Number.SINGULAR, Gender.MASCULINE,
        "ت", "", "أنتَ", "2ms"  # تَفْعَلَ
    ),
    (Person.SECOND, Number.SINGULAR, Gender.FEMININE): ConjugationSlot(
        Person.SECOND, Number.SINGULAR, Gender.FEMININE,
        "ت", "ي", "أنتِ", "2fs"  # تَفْعَلِي
    ),
    (Person.SECOND, Number.DUAL, Gender.COMMON): ConjugationSlot(
        Person.SECOND, Number.DUAL, Gender.COMMON,
        "ت", "ا", "أنتما", "2d"  # تَفْعَلَا
    ),
    (Person.SECOND, Number.PLURAL, Gender.MASCULINE): ConjugationSlot(
        Person.SECOND, Number.PLURAL, Gender.MASCULINE,
        "ت", "وا", "أنتم", "2mp"  # تَفْعَلُوا
    ),
    (Person.SECOND, Number.PLURAL, Gender.FEMININE): ConjugationSlot(
        Person.SECOND, Number.PLURAL, Gender.FEMININE,
        "ت", "نَ", "أنتن", "2fp"  # تَفْعَلْنَ
    ),

    # First person
    (Person.FIRST, Number.SINGULAR, Gender.COMMON): ConjugationSlot(
        Person.FIRST, Number.SINGULAR, Gender.COMMON,
        "أ", "", "أنا", "1s"  # أَفْعَلَ
    ),
    (Person.FIRST, Number.PLURAL, Gender.COMMON): ConjugationSlot(
        Person.FIRST, Number.PLURAL, Gender.COMMON,
        "ن", "", "نحن", "1p"  # نَفْعَلَ
    ),
}

# JUSSIVE (المجزوم) - after لم، لما، لا الناهية، etc.
# Changes: Similar to subjunctive but final vowel is sukun
PRESENT_JUSSIVE: Dict[Tuple[Person, Number, Gender], ConjugationSlot] = {
    # Third person
    (Person.THIRD, Number.SINGULAR, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.SINGULAR, Gender.MASCULINE,
        "ي", "", "هو", "3ms"  # يَفْعَلْ
    ),
    (Person.THIRD, Number.SINGULAR, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.SINGULAR, Gender.FEMININE,
        "ت", "", "هي", "3fs"  # تَفْعَلْ
    ),
    (Person.THIRD, Number.DUAL, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.DUAL, Gender.MASCULINE,
        "ي", "ا", "هما", "3md"  # يَفْعَلَا
    ),
    (Person.THIRD, Number.DUAL, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.DUAL, Gender.FEMININE,
        "ت", "ا", "هما", "3fd"  # تَفْعَلَا
    ),
    (Person.THIRD, Number.PLURAL, Gender.MASCULINE): ConjugationSlot(
        Person.THIRD, Number.PLURAL, Gender.MASCULINE,
        "ي", "وا", "هم", "3mp"  # يَفْعَلُوا
    ),
    (Person.THIRD, Number.PLURAL, Gender.FEMININE): ConjugationSlot(
        Person.THIRD, Number.PLURAL, Gender.FEMININE,
        "ي", "نَ", "هن", "3fp"  # يَفْعَلْنَ
    ),

    # Second person
    (Person.SECOND, Number.SINGULAR, Gender.MASCULINE): ConjugationSlot(
        Person.SECOND, Number.SINGULAR, Gender.MASCULINE,
        "ت", "", "أنتَ", "2ms"  # تَفْعَلْ
    ),
    (Person.SECOND, Number.SINGULAR, Gender.FEMININE): ConjugationSlot(
        Person.SECOND, Number.SINGULAR, Gender.FEMININE,
        "ت", "ي", "أنتِ", "2fs"  # تَفْعَلِي
    ),
    (Person.SECOND, Number.DUAL, Gender.COMMON): ConjugationSlot(
        Person.SECOND, Number.DUAL, Gender.COMMON,
        "ت", "ا", "أنتما", "2d"  # تَفْعَلَا
    ),
    (Person.SECOND, Number.PLURAL, Gender.MASCULINE): ConjugationSlot(
        Person.SECOND, Number.PLURAL, Gender.MASCULINE,
        "ت", "وا", "أنتم", "2mp"  # تَفْعَلُوا
    ),
    (Person.SECOND, Number.PLURAL, Gender.FEMININE): ConjugationSlot(
        Person.SECOND, Number.PLURAL, Gender.FEMININE,
        "ت", "نَ", "أنتن", "2fp"  # تَفْعَلْنَ
    ),

    # First person
    (Person.FIRST, Number.SINGULAR, Gender.COMMON): ConjugationSlot(
        Person.FIRST, Number.SINGULAR, Gender.COMMON,
        "أ", "", "أنا", "1s"  # أَفْعَلْ
    ),
    (Person.FIRST, Number.PLURAL, Gender.COMMON): ConjugationSlot(
        Person.FIRST, Number.PLURAL, Gender.COMMON,
        "ن", "", "نحن", "1p"  # نَفْعَلْ
    ),
}


# ============================================
# IMPERATIVE PARADIGM (الأمر)
# ============================================

# Imperative only has 2nd person
IMPERATIVE_PARADIGM: Dict[Tuple[Number, Gender], ConjugationSlot] = {
    (Number.SINGULAR, Gender.MASCULINE): ConjugationSlot(
        Person.SECOND, Number.SINGULAR, Gender.MASCULINE,
        "", "", "أنتَ", "2ms"  # اِفْعَلْ
    ),
    (Number.SINGULAR, Gender.FEMININE): ConjugationSlot(
        Person.SECOND, Number.SINGULAR, Gender.FEMININE,
        "", "ي", "أنتِ", "2fs"  # اِفْعَلِي
    ),
    (Number.DUAL, Gender.COMMON): ConjugationSlot(
        Person.SECOND, Number.DUAL, Gender.COMMON,
        "", "ا", "أنتما", "2d"  # اِفْعَلَا
    ),
    (Number.PLURAL, Gender.MASCULINE): ConjugationSlot(
        Person.SECOND, Number.PLURAL, Gender.MASCULINE,
        "", "وا", "أنتم", "2mp"  # اِفْعَلُوا
    ),
    (Number.PLURAL, Gender.FEMININE): ConjugationSlot(
        Person.SECOND, Number.PLURAL, Gender.FEMININE,
        "", "نَ", "أنتن", "2fp"  # اِفْعَلْنَ
    ),
}


def get_paradigm(tense: Tense, mood: Optional[Mood] = None):
    """Get the conjugation paradigm for a tense/mood."""
    if tense == Tense.PAST:
        return PAST_PARADIGM
    elif tense == Tense.PRESENT:
        if mood == Mood.SUBJUNCTIVE:
            return PRESENT_SUBJUNCTIVE
        elif mood == Mood.JUSSIVE:
            return PRESENT_JUSSIVE
        else:
            return PRESENT_INDICATIVE
    elif tense == Tense.IMPERATIVE:
        return IMPERATIVE_PARADIGM
    else:
        raise ValueError(f"Unknown tense: {tense}")


# ============================================
# TESTING
# ============================================

if __name__ == '__main__':
    print("="*70)
    print("ARABIC CONJUGATION PARADIGMS")
    print("="*70)

    print("\nPast Tense (الماضي):")
    for key, slot in PAST_PARADIGM.items():
        print(f"  {slot.name_ar:8} ({slot.name_en}): suffix='{slot.suffix}'")

    print("\nPresent Indicative (المضارع المرفوع):")
    for key, slot in PRESENT_INDICATIVE.items():
        print(f"  {slot.name_ar:8} ({slot.name_en}): prefix='{slot.prefix}', suffix='{slot.suffix}'")

    print("\nPresent Subjunctive (المضارع المنصوب):")
    for key, slot in PRESENT_SUBJUNCTIVE.items():
        print(f"  {slot.name_ar:8} ({slot.name_en}): prefix='{slot.prefix}', suffix='{slot.suffix}'")

    print("\nImperative (الأمر):")
    for key, slot in IMPERATIVE_PARADIGM.items():
        print(f"  {slot.name_ar:8} ({slot.name_en}): suffix='{slot.suffix}'")
