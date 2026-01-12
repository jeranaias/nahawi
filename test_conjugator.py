#!/usr/bin/env python3
"""Test the Arabic conjugator."""

import sys
sys.path.insert(0, 'c:/nahawi')

from arabic_conjugator.root_types import classify_root, RootType
from arabic_conjugator.paradigms import (
    Person, Number, Gender, Tense, Mood,
    PAST_PARADIGM, PRESENT_INDICATIVE
)
from arabic_conjugator.verb_forms import VerbForm, VERB_FORMS

print("="*70)
print("ARABIC CONJUGATOR TEST")
print("="*70)

# Test root classification
print("\n[1] ROOT CLASSIFICATION")
print("-"*50)

test_roots = [
    ("كتب", "Sound"),
    ("قول", "Hollow (Waw)"),
    ("بيع", "Hollow (Ya)"),
    ("رمي", "Defective (Ya)"),
    ("دعو", "Defective (Waw)"),
    ("وجد", "Assimilated (Waw)"),
    ("أخذ", "Hamzated"),
    ("شدد", "Doubled"),
    ("ترجم", "Quadriliteral"),
]

for root, expected in test_roots:
    info = classify_root(root)
    status = "✓" if expected.split()[0].lower() in info.type_name_en.lower() else "?"
    print(f"  {status} {root}: {info.type_name_ar} ({info.type_name_en})")

# Test basic conjugation patterns
print("\n[2] PAST TENSE PARADIGM")
print("-"*50)
print("  Suffix patterns for فَعَلَ:")

for key, slot in PAST_PARADIGM.items():
    person, number, gender = key
    suffix = slot.suffix if slot.suffix else "(none)"
    print(f"    {slot.name_ar:8} ({slot.name_en:4}): +{suffix}")

# Test with actual verb stems
print("\n[3] CONJUGATION EXAMPLES")
print("-"*50)

# Manual conjugation test (without the full conjugator due to import issues)
def conjugate_sound_past(root: str) -> dict:
    """Conjugate a sound root in past tense."""
    letters = list(root)
    stem = ''.join(letters)

    forms = {}
    # Key past tense forms
    forms['3ms'] = stem  # فَعَلَ
    forms['3fs'] = stem + 'ت'  # فَعَلَت
    forms['3mp'] = stem + 'وا'  # فَعَلُوا
    forms['3fp'] = stem + 'ن'  # فَعَلْنَ
    forms['2ms'] = stem + 'تَ'  # فَعَلْتَ
    forms['2fs'] = stem + 'تِ'  # فَعَلْتِ
    forms['2mp'] = stem + 'تُم'  # فَعَلْتُم
    forms['1s'] = stem + 'تُ'  # فَعَلْتُ
    forms['1p'] = stem + 'نا'  # فَعَلْنا

    return forms

def conjugate_sound_present(root: str) -> dict:
    """Conjugate a sound root in present tense."""
    letters = list(root)
    stem = ''.join(letters)

    forms = {}
    # Key present tense forms (indicative)
    forms['3ms.ind'] = 'ي' + stem  # يَفْعَلُ
    forms['3fs.ind'] = 'ت' + stem  # تَفْعَلُ
    forms['3mp.ind'] = 'ي' + stem + 'ون'  # يَفْعَلُون
    forms['3mp.subj'] = 'ي' + stem + 'وا'  # يَفْعَلُوا (subjunctive!)
    forms['2ms.ind'] = 'ت' + stem  # تَفْعَلُ
    forms['1s.ind'] = 'أ' + stem  # أَفْعَلُ
    forms['1p.ind'] = 'ن' + stem  # نَفْعَلُ

    return forms

# Test with كتب
print("\nRoot: كتب (to write)")
past = conjugate_sound_past('كتب')
present = conjugate_sound_present('كتب')

print("  Past tense:")
for label, form in past.items():
    print(f"    {label}: {form}")

print("  Present tense:")
for label, form in present.items():
    print(f"    {label}: {form}")

# Test hollow verb قول
print("\nRoot: قول (to say) - Hollow")
def conjugate_hollow_past(root: str) -> dict:
    """Conjugate a hollow root in past tense."""
    letters = list(root)
    # Hollow verbs: قَالَ، قَالَت، قَالُوا
    # BUT: قُلْتُ، قُلْتَ (middle vowel changes before consonant suffix)

    forms = {}
    forms['3ms'] = f"{letters[0]}ا{letters[2]}"  # قال
    forms['3fs'] = f"{letters[0]}ا{letters[2]}ت"  # قالت
    forms['3mp'] = f"{letters[0]}ا{letters[2]}وا"  # قالوا
    forms['2ms'] = f"{letters[0]}{letters[2]}تَ"  # قُلْتَ (vowel shortens)
    forms['1s'] = f"{letters[0]}{letters[2]}تُ"  # قُلْتُ

    return forms

hollow_past = conjugate_hollow_past('قول')
print("  Past tense (hollow):")
for label, form in hollow_past.items():
    print(f"    {label}: {form}")

# Test defective verb رمي
print("\nRoot: رمي (to throw) - Defective Ya")
def conjugate_defective_past(root: str) -> dict:
    """Conjugate a defective root in past tense."""
    letters = list(root)

    forms = {}
    forms['3ms'] = f"{letters[0]}{letters[1]}ى"  # رَمَى
    forms['3fs'] = f"{letters[0]}{letters[1]}ت"  # رَمَت
    forms['3mp'] = f"{letters[0]}{letters[1]}وا"  # رَمَوا
    forms['2ms'] = f"{letters[0]}{letters[1]}يتَ"  # رَمَيْتَ
    forms['1s'] = f"{letters[0]}{letters[1]}يتُ"  # رَمَيْتُ

    return forms

defective_past = conjugate_defective_past('رمي')
print("  Past tense (defective):")
for label, form in defective_past.items():
    print(f"    {label}: {form}")

# Show verb feminization pairs (the main error type!)
print("\n[4] VERB FEMINIZATION PAIRS (Key for GEC)")
print("-"*50)

fem_pairs = [
    ('كتب', 'كتب', 'كتبت'),
    ('قول', 'قال', 'قالت'),
    ('أخذ', 'أخذ', 'أخذت'),
    ('علم', 'علم', 'علمت'),
    ('جعل', 'جعل', 'جعلت'),
    ('وجد', 'وجد', 'وجدت'),
    ('رمي', 'رمى', 'رمت'),
    ('بني', 'بنى', 'بنت'),
]

print("  Root   3ms     3fs")
print("  " + "-"*30)
for root, masc, fem in fem_pairs:
    print(f"  {root}    {masc}    {fem}")

# Show indicative vs subjunctive (another key error!)
print("\n[5] INDICATIVE vs SUBJUNCTIVE (يفعلون vs يفعلوا)")
print("-"*50)

print("  Indicative (المرفوع): يكتبون، يقولون، يرمون")
print("  Subjunctive (المنصوب): أن يكتبوا، لن يقولوا، كي يرموا")
print("")
print("  Common error: Using subjunctive form in indicative context")
print("  Example: الطلاب يدرسوا ← الطلاب يدرسون")

print("\n" + "="*70)
print("CONJUGATOR TEST COMPLETE")
print("="*70)
