#!/usr/bin/env python3
"""
Arabic Root Type Classification

Root Types (أنواع الجذور):
1. صحيح سالم (Sound Regular) - All letters are strong consonants
2. مهموز (Hamzated) - Contains hamza (ء)
   - مهموز الفاء (First radical)
   - مهموز العين (Second radical)
   - مهموز اللام (Third radical)
3. مضعّف (Doubled) - Second and third radicals identical
4. مثال (Assimilated) - First radical is و or ي
5. أجوف (Hollow) - Second radical is و or ي
6. ناقص (Defective) - Third radical is و or ي
7. لفيف مفروق (Doubly Weak - Separated) - First and third are weak
8. لفيف مقرون (Doubly Weak - Adjacent) - Second and third are weak
9. رباعي (Quadriliteral) - Four-letter root
"""

from enum import Enum, auto
from typing import Tuple, List, Optional
from dataclasses import dataclass


class RootType(Enum):
    """Arabic verb root types."""
    # Sound roots
    SOUND = auto()                    # صحيح سالم (e.g., كتب، علم)

    # Hamzated roots
    HAMZATED_FIRST = auto()           # مهموز الفاء (e.g., أخذ، أكل)
    HAMZATED_MIDDLE = auto()          # مهموز العين (e.g., سأل)
    HAMZATED_LAST = auto()            # مهموز اللام (e.g., قرأ)

    # Doubled root
    DOUBLED = auto()                  # مضعّف (e.g., شدّ، مدّ، ردّ)

    # Weak roots (single weak letter)
    ASSIMILATED_WAW = auto()          # مثال واوي (e.g., وجد، وصل)
    ASSIMILATED_YA = auto()           # مثال يائي (e.g., يسر)
    HOLLOW_WAW = auto()               # أجوف واوي (e.g., قول، نوم)
    HOLLOW_YA = auto()                # أجوف يائي (e.g., سير، بيع)
    DEFECTIVE_WAW = auto()            # ناقص واوي (e.g., دعو، غزو)
    DEFECTIVE_YA = auto()             # ناقص يائي (e.g., رمي، بني)

    # Doubly weak roots
    DOUBLE_WEAK_SEPARATED = auto()    # لفيف مفروق (e.g., وقي، وفي)
    DOUBLE_WEAK_ADJACENT = auto()     # لفيف مقرون (e.g., روي، طوي)

    # Quadriliteral
    QUADRILITERAL = auto()            # رباعي (e.g., ترجم، دحرج)
    QUADRILITERAL_DOUBLED = auto()    # رباعي مضعّف (e.g., زلزل)


@dataclass
class RootInfo:
    """Information about a classified root."""
    root: str
    letters: List[str]
    root_type: RootType
    type_name_ar: str
    type_name_en: str
    weak_positions: List[int]  # 0-indexed positions of weak letters
    hamza_positions: List[int]  # 0-indexed positions of hamza


# Weak letters
WEAK_LETTERS = {'و', 'ي', 'ا', 'ى'}
WAW = 'و'
YA = 'ي'
ALIF = 'ا'
ALIF_MAQSURA = 'ى'

# Hamza forms
HAMZA_FORMS = {'ء', 'أ', 'إ', 'ؤ', 'ئ', 'آ'}

# All hamza variations for detection
def is_hamza(char: str) -> bool:
    """Check if character is any form of hamza."""
    return char in HAMZA_FORMS


def is_weak(char: str) -> bool:
    """Check if character is a weak letter (و، ي، ا، ى)."""
    return char in WEAK_LETTERS


def is_waw(char: str) -> bool:
    """Check if character is waw."""
    return char == WAW


def is_ya(char: str) -> bool:
    """Check if character is ya or alif maqsura."""
    return char in {YA, ALIF_MAQSURA}


def normalize_root(root: str) -> str:
    """
    Normalize a root string:
    - Remove dashes, spaces
    - Keep only consonants
    - Normalize hamza forms
    """
    # Remove separators
    root = root.replace('-', '').replace(' ', '').replace('ـ', '')

    # Remove diacritics (tashkeel)
    diacritics = 'ًٌٍَُِّْٰ'
    for d in diacritics:
        root = root.replace(d, '')

    return root


def extract_radicals(root: str) -> List[str]:
    """Extract the radical letters from a root."""
    root = normalize_root(root)
    return list(root)


def classify_root(root: str) -> RootInfo:
    """
    Classify an Arabic verb root by its type.

    Args:
        root: The root string (e.g., "كتب", "ك-ت-ب", "قول")

    Returns:
        RootInfo with classification details
    """
    letters = extract_radicals(root)
    n = len(letters)

    # Track weak and hamza positions
    weak_positions = []
    hamza_positions = []

    for i, letter in enumerate(letters):
        if is_weak(letter):
            weak_positions.append(i)
        if is_hamza(letter):
            hamza_positions.append(i)

    # Quadriliteral roots (4 letters)
    if n == 4:
        if letters[1] == letters[3]:  # Like زلزل
            root_type = RootType.QUADRILITERAL_DOUBLED
            type_ar = "رباعي مضعّف"
            type_en = "Quadriliteral Doubled"
        else:
            root_type = RootType.QUADRILITERAL
            type_ar = "رباعي"
            type_en = "Quadriliteral"

        return RootInfo(
            root=root,
            letters=letters,
            root_type=root_type,
            type_name_ar=type_ar,
            type_name_en=type_en,
            weak_positions=weak_positions,
            hamza_positions=hamza_positions
        )

    # Triliteral roots (3 letters)
    if n != 3:
        # Default to sound if unexpected length
        return RootInfo(
            root=root,
            letters=letters,
            root_type=RootType.SOUND,
            type_name_ar="صحيح",
            type_name_en="Sound",
            weak_positions=weak_positions,
            hamza_positions=hamza_positions
        )

    fa, ain, lam = letters[0], letters[1], letters[2]

    # Check for doubled (second == third)
    if ain == lam and not is_weak(ain):
        return RootInfo(
            root=root,
            letters=letters,
            root_type=RootType.DOUBLED,
            type_name_ar="مضعّف",
            type_name_en="Doubled",
            weak_positions=weak_positions,
            hamza_positions=hamza_positions
        )

    # Count weak letters
    num_weak = len(weak_positions)

    # Doubly weak roots (2 weak letters)
    if num_weak >= 2:
        if 0 in weak_positions and 2 in weak_positions:
            # First and third weak (لفيف مفروق)
            return RootInfo(
                root=root,
                letters=letters,
                root_type=RootType.DOUBLE_WEAK_SEPARATED,
                type_name_ar="لفيف مفروق",
                type_name_en="Doubly Weak (Separated)",
                weak_positions=weak_positions,
                hamza_positions=hamza_positions
            )
        elif 1 in weak_positions and 2 in weak_positions:
            # Second and third weak (لفيف مقرون)
            return RootInfo(
                root=root,
                letters=letters,
                root_type=RootType.DOUBLE_WEAK_ADJACENT,
                type_name_ar="لفيف مقرون",
                type_name_en="Doubly Weak (Adjacent)",
                weak_positions=weak_positions,
                hamza_positions=hamza_positions
            )

    # Single weak letter
    if num_weak == 1:
        weak_pos = weak_positions[0]
        weak_letter = letters[weak_pos]

        if weak_pos == 0:
            # Assimilated (مثال)
            if is_waw(weak_letter):
                return RootInfo(
                    root=root,
                    letters=letters,
                    root_type=RootType.ASSIMILATED_WAW,
                    type_name_ar="مثال واوي",
                    type_name_en="Assimilated (Waw)",
                    weak_positions=weak_positions,
                    hamza_positions=hamza_positions
                )
            else:
                return RootInfo(
                    root=root,
                    letters=letters,
                    root_type=RootType.ASSIMILATED_YA,
                    type_name_ar="مثال يائي",
                    type_name_en="Assimilated (Ya)",
                    weak_positions=weak_positions,
                    hamza_positions=hamza_positions
                )

        elif weak_pos == 1:
            # Hollow (أجوف)
            if is_waw(weak_letter):
                return RootInfo(
                    root=root,
                    letters=letters,
                    root_type=RootType.HOLLOW_WAW,
                    type_name_ar="أجوف واوي",
                    type_name_en="Hollow (Waw)",
                    weak_positions=weak_positions,
                    hamza_positions=hamza_positions
                )
            else:
                return RootInfo(
                    root=root,
                    letters=letters,
                    root_type=RootType.HOLLOW_YA,
                    type_name_ar="أجوف يائي",
                    type_name_en="Hollow (Ya)",
                    weak_positions=weak_positions,
                    hamza_positions=hamza_positions
                )

        else:  # weak_pos == 2
            # Defective (ناقص)
            if is_waw(weak_letter):
                return RootInfo(
                    root=root,
                    letters=letters,
                    root_type=RootType.DEFECTIVE_WAW,
                    type_name_ar="ناقص واوي",
                    type_name_en="Defective (Waw)",
                    weak_positions=weak_positions,
                    hamza_positions=hamza_positions
                )
            else:
                return RootInfo(
                    root=root,
                    letters=letters,
                    root_type=RootType.DEFECTIVE_YA,
                    type_name_ar="ناقص يائي",
                    type_name_en="Defective (Ya)",
                    weak_positions=weak_positions,
                    hamza_positions=hamza_positions
                )

    # Hamzated roots (no weak letters, but has hamza)
    if hamza_positions:
        hamza_pos = hamza_positions[0]
        if hamza_pos == 0:
            return RootInfo(
                root=root,
                letters=letters,
                root_type=RootType.HAMZATED_FIRST,
                type_name_ar="مهموز الفاء",
                type_name_en="Hamzated (First)",
                weak_positions=weak_positions,
                hamza_positions=hamza_positions
            )
        elif hamza_pos == 1:
            return RootInfo(
                root=root,
                letters=letters,
                root_type=RootType.HAMZATED_MIDDLE,
                type_name_ar="مهموز العين",
                type_name_en="Hamzated (Middle)",
                weak_positions=weak_positions,
                hamza_positions=hamza_positions
            )
        else:
            return RootInfo(
                root=root,
                letters=letters,
                root_type=RootType.HAMZATED_LAST,
                type_name_ar="مهموز اللام",
                type_name_en="Hamzated (Last)",
                weak_positions=weak_positions,
                hamza_positions=hamza_positions
            )

    # Default: Sound root
    return RootInfo(
        root=root,
        letters=letters,
        root_type=RootType.SOUND,
        type_name_ar="صحيح سالم",
        type_name_en="Sound",
        weak_positions=weak_positions,
        hamza_positions=hamza_positions
    )


# ============================================
# TESTING
# ============================================

if __name__ == '__main__':
    test_roots = [
        # Sound
        ("كتب", "Sound"),
        ("علم", "Sound"),
        ("فعل", "Sound"),

        # Hamzated
        ("أخذ", "Hamzated First"),
        ("أكل", "Hamzated First"),
        ("سأل", "Hamzated Middle"),
        ("قرأ", "Hamzated Last"),

        # Doubled
        ("شدد", "Doubled"),
        ("مدد", "Doubled"),
        ("ردد", "Doubled"),

        # Assimilated
        ("وجد", "Assimilated Waw"),
        ("وصل", "Assimilated Waw"),
        ("وعد", "Assimilated Waw"),
        ("يسر", "Assimilated Ya"),

        # Hollow
        ("قول", "Hollow Waw"),
        ("نوم", "Hollow Waw"),
        ("صوم", "Hollow Waw"),
        ("سير", "Hollow Ya"),
        ("بيع", "Hollow Ya"),

        # Defective
        ("دعو", "Defective Waw"),
        ("غزو", "Defective Waw"),
        ("رمي", "Defective Ya"),
        ("بني", "Defective Ya"),

        # Doubly weak
        ("وقي", "Doubly Weak Separated"),
        ("روي", "Doubly Weak Adjacent"),
        ("طوي", "Doubly Weak Adjacent"),

        # Quadriliteral
        ("ترجم", "Quadriliteral"),
        ("دحرج", "Quadriliteral"),
        ("زلزل", "Quadriliteral Doubled"),
    ]

    print("="*70)
    print("ARABIC ROOT TYPE CLASSIFIER TEST")
    print("="*70)

    for root, expected in test_roots:
        info = classify_root(root)
        status = "✓" if expected.lower() in info.type_name_en.lower() else "✗"
        print(f"{status} {root:8} → {info.type_name_ar:15} ({info.type_name_en})")
