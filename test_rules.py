#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Quick standalone test of rule-based Arabic GEC.
No external dependencies required.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import re
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Correction:
    original: str
    corrected: str
    start_idx: int
    end_idx: int
    error_type: str
    confidence: float
    model_name: str


class TaaMarbutaFixer:
    """Fixes ة (taa marbuta) ↔ ه (ha) confusion."""

    def __init__(self):
        self.name = "TaaMarbutaFixer"
        # Common words where ه should be ة
        self.taa_words = {
            'مدرسه': 'مدرسة', 'جامعه': 'جامعة', 'مدينه': 'مدينة',
            'حكومه': 'حكومة', 'دوله': 'دولة', 'لغه': 'لغة',
            'حياه': 'حياة', 'قصه': 'قصة', 'صوره': 'صورة',
            'فكره': 'فكرة', 'مره': 'مرة', 'سنه': 'سنة',
            'جميله': 'جميلة', 'كبيره': 'كبيرة', 'صغيره': 'صغيرة',
            'طويله': 'طويلة', 'قصيره': 'قصيرة', 'جديده': 'جديدة',
            'قديمه': 'قديمة', 'سعيده': 'سعيدة', 'حزينه': 'حزينة',
            'المدرسه': 'المدرسة', 'الجامعه': 'الجامعة', 'المدينه': 'المدينة',
            'الكبيره': 'الكبيرة', 'الجميله': 'الجميلة', 'الصغيره': 'الصغيرة',
        }

    def correct(self, text: str) -> Tuple[str, List[Correction]]:
        corrections = []
        words = text.split()
        new_words = []
        char_pos = 0

        for word in words:
            new_word = word

            # Check dictionary first
            if word in self.taa_words:
                new_word = self.taa_words[word]
            # Pattern: word ending in ه that should be ة (feminine pattern)
            elif word.endswith('ه') and len(word) > 2:
                # Common feminine adjective patterns ending in يه -> ية
                if word.endswith('يه'):
                    new_word = word[:-1] + 'ة'

            if new_word != word:
                corrections.append(Correction(
                    original=word, corrected=new_word,
                    start_idx=char_pos, end_idx=char_pos + len(word),
                    error_type="taa_marbuta", confidence=0.98,
                    model_name=self.name
                ))

            new_words.append(new_word)
            char_pos += len(word) + 1

        return ' '.join(new_words), corrections


class AlifMaksuraFixer:
    """Fixes ى (alif maksura) ↔ ي (ya) confusion."""

    def __init__(self):
        self.name = "AlifMaksuraFixer"
        self.maksura_words = {
            'علي': 'على', 'الي': 'إلى', 'حتي': 'حتى',
            'متي': 'متى', 'مستشفي': 'مستشفى', 'اخري': 'أخرى',
        }

    def correct(self, text: str) -> Tuple[str, List[Correction]]:
        corrections = []
        words = text.split()
        new_words = []
        char_pos = 0

        for word in words:
            clean = word.strip('،.؟!؛:')
            suffix = word[len(clean):] if len(word) > len(clean) else ''

            if clean in self.maksura_words:
                new_word = self.maksura_words[clean] + suffix
                corrections.append(Correction(
                    original=word, corrected=new_word,
                    start_idx=char_pos, end_idx=char_pos + len(word),
                    error_type="alif_maksura", confidence=0.97,
                    model_name=self.name
                ))
                new_words.append(new_word)
            else:
                new_words.append(word)
            char_pos += len(word) + 1

        return ' '.join(new_words), corrections


class PunctuationFixer:
    """Fixes Arabic punctuation."""

    def __init__(self):
        self.name = "PunctuationFixer"
        self.replacements = [(',', '،'), (';', '؛'), ('?', '؟')]

    def correct(self, text: str) -> Tuple[str, List[Correction]]:
        corrections = []
        corrected = text

        for eng, arabic in self.replacements:
            pos = 0
            while True:
                idx = corrected.find(eng, pos)
                if idx == -1:
                    break
                context = corrected[max(0, idx-5):min(len(corrected), idx+5)]
                has_arabic = any('\u0600' <= c <= '\u06FF' for c in context)
                if has_arabic:
                    corrections.append(Correction(
                        original=eng, corrected=arabic,
                        start_idx=idx, end_idx=idx + 1,
                        error_type="punctuation", confidence=0.99,
                        model_name=self.name
                    ))
                    corrected = corrected[:idx] + arabic + corrected[idx+1:]
                pos = idx + 1

        return corrected, corrections


class RepeatedWordFixer:
    """Fixes repeated words."""

    def __init__(self):
        self.name = "RepeatedWordFixer"
        self.allow_repeat = {'لا', 'نعم', 'جدا'}

    def correct(self, text: str) -> Tuple[str, List[Correction]]:
        corrections = []
        words = text.split()

        if len(words) < 2:
            return text, corrections

        new_words = [words[0]]
        char_pos = len(words[0]) + 1

        for i in range(1, len(words)):
            prev = words[i-1].strip('،.؟!؛:')
            curr = words[i].strip('،.؟!؛:')

            if prev == curr and curr not in self.allow_repeat:
                corrections.append(Correction(
                    original=words[i], corrected="",
                    start_idx=char_pos, end_idx=char_pos + len(words[i]),
                    error_type="repeated_word", confidence=0.99,
                    model_name=self.name
                ))
            else:
                new_words.append(words[i])
            char_pos += len(words[i]) + 1

        return ' '.join(new_words), corrections


def test_all():
    print("=" * 60)
    print("NAHAWI RULE-BASED MODEL TESTS")
    print("=" * 60)

    # Test TaaMarbutaFixer
    print("\n[TaaMarbutaFixer]")
    fixer = TaaMarbutaFixer()
    tests = [
        ("هذه مدرسه جميله", "هذه مدرسة جميلة"),
        ("الجامعه الكبيره", "الجامعة الكبيرة"),
    ]
    for inp, exp in tests:
        out, corr = fixer.correct(inp)
        status = "PASS" if out == exp else "FAIL"
        print(f"  [{status}] '{inp}' -> '{out}'")
        if corr:
            print(f"         Corrections: {len(corr)}")

    # Test AlifMaksuraFixer
    print("\n[AlifMaksuraFixer]")
    fixer = AlifMaksuraFixer()
    tests = [
        ("ذهب الي المدرسه", "ذهب إلى المدرسه"),
        ("متي سافرت", "متى سافرت"),
    ]
    for inp, exp in tests:
        out, corr = fixer.correct(inp)
        status = "PASS" if out == exp else "FAIL"
        print(f"  [{status}] '{inp}' -> '{out}'")

    # Test PunctuationFixer
    print("\n[PunctuationFixer]")
    fixer = PunctuationFixer()
    tests = [
        ("مرحبا, كيف حالك?", "مرحبا، كيف حالك؟"),
    ]
    for inp, exp in tests:
        out, corr = fixer.correct(inp)
        status = "PASS" if out == exp else "FAIL"
        print(f"  [{status}] '{inp}' -> '{out}'")

    # Test RepeatedWordFixer
    print("\n[RepeatedWordFixer]")
    fixer = RepeatedWordFixer()
    tests = [
        ("ذهبت الى الى المدرسة", "ذهبت الى المدرسة"),
        ("لا لا اريد", "لا لا اريد"),  # Allowed
    ]
    for inp, exp in tests:
        out, corr = fixer.correct(inp)
        status = "PASS" if out == exp else "FAIL"
        print(f"  [{status}] '{inp}' -> '{out}'")

    print("\n" + "=" * 60)
    print("TESTS COMPLETE")
    print("=" * 60)


if __name__ == '__main__':
    test_all()
