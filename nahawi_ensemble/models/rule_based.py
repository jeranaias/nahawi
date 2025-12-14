"""
Rule-based models for high-precision, fast corrections.

These handle specific error types with linguistic rules:
- TaaMarbutaFixer: ة ↔ ه
- AlifMaksuraFixer: ى ↔ ي
- PunctuationFixer: Arabic punctuation
- RepeatedWordFixer: Duplicate words
"""

import re
from typing import List, Tuple, Dict, Set
from .base import RuleBasedModel, Correction, CorrectionResult


class TaaMarbutaFixer(RuleBasedModel):
    """
    Fixes ة (taa marbuta) ↔ ه (ha) confusion.

    Rules:
    - ه at end of word after Arabic letter → likely should be ة
    - ة at end of word before non-Arabic → keep as is
    - Common word patterns that should end in ة
    """

    def __init__(self):
        super().__init__("TaaMarbutaFixer", ["taa_marbuta"])

        # Words that MUST end in ة (feminine markers, common nouns)
        self.taa_words = {
            # Nouns
            'مدرسه': 'مدرسة', 'جامعه': 'جامعة', 'مدينه': 'مدينة',
            'حكومه': 'حكومة', 'دوله': 'دولة', 'لغه': 'لغة',
            'حياه': 'حياة', 'قصه': 'قصة', 'صوره': 'صورة',
            'فكره': 'فكرة', 'مره': 'مرة', 'سنه': 'سنة',
            'ساعه': 'ساعة', 'دقيقه': 'دقيقة', 'ثانيه': 'ثانية',
            'كلمه': 'كلمة', 'جمله': 'جملة', 'فقره': 'فقرة',
            'صفحه': 'صفحة', 'رساله': 'رسالة', 'مقاله': 'مقالة',
            'قراءه': 'قراءة', 'كتابه': 'كتابة', 'طريقه': 'طريقة',
            'نتيجه': 'نتيجة', 'مشكله': 'مشكلة', 'حاله': 'حالة',
            'علاقه': 'علاقة', 'منطقه': 'منطقة', 'ثقافه': 'ثقافة',
            'سياسه': 'سياسة', 'تجربه': 'تجربة', 'خبره': 'خبرة',
            'شركه': 'شركة', 'مؤسسه': 'مؤسسة', 'منظمه': 'منظمة',
            'هيئه': 'هيئة', 'لجنه': 'لجنة', 'وزاره': 'وزارة',
            'اداره': 'إدارة', 'رئاسه': 'رئاسة', 'قياده': 'قيادة',
            'زياده': 'زيادة', 'عوده': 'عودة', 'بدايه': 'بداية',
            'نهايه': 'نهاية', 'محاوله': 'محاولة', 'مساعده': 'مساعدة',
            'مشاركه': 'مشاركة', 'متابعه': 'متابعة', 'مراجعه': 'مراجعة',
            'دراسه': 'دراسة',
            # Adjectives (feminine)
            'جميله': 'جميلة', 'كبيره': 'كبيرة', 'صغيره': 'صغيرة',
            'طويله': 'طويلة', 'قصيره': 'قصيرة', 'جديده': 'جديدة',
            'قديمه': 'قديمة', 'سعيده': 'سعيدة', 'حزينه': 'حزينة',
            'اقتصاديه': 'اقتصادية', 'اجتماعيه': 'اجتماعية',
            'علميه': 'علمية', 'عمليه': 'عملية',
            # With al- prefix
            'المدرسه': 'المدرسة', 'الجامعه': 'الجامعة', 'المدينه': 'المدينة',
            'الكبيره': 'الكبيرة', 'الجميله': 'الجميلة', 'الصغيره': 'الصغيرة',
        }

        # Feminine adjective endings
        self.feminine_patterns = [
            (r'(\w+)يه\b', r'\1ية'),  # عربيه → عربية
            (r'(\w+)ويه\b', r'\1وية'),  # قويه → قوية
        ]

    def apply_rules(self, text: str) -> Tuple[str, List[Correction]]:
        corrections = []
        corrected = text

        # Dictionary lookup
        words = text.split()
        new_words = []
        char_pos = 0

        for word in words:
            if word in self.taa_words:
                new_word = self.taa_words[word]
                corrections.append(Correction(
                    original=word,
                    corrected=new_word,
                    start_idx=char_pos,
                    end_idx=char_pos + len(word),
                    error_type="taa_marbuta",
                    confidence=0.98,
                    model_name=self.name
                ))
                new_words.append(new_word)
            else:
                new_words.append(word)
            char_pos += len(word) + 1

        corrected = ' '.join(new_words)

        # Pattern-based corrections
        for pattern, replacement in self.feminine_patterns:
            matches = list(re.finditer(pattern, corrected))
            for match in reversed(matches):  # Reverse to not mess up indices
                original = match.group(0)
                new_text = re.sub(pattern, replacement, original)
                if original != new_text:
                    corrections.append(Correction(
                        original=original,
                        corrected=new_text,
                        start_idx=match.start(),
                        end_idx=match.end(),
                        error_type="taa_marbuta",
                        confidence=0.95,
                        model_name=self.name
                    ))
                    corrected = corrected[:match.start()] + new_text + corrected[match.end():]

        return corrected, corrections


class AlifMaksuraFixer(RuleBasedModel):
    """
    Fixes ى (alif maksura) ↔ ي (ya) confusion.

    Rules:
    - Final ى in verbs: past tense 3rd person (مشى، رمى، سعى)
    - Final ى in nouns: certain patterns (هدى، فتى، مستشفى)
    - Final ي in nouns: nisba adjectives (مصري، عربي)
    """

    def __init__(self):
        super().__init__("AlifMaksuraFixer", ["alif_maksura"])

        # Words that MUST end in ى
        self.maksura_words = {
            'علي': 'على',  # preposition
            'الي': 'إلى',
            'حتي': 'حتى',
            'متي': 'متى',
            'انثي': 'أنثى',
            'مستشفي': 'مستشفى',
            'مصطفي': 'مصطفى',
            'موسي': 'موسى',
            'عيسي': 'عيسى',
            'يحيي': 'يحيى',
            'هدي': 'هدى',
            'سلوي': 'سلوى',
            'نجوي': 'نجوى',
            'فتوي': 'فتوى',
            'دعوي': 'دعوى',
            'شوري': 'شورى',
            'كبري': 'كبرى',
            'صغري': 'صغرى',
            'اخري': 'أخرى',
            'احدي': 'إحدى',
            'الذي': 'الذي',  # Keep as is (not maksura)
        }

        # Past tense verbs that end in ى
        self.verb_patterns = [
            r'\bمشي\b',  # مشى
            r'\bرمي\b',  # رمى
            r'\bسعي\b',  # سعى
            r'\bبني\b',  # بنى
            r'\bجري\b',  # جرى
            r'\bقضي\b',  # قضى (but قضي can be passive)
        ]

    def apply_rules(self, text: str) -> Tuple[str, List[Correction]]:
        corrections = []
        words = text.split()
        new_words = []
        char_pos = 0

        for word in words:
            clean_word = word.strip('،.؟!؛:')
            suffix = word[len(clean_word):] if len(word) > len(clean_word) else ''

            if clean_word in self.maksura_words:
                new_word = self.maksura_words[clean_word] + suffix
                corrections.append(Correction(
                    original=word,
                    corrected=new_word,
                    start_idx=char_pos,
                    end_idx=char_pos + len(word),
                    error_type="alif_maksura",
                    confidence=0.97,
                    model_name=self.name
                ))
                new_words.append(new_word)
            else:
                new_words.append(word)

            char_pos += len(word) + 1

        corrected = ' '.join(new_words)
        return corrected, corrections


class PunctuationFixer(RuleBasedModel):
    """
    Fixes Arabic punctuation issues.

    Rules:
    - English comma (,) → Arabic comma (،)
    - English semicolon (;) → Arabic semicolon (؛)
    - English question mark (?) → Arabic question mark (؟)
    - Fix spacing around punctuation
    """

    def __init__(self):
        super().__init__("PunctuationFixer", ["punctuation"])

        self.replacements = [
            (',', '،'),   # comma
            (';', '؛'),   # semicolon
            ('?', '؟'),   # question mark
        ]

    def apply_rules(self, text: str) -> Tuple[str, List[Correction]]:
        corrections = []
        corrected = text

        for eng, arabic in self.replacements:
            pos = 0
            while True:
                idx = corrected.find(eng, pos)
                if idx == -1:
                    break

                # Check if it's in an Arabic context
                # (has Arabic letters nearby)
                context_start = max(0, idx - 5)
                context_end = min(len(corrected), idx + 5)
                context = corrected[context_start:context_end]

                has_arabic = any('\u0600' <= c <= '\u06FF' for c in context)

                if has_arabic:
                    corrections.append(Correction(
                        original=eng,
                        corrected=arabic,
                        start_idx=idx,
                        end_idx=idx + 1,
                        error_type="punctuation",
                        confidence=0.99,
                        model_name=self.name
                    ))
                    corrected = corrected[:idx] + arabic + corrected[idx+1:]

                pos = idx + 1

        # Fix spacing: no space before punctuation, space after
        # ، should have space after, not before
        corrected = re.sub(r'\s+([،؛؟!.])', r'\1', corrected)
        corrected = re.sub(r'([،؛؟])(\S)', r'\1 \2', corrected)

        return corrected, corrections


class RepeatedWordFixer(RuleBasedModel):
    """
    Fixes repeated/duplicate words.

    Rules:
    - Same word appearing twice consecutively → remove one
    - Common patterns like "في في" → "في"
    """

    def __init__(self):
        super().__init__("RepeatedWordFixer", ["repeated_word"])

        # Words that CAN be repeated legitimately
        self.allow_repeat = {
            'لا',      # لا لا (for emphasis)
            'نعم',     # نعم نعم
            'جدا',     # جدا جدا (very very)
            'كثيرا',   # كثيرا كثيرا
        }

    def apply_rules(self, text: str) -> Tuple[str, List[Correction]]:
        corrections = []
        words = text.split()

        if len(words) < 2:
            return text, corrections

        new_words = [words[0]]
        char_pos = len(words[0]) + 1

        for i in range(1, len(words)):
            prev_word = words[i-1].strip('،.؟!؛:')
            curr_word = words[i].strip('،.؟!؛:')

            if prev_word == curr_word and curr_word not in self.allow_repeat:
                # Duplicate detected
                corrections.append(Correction(
                    original=words[i],
                    corrected="",
                    start_idx=char_pos,
                    end_idx=char_pos + len(words[i]),
                    error_type="repeated_word",
                    confidence=0.99,
                    model_name=self.name
                ))
                # Don't add this word
            else:
                new_words.append(words[i])

            char_pos += len(words[i]) + 1

        corrected = ' '.join(new_words)
        return corrected, corrections


# Factory function
def get_rule_based_model(model_type: str) -> RuleBasedModel:
    """Get a rule-based model by type."""
    models = {
        'taa_marbuta': TaaMarbutaFixer,
        'alif_maksura': AlifMaksuraFixer,
        'punctuation': PunctuationFixer,
        'repeated_word': RepeatedWordFixer,
    }

    if model_type not in models:
        raise ValueError(f"Unknown rule-based model: {model_type}")

    return models[model_type]()
