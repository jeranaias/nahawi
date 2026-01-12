#!/usr/bin/env python3
"""
Rule-based post-processor to boost V5 model's weak categories.
Applies deterministic corrections after neural model output.
"""
import re

class RulePostProcessor:
    """Post-process neural model output with deterministic rules."""

    def __init__(self):
        # Gender agreement: عامة → عام before years
        self.year_pattern = re.compile(r'عامة\s+(\d{4})')

        # Gender agreement: أولى → أول before masculine nouns
        self.first_masc = ['ظهور', 'فيلم', 'كتاب', 'عمل', 'مشروع', 'منتج',
                          'إصدار', 'تسجيل', 'برنامج', 'لقاء', 'اجتماع', 'يوم']

        # Wrong preposition corrections
        self.prep_rules = [
            (r'يبحث في\b', 'يبحث عن'),
            (r'يبحث على\b', 'يبحث عن'),
            (r'بحث في\b', 'بحث عن'),
            (r'بحث على\b', 'بحث عن'),
            (r'يعتمد في\b', 'يعتمد على'),
            (r'يعتمد عن\b', 'يعتمد على'),
            (r'اعتمد في\b', 'اعتمد على'),
            (r'اعتمد عن\b', 'اعتمد على'),
            (r'يفكر عن\b', 'يفكر في'),
            (r'يفكر على\b', 'يفكر في'),
            (r'فكر عن\b', 'فكر في'),
            (r'فكر على\b', 'فكر في'),
            (r'يرغب عن\b', 'يرغب في'),
            (r'يرغب على\b', 'يرغب في'),
            (r'ذهب على\b', 'ذهب إلى'),
            (r'يذهب على\b', 'يذهب إلى'),
            (r'وصل على\b', 'وصل إلى'),
            (r'يصل على\b', 'يصل إلى'),
            (r'تحدث في\s', 'تحدث عن '),
            (r'يتحدث في\s', 'يتحدث عن '),
            (r'سأل في\b', 'سأل عن'),
            (r'يسأل في\b', 'يسأل عن'),
            (r'دافع على\b', 'دافع عن'),
            (r'يدافع على\b', 'يدافع عن'),
            (r'أجاب على\b', 'أجاب عن'),
            (r'يجيب على\b', 'يجيب عن'),
        ]

        # Taa marbuta: common words where ه should be ة
        self.taa_fixes = {
            'المدينه': 'المدينة', 'الجامعه': 'الجامعة', 'المنطقه': 'المنطقة',
            'الدوله': 'الدولة', 'الحكومه': 'الحكومة', 'المدرسه': 'المدرسة',
            'الشركه': 'الشركة', 'المنظمه': 'المنظمة', 'اللغه': 'اللغة',
            'الثقافه': 'الثقافة', 'السياسه': 'السياسة', 'الحياه': 'الحياة',
            'القوه': 'القوة', 'الفتره': 'الفترة', 'المره': 'المرة',
            'السنه': 'السنة', 'الساعه': 'الساعة', 'الصوره': 'الصورة',
            'الفكره': 'الفكرة', 'الطريقه': 'الطريقة', 'المرحله': 'المرحلة',
            'الحاله': 'الحالة', 'النتيجه': 'النتيجة', 'المشكله': 'المشكلة',
            'الخطه': 'الخطة', 'القصه': 'القصة', 'الرحله': 'الرحلة',
            'البدايه': 'البداية', 'النهايه': 'النهاية', 'الإجابه': 'الإجابة',
            'العلاقه': 'العلاقة', 'المعركه': 'المعركة', 'الثوره': 'الثورة',
            'الحضاره': 'الحضارة', 'القاره': 'القارة', 'الجزيره': 'الجزيرة',
            'مدينه': 'مدينة', 'منطقه': 'منطقة', 'دوله': 'دولة',
            'جامعه': 'جامعة', 'مدرسه': 'مدرسة', 'شركه': 'شركة',
            'قريه': 'قرية', 'جزيره': 'جزيرة', 'كلمه': 'كلمة',
        }

        # Alif maqsura: ONLY prepositions - these are never names
        # Removed علي because it's also the name "Ali"
        self.maqsura_fixes = {
            'إلي': 'إلى', 'حتي': 'حتى', 'لدي': 'لدى', 'متي': 'متى',
        }

        # Context-aware maqsura: only fix علي when followed by specific words
        self.ali_contexts = ['الرغم', 'سبيل', 'أي', 'حد', 'كل', 'أساس', 'الأقل', 'الأكثر', 'الفور', 'مدار', 'مستوى', 'صعيد']

    def process(self, text):
        """Apply all rules to text."""
        result = text

        # 1. Gender: عامة + year → عام + year
        result = self.year_pattern.sub(r'عام \1', result)

        # 2. Gender: أولى + masc noun → أول + masc noun
        for noun in self.first_masc:
            result = result.replace(f'أولى {noun}', f'أول {noun}')

        # 3. Wrong prepositions
        for pattern, replacement in self.prep_rules:
            result = re.sub(pattern, replacement, result)

        # 4. Taa marbuta
        for wrong, correct in self.taa_fixes.items():
            result = result.replace(wrong, correct)

        # 5. Alif maqsura (careful: only at word boundaries)
        words = result.split()
        for i, word in enumerate(words):
            # Strip punctuation for matching
            clean = word.rstrip('،.؟!؛:')
            punct = word[len(clean):]
            if clean in self.maqsura_fixes:
                words[i] = self.maqsura_fixes[clean] + punct
            # Context-aware: علي → على only before specific words
            elif clean == 'علي' and i + 1 < len(words):
                next_word = words[i + 1].rstrip('،.؟!؛:')
                if next_word in self.ali_contexts:
                    words[i] = 'على' + punct
        result = ' '.join(words)

        return result


def test_rules():
    """Test the rule post-processor."""
    pp = RulePostProcessor()

    tests = [
        # Gender agreement
        ("في عامة 1975", "في عام 1975"),
        ("حوالي عامة 2010", "حوالي عام 2010"),
        ("أولى ظهور له", "أول ظهور له"),
        ("أولى فيلم", "أول فيلم"),

        # Wrong preposition
        ("يبحث في المعلومات", "يبحث عن المعلومات"),
        ("يعتمد في ذلك", "يعتمد على ذلك"),
        ("يفكر عن الموضوع", "يفكر في الموضوع"),
        ("ذهب على المدرسة", "ذهب إلى المدرسة"),

        # Taa marbuta
        ("المدينه الكبيرة", "المدينة الكبيرة"),
        ("الجامعه العريقة", "الجامعة العريقة"),

        # Alif maqsura
        ("ذهب إلي المدرسة", "ذهب إلى المدرسة"),
        ("علي الرغم", "على الرغم"),
        ("حتي الآن", "حتى الآن"),
    ]

    print("Testing rule post-processor:")
    print("="*60)
    passed = 0
    for input_text, expected in tests:
        output = pp.process(input_text)
        status = "✓" if output == expected else "✗"
        if output == expected:
            passed += 1
        print(f"{status} '{input_text}'")
        print(f"  → '{output}'")
        if output != expected:
            print(f"  Expected: '{expected}'")
        print()

    print(f"Passed: {passed}/{len(tests)}")
    return passed == len(tests)


if __name__ == '__main__':
    test_rules()
