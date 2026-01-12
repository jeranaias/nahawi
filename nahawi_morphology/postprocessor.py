"""
Arabic GEC Post-Processor using Morphology

Uses the Nahawi morphology tool to fix agreement errors
that the neural model might have missed or introduced.

Key fixes:
1. Noun-adjective gender agreement
2. Subject-verb gender agreement
3. Noun-adjective definiteness agreement
4. Number agreement corrections
"""

from typing import List, Tuple, Optional
from .analyzer import ArabicAnalyzer, MorphInfo
from .agreement import AgreementChecker
from .data import (
    ADJECTIVE_GENDER_PAIRS, ADJECTIVE_FEM_TO_MASC,
    IRREGULAR_VERBS, VOCAB_CACHE
)


class MorphologyPostProcessor:
    """
    Post-processor that uses morphological analysis to fix
    agreement errors in GEC model output.

    Usage:
        pp = MorphologyPostProcessor()
        fixed = pp.process("الطالبة المجتهد نجح في الامتحان")
        # Returns: "الطالبة المجتهدة نجحت في الامتحان"
    """

    def __init__(self, conservative: bool = True):
        """
        Initialize post-processor.

        Args:
            conservative: If True, only fix high-confidence errors
        """
        self.analyzer = ArabicAnalyzer()
        self.agreement_checker = AgreementChecker(self.analyzer)
        self.conservative = conservative

        # Build quick lookup tables
        self.adj_masc_to_fem = ADJECTIVE_GENDER_PAIRS
        self.adj_fem_to_masc = ADJECTIVE_FEM_TO_MASC
        self.irregular_verbs = IRREGULAR_VERBS

        # Common nouns that are often followed by adjectives
        self.common_nouns = {
            'الطالب', 'الطالبة', 'المدرسة', 'الجامعة', 'المدينة',
            'الشركة', 'الحكومة', 'الدولة', 'اللغة', 'المرأة',
            'البنت', 'الولد', 'الرجل', 'المعلم', 'المعلمة',
            'الكتاب', 'القصة', 'الرسالة', 'السيارة', 'الطائرة',
        }

        # Feminine subject markers (for verb agreement)
        self.fem_subjects = {
            'الشركة', 'الحكومة', 'الدولة', 'المدرسة', 'الجامعة',
            'اللجنة', 'الوزارة', 'المنظمة', 'الهيئة', 'المؤسسة',
            'الطالبة', 'المعلمة', 'المرأة', 'البنت', 'الأم',
        }

        # Verbs that need gender agreement with previous noun
        self.agreement_verbs = {'كان', 'كانت', 'أصبح', 'أصبحت', 'صار', 'صارت'}

    def process(self, text: str) -> str:
        """
        Process text and fix agreement errors.

        Args:
            text: Input text (possibly from neural GEC model)

        Returns:
            Text with agreement errors fixed
        """
        words = text.split()
        if len(words) < 2:
            return text

        result = list(words)  # Make a copy

        # Pass 1: Fix noun-adjective agreement
        result = self._fix_noun_adj_agreement(result)

        # Pass 2: Fix subject-verb agreement
        result = self._fix_subject_verb_agreement(result)

        # Pass 3: Fix definiteness agreement
        result = self._fix_definiteness_agreement(result)

        return ' '.join(result)

    def _fix_noun_adj_agreement(self, words: List[str]) -> List[str]:
        """Fix noun-adjective gender agreement."""
        result = list(words)

        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]

            info1 = self.analyzer.analyze(word1)
            info2 = self.analyzer.analyze(word2)

            # Check if word1 is noun and word2 could be adjective
            if info1.pos not in ('noun', 'adj'):
                continue

            # Check if word2 is likely an adjective
            if not self._is_likely_adjective(word2, info2):
                continue

            # Check gender agreement
            if info1.gender != info2.gender:
                if info1.gender == 'fem' and info2.gender == 'masc':
                    # Need to feminize the adjective
                    fixed = self._feminize_word(word2)
                    if fixed and fixed != word2:
                        result[i + 1] = fixed
                elif info1.gender == 'masc' and info2.gender == 'fem':
                    # Need to masculinize the adjective (less common fix)
                    if not self.conservative:
                        fixed = self._masculinize_word(word2)
                        if fixed and fixed != word2:
                            result[i + 1] = fixed

        return result

    def _fix_subject_verb_agreement(self, words: List[str]) -> List[str]:
        """Fix subject-verb gender agreement."""
        result = list(words)

        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]

            info1 = self.analyzer.analyze(word1)
            info2 = self.analyzer.analyze(word2)

            # Check for noun + verb pattern
            if info1.pos == 'noun' and info2.pos == 'verb':
                # Check gender agreement
                if info1.gender == 'fem' and info2.gender == 'masc':
                    # Need feminine verb
                    fixed = self._feminize_verb(word2, info2)
                    if fixed and fixed != word2:
                        result[i + 1] = fixed

            # Also check for common agreement verb patterns
            # e.g., الشركة كان → الشركة كانت
            if word1 in self.fem_subjects and word2 in {'كان', 'أصبح', 'صار'}:
                # These should be feminine
                fem_verbs = {'كان': 'كانت', 'أصبح': 'أصبحت', 'صار': 'صارت'}
                if word2 in fem_verbs:
                    result[i + 1] = fem_verbs[word2]

        return result

    def _fix_definiteness_agreement(self, words: List[str]) -> List[str]:
        """Fix noun-adjective definiteness agreement."""
        result = list(words)

        for i in range(len(words) - 1):
            word1 = words[i]
            word2 = words[i + 1]

            info1 = self.analyzer.analyze(word1)
            info2 = self.analyzer.analyze(word2)

            # Check if word1 is definite noun and word2 is indefinite adjective
            if info1.pos in ('noun', 'adj') and self._is_likely_adjective(word2, info2):
                if info1.definite and not info2.definite:
                    # Add definite article to adjective
                    if not word2.startswith('ال'):
                        result[i + 1] = 'ال' + word2
                elif not info1.definite and info2.definite:
                    # Remove definite article from adjective (less common)
                    if not self.conservative and word2.startswith('ال'):
                        result[i + 1] = word2[2:]

        return result

    def _is_likely_adjective(self, word: str, info: MorphInfo) -> bool:
        """Check if a word is likely an adjective."""
        # Direct POS check
        if info.pos == 'adj':
            return True

        # Strip definite article for lookup
        base = word[2:] if word.startswith('ال') else word

        # Check known adjective pairs
        if base in self.adj_masc_to_fem or base in self.adj_fem_to_masc:
            return True

        # Check with definite article
        if word in self.adj_masc_to_fem or word in self.adj_fem_to_masc:
            return True

        # Heuristic: definite words following nouns that aren't verbs/prepositions
        if word.startswith('ال') and info.pos not in ('verb', 'prep', 'conj'):
            # Not ending in plural markers that indicate nouns
            if not word.endswith(('ون', 'ين', 'ات')):
                return True

        return False

    def _feminize_word(self, word: str) -> Optional[str]:
        """Convert word to feminine form."""
        # Strip definite article
        base = word[2:] if word.startswith('ال') else word
        prefix = 'ال' if word.startswith('ال') else ''

        # Check known pairs
        if base in self.adj_masc_to_fem:
            fem = self.adj_masc_to_fem[base]
            return prefix + fem if not fem.startswith('ال') else fem

        # Generic: add ة if not already there
        if not base.endswith('ة') and not base.endswith(('ي', 'ا', 'و', 'ى')):
            return prefix + base + 'ة'

        return None

    def _masculinize_word(self, word: str) -> Optional[str]:
        """Convert word to masculine form."""
        base = word[2:] if word.startswith('ال') else word
        prefix = 'ال' if word.startswith('ال') else ''

        # Check known pairs
        if base in self.adj_fem_to_masc:
            masc = self.adj_fem_to_masc[base]
            return prefix + masc if not masc.startswith('ال') else masc

        # Generic: remove ة
        if base.endswith('ة'):
            return prefix + base[:-1]

        return None

    def _feminize_verb(self, verb: str, info: MorphInfo) -> Optional[str]:
        """Convert verb to feminine form."""
        # Check irregular verbs
        for root, forms in self.irregular_verbs.items():
            for tense, persons in forms.items():
                for person, form in persons.items():
                    if verb == form and 'm' in person:
                        # Find feminine equivalent
                        fem_person = person.replace('m', 'f')
                        if fem_person in persons:
                            return persons[fem_person]

        # Past tense: add ت
        if info.tense == 'past':
            if not verb.endswith('ت') and not verb.endswith('وا') and not verb.endswith('ن'):
                return verb + 'ت'

        # Present tense: ي→ت prefix
        if info.tense == 'present':
            if verb.startswith('ي') and not verb.endswith('ن'):
                return 'ت' + verb[1:]

        return None

    def get_fixes(self, text: str) -> List[dict]:
        """
        Get list of fixes that would be applied.

        Returns:
            List of dicts with 'original', 'fixed', 'type' keys
        """
        words = text.split()
        fixes = []

        processed = self.process(text).split()

        for i, (orig, fixed) in enumerate(zip(words, processed)):
            if orig != fixed:
                fixes.append({
                    'index': i,
                    'original': orig,
                    'fixed': fixed,
                    'type': self._infer_fix_type(orig, fixed)
                })

        return fixes

    def _infer_fix_type(self, orig: str, fixed: str) -> str:
        """Infer the type of fix applied."""
        if fixed == orig + 'ة':
            return 'gender_fem'
        if orig == fixed + 'ة':
            return 'gender_masc'
        if fixed == orig + 'ت':
            return 'verb_fem'
        if orig == fixed + 'ت':
            return 'verb_masc'
        if fixed == 'ال' + orig:
            return 'definiteness_add'
        if orig == 'ال' + fixed:
            return 'definiteness_remove'
        return 'other'


def create_postprocessor(conservative: bool = True) -> MorphologyPostProcessor:
    """Factory function to create a post-processor."""
    return MorphologyPostProcessor(conservative=conservative)
