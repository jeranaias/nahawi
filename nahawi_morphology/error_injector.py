"""
Arabic Error Injector

Injects realistic Arabic errors for training data generation.
Based on actual error patterns observed in QALB and the FASIH benchmark.

Error categories:
1. Gender agreement errors (verb/adjective with noun)
2. Number agreement errors (plural markers)
3. Truncation errors (missing final letters)
4. Verb form errors (ون/وا confusion, feminine markers)
5. Preposition errors
6. Definiteness errors
"""

import random
from typing import Optional, Tuple, List
from .data import (
    ADJECTIVE_GENDER_PAIRS, ADJECTIVE_FEM_TO_MASC,
    VERB_PREPOSITIONS, IRREGULAR_VERBS, VOCAB_CACHE,
    BROKEN_PLURALS, SOUND_FEM_PLURAL_WORDS
)


class ErrorInjector:
    """
    Injects realistic Arabic errors for training data generation.

    Usage:
        injector = ErrorInjector(analyzer, conjugator)
        error_word = injector.inject("ذهبت", "gender")  # ذهب
        error_word, error_type = injector.inject_random("كانت")  # (كان, 'gender')
    """

    def __init__(self, analyzer, conjugator):
        self.analyzer = analyzer
        self.conjugator = conjugator
        self.adj_masc_to_fem = ADJECTIVE_GENDER_PAIRS
        self.adj_fem_to_masc = ADJECTIVE_FEM_TO_MASC

        # Error injection methods
        self.error_methods = {
            'gender': self._inject_gender_error,
            'number': self._inject_number_error,
            'truncation': self._inject_truncation_error,
            'verb_form': self._inject_verb_form_error,
            'preposition': self._inject_preposition_error,
            'definiteness': self._inject_definiteness_error,
            'spelling': self._inject_spelling_error,
        }

        # Common prepositions and their confusions
        self.prep_confusions = {
            'إلى': ['في', 'على', 'ل', 'ب'],
            'على': ['في', 'إلى', 'ب'],
            'في': ['على', 'ب', 'إلى'],
            'من': ['عن', 'ب'],
            'عن': ['من', 'على', 'إلى'],
            'ب': ['في', 'ل', 'على'],
            'ل': ['إلى', 'ب'],
            'مع': ['ب', 'ل'],
        }

        # Letter confusion patterns
        self.letter_confusions = {
            'ذ': 'د', 'د': 'ذ',
            'ض': 'ظ', 'ظ': 'ض',
            'ث': 'ت', 'ت': 'ث',
            'ص': 'س', 'س': 'ص',
            'ط': 'ت', 'ت': 'ط',
            'ة': 'ه', 'ه': 'ة',
            'أ': 'ا', 'إ': 'ا', 'آ': 'ا',
            'ى': 'ي', 'ي': 'ى',
        }

    def inject(self, word: str, error_type: str) -> Optional[str]:
        """
        Inject a specific type of error.

        Args:
            word: The correct word
            error_type: Type of error to inject

        Returns:
            The word with injected error, or None if not applicable
        """
        if error_type not in self.error_methods:
            return None

        result = self.error_methods[error_type](word)
        # Ensure we actually changed something
        if result and result != word:
            return result
        return None

    def inject_random(self, word: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Inject a random applicable error.

        Returns:
            (error_word, error_type) or (None, None) if no error applicable
        """
        # Try each error type in random order
        error_types = list(self.error_methods.keys())
        random.shuffle(error_types)

        for error_type in error_types:
            result = self.inject(word, error_type)
            if result and result != word:
                return result, error_type

        return None, None

    def _inject_gender_error(self, word: str) -> Optional[str]:
        """Inject a gender agreement error."""
        info = self.analyzer.analyze(word)

        # For adjectives: swap gender
        if word in self.adj_masc_to_fem:
            return self.adj_masc_to_fem[word]
        if word in self.adj_fem_to_masc:
            return self.adj_fem_to_masc[word]

        # For verbs: change gender marker
        if info.pos == 'verb':
            if info.tense == 'past':
                if word.endswith('ت') and len(word) > 2:
                    # Remove feminine marker: ذهبت → ذهب
                    return word[:-1]
                elif not word.endswith('ت') and not word.endswith('وا') and not word.endswith('ن'):
                    # Add feminine marker: ذهب → ذهبت
                    return word + 'ت'
            elif info.tense == 'present':
                if word.startswith('ت') and len(word) > 2:
                    # Change prefix: تذهب → يذهب
                    return 'ي' + word[1:]
                elif word.startswith('ي') and len(word) > 2:
                    # Change prefix: يذهب → تذهب
                    return 'ت' + word[1:]

        # For nouns ending in ة: remove it
        if word.endswith('ة') and len(word) > 2:
            return word[:-1]

        # For nouns not ending in ة: add it (less common)
        if not word.endswith('ة') and not word.endswith(('ا', 'و', 'ي', 'ى', 'ون', 'ين', 'ات')):
            if random.random() < 0.3:
                return word + 'ة'

        return None

    def _inject_number_error(self, word: str) -> Optional[str]:
        """Inject a number agreement error."""
        info = self.analyzer.analyze(word)

        # Present tense plural: ون → وا (very common error)
        if word.endswith('ون') and len(word) > 3:
            return word[:-2] + 'وا'

        # Present tense: وا → ون (reverse)
        if word.endswith('وا') and word[0] in 'يتنأ' and len(word) > 3:
            return word[:-2] + 'ون'

        # Past plural: وا → و (missing alif)
        if word.endswith('وا') and word[0] not in 'يتنأ' and len(word) > 3:
            return word[:-1]

        # Sound feminine plural: ات → ا (truncation)
        if word.endswith('ات') and len(word) > 4:
            return word[:-2] + 'ا'

        # Remove plural marker entirely
        if word.endswith('ون'):
            return word[:-2]
        if word.endswith('ين'):
            return word[:-2]

        return None

    def _inject_truncation_error(self, word: str) -> Optional[str]:
        """Inject a truncation error (missing final letters)."""
        # This is one of the most common error types in the benchmark

        # Remove final ت
        if word.endswith('ت') and len(word) > 2:
            return word[:-1]

        # Remove final ن
        if word.endswith('ن') and len(word) > 2:
            return word[:-1]

        # Remove final ة (becomes ـه looking)
        if word.endswith('ة') and len(word) > 2:
            return word[:-1] + 'ه'  # Common confusion

        # Remove last letter generally
        if len(word) > 3:
            return word[:-1]

        return None

    def _inject_verb_form_error(self, word: str) -> Optional[str]:
        """Inject verb form errors."""
        info = self.analyzer.analyze(word)

        if info.pos != 'verb':
            return None

        # كانت → كان (most common in benchmark!)
        if word in VOCAB_CACHE and VOCAB_CACHE[word].get('person') == '3fs':
            # Try to find masculine equivalent
            for v, c in VOCAB_CACHE.items():
                if (c.get('root') == VOCAB_CACHE[word].get('root') and
                    c.get('tense') == VOCAB_CACHE[word].get('tense') and
                    c.get('person') == '3ms'):
                    return v

        # Generic past tense: remove ت
        if info.tense == 'past' and word.endswith('ت'):
            return word[:-1]

        # Generic present tense: ون → وا
        if info.tense == 'present' and word.endswith('ون'):
            return word[:-2] + 'وا'

        return None

    def _inject_preposition_error(self, word: str) -> Optional[str]:
        """Inject a preposition error."""
        if word in self.prep_confusions:
            wrong_preps = self.prep_confusions[word]
            return random.choice(wrong_preps)
        return None

    def _inject_definiteness_error(self, word: str) -> Optional[str]:
        """Inject a definiteness error."""
        if word.startswith('ال') and len(word) > 3:
            # Remove definite article
            return word[2:]

        if not word.startswith('ال') and len(word) > 2:
            # Add definite article (incorrectly)
            if word[0] not in 'وفبكل' and not word.startswith(('من', 'عن', 'إلى')):
                return 'ال' + word

        return None

    def _inject_spelling_error(self, word: str) -> Optional[str]:
        """Inject a spelling/orthography error."""
        # Letter confusions
        for correct, wrong in self.letter_confusions.items():
            if correct in word:
                return word.replace(correct, wrong, 1)

        return None

    def inject_sentence_errors(self, sentence: str, num_errors: int = 2,
                               error_types: List[str] = None) -> Tuple[str, List[dict]]:
        """
        Inject multiple errors into a sentence.

        Args:
            sentence: The correct sentence
            num_errors: Number of errors to inject
            error_types: List of allowed error types (None = all)

        Returns:
            (corrupted_sentence, list of error details)
        """
        words = sentence.split()
        errors_injected = []
        available_types = error_types or list(self.error_methods.keys())

        # Shuffle word indices
        indices = list(range(len(words)))
        random.shuffle(indices)

        errors_added = 0
        for idx in indices:
            if errors_added >= num_errors:
                break

            word = words[idx]
            random.shuffle(available_types)

            for error_type in available_types:
                result = self.inject(word, error_type)
                if result and result != word:
                    words[idx] = result
                    errors_injected.append({
                        'index': idx,
                        'original': word,
                        'error': result,
                        'type': error_type
                    })
                    errors_added += 1
                    break

        return ' '.join(words), errors_injected

    def get_applicable_errors(self, word: str) -> List[str]:
        """Get list of error types applicable to a word."""
        applicable = []
        for error_type in self.error_methods:
            result = self.inject(word, error_type)
            if result and result != word:
                applicable.append(error_type)
        return applicable
