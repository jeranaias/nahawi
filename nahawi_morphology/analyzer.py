"""
Arabic Word Analyzer

Analyzes Arabic words to extract morphological information:
- Root extraction
- Gender detection
- Number detection
- Part of speech
- Definiteness
"""

from dataclasses import dataclass
from typing import Optional, Dict, List
from .data import (
    ROOTS, VOCAB_CACHE, INHERENT_FEMININE, MASCULINE_EXCEPTIONS,
    ADJECTIVE_GENDER_PAIRS, ADJECTIVE_FEM_TO_MASC,
    BROKEN_PLURALS, PLURAL_TO_SINGULAR,
    SOUND_MASC_PLURAL_WORDS, SOUND_FEM_PLURAL_WORDS,
    IRREGULAR_VERBS
)


@dataclass
class MorphInfo:
    """Morphological information about an Arabic word."""
    word: str
    root: Optional[str] = None
    pattern: Optional[str] = None
    pos: str = 'unknown'  # noun, verb, adj, particle
    gender: str = 'unknown'  # masc, fem, common
    number: str = 'unknown'  # sing, dual, plural
    person: Optional[str] = None  # 1s, 2ms, 3fs, etc.
    tense: Optional[str] = None  # past, present, imperative
    definite: bool = False
    case: Optional[str] = None  # nom, acc, gen


class ArabicAnalyzer:
    """
    Analyzes Arabic words for morphological features.

    Usage:
        analyzer = ArabicAnalyzer()
        info = analyzer.analyze("الطالبات")
        print(info.gender, info.number)  # fem, plural
    """

    def __init__(self):
        self.vocab_cache = VOCAB_CACHE
        self.roots = ROOTS
        self.inherent_fem = INHERENT_FEMININE
        self.masc_exceptions = MASCULINE_EXCEPTIONS
        self.adj_pairs = ADJECTIVE_GENDER_PAIRS
        self.adj_fem_to_masc = ADJECTIVE_FEM_TO_MASC
        self.broken_plurals = BROKEN_PLURALS
        self.plural_to_sing = PLURAL_TO_SINGULAR
        self.irregular_verbs = IRREGULAR_VERBS

    def analyze(self, word: str) -> MorphInfo:
        """Analyze a word and return morphological information."""
        if not word:
            return MorphInfo(word='')

        # Strip common prefixes for analysis
        base_word, definite, prefix = self._strip_prefixes(word)

        # Check vocabulary cache first
        if base_word in self.vocab_cache:
            cached = self.vocab_cache[base_word]
            return MorphInfo(
                word=word,
                root=cached.get('root'),
                pos=cached.get('pos', 'unknown'),
                gender=cached.get('gender', 'unknown'),
                number=cached.get('number', 'unknown'),
                person=cached.get('person'),
                tense=cached.get('tense'),
                definite=definite or word.startswith('ال')
            )

        # Analyze from scratch
        info = MorphInfo(word=word, definite=definite or word.startswith('ال'))

        # Detect gender
        info.gender = self._detect_gender(base_word)

        # Detect number
        info.number = self._detect_number(base_word)

        # Detect POS and other features
        info.pos = self._detect_pos(base_word)

        # Try to extract root
        info.root = self._extract_root(base_word)

        # Verb-specific analysis
        if info.pos == 'verb':
            verb_info = self._analyze_verb(base_word)
            info.tense = verb_info.get('tense')
            info.person = verb_info.get('person')

        return info

    def _strip_prefixes(self, word: str) -> tuple:
        """Strip common prefixes and return (base, is_definite, prefix)."""
        definite = False
        prefix = ''

        # Definite article
        if word.startswith('ال') and len(word) > 2:
            definite = True
            prefix = 'ال'
            word = word[2:]
            # Handle sun letters (assimilation)
            # ال + ش = الش (shams letters)

        # Conjunctions and prepositions attached
        elif word.startswith('وال') and len(word) > 3:
            definite = True
            prefix = 'وال'
            word = word[3:]
        elif word.startswith('بال') and len(word) > 3:
            definite = True
            prefix = 'بال'
            word = word[3:]
        elif word.startswith('كال') and len(word) > 3:
            definite = True
            prefix = 'كال'
            word = word[3:]
        elif word.startswith('لل') and len(word) > 2:
            definite = True
            prefix = 'لل'
            word = word[2:]

        return word, definite, prefix

    def _detect_gender(self, word: str) -> str:
        """Detect the gender of a word."""
        # Check exceptions first
        if word in self.masc_exceptions:
            return 'masc'

        # Check inherently feminine
        if word in self.inherent_fem:
            return 'fem'

        # Check adjective pairs
        if word in self.adj_pairs:
            return 'masc'
        if word in self.adj_fem_to_masc:
            return 'fem'

        # Taa marbuta ending - usually feminine
        if word.endswith('ة'):
            return 'fem'

        # Alif mamduda ending (اء) - often feminine
        if word.endswith('اء'):
            # But not all - فقراء، علماء are plural
            if word in self.plural_to_sing:
                return self._detect_gender(self.plural_to_sing[word])
            return 'fem'

        # Alif maqsura ending (ى) - can be feminine
        if word.endswith('ى') and len(word) > 2:
            # Often feminine in adjectives (كبرى، صغرى)
            return 'fem'

        # Sound feminine plural - feminine
        if word.endswith('ات'):
            return 'fem'

        # Sound masculine plural endings - masculine
        if word.endswith('ون') or word.endswith('ين'):
            return 'masc'

        # Default to masculine
        return 'masc'

    def _detect_number(self, word: str) -> str:
        """Detect the number (singular/dual/plural) of a word."""
        # Check plural mappings
        if word in self.plural_to_sing:
            return 'plural'

        # Sound masculine plural
        if word.endswith('ون') or word.endswith('ين'):
            return 'plural'

        # Sound feminine plural
        if word.endswith('ات'):
            return 'plural'

        # Dual endings
        if word.endswith('ان') or word.endswith('ين'):
            # Could be dual or sound masculine plural
            # Need context - assume plural for ين
            if word.endswith('ان') and len(word) > 3:
                return 'dual'

        # Broken plural patterns (check common ones)
        # Pattern: أَفْعَال (أولاد، أقلام)
        if word.startswith('أ') and len(word) >= 4:
            if word in self.broken_plurals.values():
                return 'plural'

        # Check if in broken plurals values
        for singular, plural in self.broken_plurals.items():
            if word == plural:
                return 'plural'

        return 'sing'

    def _detect_pos(self, word: str) -> str:
        """Detect part of speech."""
        # Check verb patterns
        if self._is_verb(word):
            return 'verb'

        # Check if in adjective pairs
        if word in self.adj_pairs or word in self.adj_fem_to_masc:
            return 'adj'

        # Particles
        particles = {'في', 'من', 'إلى', 'على', 'عن', 'مع', 'ب', 'ل', 'ك',
                    'و', 'أو', 'لكن', 'إن', 'أن', 'أنّ', 'إنّ',
                    'هذا', 'هذه', 'ذلك', 'تلك', 'الذي', 'التي',
                    'هو', 'هي', 'هم', 'هن', 'أنت', 'أنتِ', 'أنا', 'نحن'}
        if word in particles:
            return 'particle'

        # Default to noun
        return 'noun'

    def _is_verb(self, word: str) -> bool:
        """Check if a word is likely a verb."""
        # Present tense markers
        if len(word) >= 3:
            if word[0] in 'يتنأ' and word not in {'يوم', 'يد', 'أم', 'أب', 'أخ'}:
                # Likely present tense verb
                # But exclude common nouns
                if word.endswith('ون') or word.endswith('ين') or word.endswith('ان'):
                    return True
                if len(word) >= 4:
                    return True

        # Past tense - check if in vocab cache or irregular verbs
        if word in self.vocab_cache:
            return self.vocab_cache[word].get('pos') == 'verb'

        # Check irregular verbs
        for root, forms in self.irregular_verbs.items():
            for tense_forms in forms.values():
                if word in tense_forms.values():
                    return True

        # Past tense pattern: ends with ت/ا/وا for conjugation
        if word.endswith('وا') or word.endswith('تا'):
            return True

        return False

    def _analyze_verb(self, word: str) -> Dict:
        """Analyze verb-specific features."""
        result = {}

        # Check irregular verbs first
        for root, forms in self.irregular_verbs.items():
            for tense, persons in forms.items():
                for person, form in persons.items():
                    if word == form:
                        return {'tense': tense, 'person': person, 'root': root}

        # Check vocab cache
        if word in self.vocab_cache:
            cached = self.vocab_cache[word]
            if cached.get('pos') == 'verb':
                return {
                    'tense': cached.get('tense'),
                    'person': cached.get('person'),
                    'root': cached.get('root')
                }

        # Analyze from morphology
        if word[0] in 'يتنأ':
            result['tense'] = 'present'
            if word[0] == 'ي':
                if word.endswith('ون'):
                    result['person'] = '3mp'
                elif word.endswith('ن'):
                    result['person'] = '3fp'
                elif word.endswith('ان'):
                    result['person'] = '3md'
                else:
                    result['person'] = '3ms'
            elif word[0] == 'ت':
                if word.endswith('ون'):
                    result['person'] = '2mp'
                elif word.endswith('ين'):
                    result['person'] = '2fs'
                elif word.endswith('ان'):
                    result['person'] = '2md'
                else:
                    # Could be 3fs or 2ms
                    result['person'] = '3fs'  # Default
            elif word[0] == 'أ':
                result['person'] = '1s'
            elif word[0] == 'ن':
                result['person'] = '1p'
        else:
            result['tense'] = 'past'
            if word.endswith('وا'):
                result['person'] = '3mp'
            elif word.endswith('ن'):
                result['person'] = '3fp'
            elif word.endswith('تا'):
                result['person'] = '3fd'
            elif word.endswith('ا') and not word.endswith('وا'):
                result['person'] = '3md'
            elif word.endswith('ت'):
                # Could be 3fs, 2ms, 2fs, 1s
                result['person'] = '3fs'  # Default
            elif word.endswith('تم'):
                result['person'] = '2mp'
            elif word.endswith('نا'):
                result['person'] = '1p'
            else:
                result['person'] = '3ms'

        return result

    def _extract_root(self, word: str) -> Optional[str]:
        """Attempt to extract the root from a word."""
        # Check vocab cache
        if word in self.vocab_cache:
            return self.vocab_cache[word].get('root')

        # Check broken plurals
        if word in self.plural_to_sing:
            singular = self.plural_to_sing[word]
            if singular in self.vocab_cache:
                return self.vocab_cache[singular].get('root')

        # Simple heuristic for trilateral roots
        # Remove common affixes
        base = word

        # Remove prefixes
        for prefix in ['است', 'ان', 'ا', 'ت', 'م', 'ي', 'ن']:
            if base.startswith(prefix) and len(base) > len(prefix) + 2:
                base = base[len(prefix):]
                break

        # Remove suffixes
        for suffix in ['ون', 'ين', 'ات', 'ة', 'وا', 'ان', 'تا', 'نا', 'تم', 'ت']:
            if base.endswith(suffix) and len(base) > len(suffix) + 2:
                base = base[:-len(suffix)]
                break

        # If we have exactly 3 letters, that might be the root
        if len(base) == 3:
            # Check if it's a known root
            if base in self.roots:
                return base

        return None

    def get_gender(self, word: str) -> str:
        """Quick gender detection."""
        base, _, _ = self._strip_prefixes(word)
        return self._detect_gender(base)

    def get_number(self, word: str) -> str:
        """Quick number detection."""
        base, _, _ = self._strip_prefixes(word)
        return self._detect_number(base)

    def is_feminine(self, word: str) -> bool:
        """Check if word is feminine."""
        return self.get_gender(word) == 'fem'

    def is_plural(self, word: str) -> bool:
        """Check if word is plural."""
        return self.get_number(word) == 'plural'

    def is_verb(self, word: str) -> bool:
        """Check if word is a verb."""
        base, _, _ = self._strip_prefixes(word)
        return self._is_verb(base)
