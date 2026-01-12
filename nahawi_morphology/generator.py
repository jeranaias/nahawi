"""
Arabic Word Generator

Generates Arabic words from roots and patterns.
"""

from typing import Optional
from .data import (
    ADJECTIVE_GENDER_PAIRS, ADJECTIVE_FEM_TO_MASC,
    BROKEN_PLURALS, SOUND_MASC_PLURAL_WORDS, SOUND_FEM_PLURAL_WORDS
)


class WordGenerator:
    """
    Generates Arabic words from roots and patterns.

    Usage:
        gen = WordGenerator()
        word = gen.generate('كتب', 'فاعل', gender='fem', number='plural', definite=True)
        # الكاتبات
    """

    def __init__(self):
        self.adj_masc_to_fem = ADJECTIVE_GENDER_PAIRS
        self.adj_fem_to_masc = ADJECTIVE_FEM_TO_MASC
        self.broken_plurals = BROKEN_PLURALS
        self.sound_masc_plurals = SOUND_MASC_PLURAL_WORDS
        self.sound_fem_plurals = SOUND_FEM_PLURAL_WORDS

    def generate(self, root: str, pattern: str, gender: str = 'masc',
                 number: str = 'singular', definite: bool = False) -> Optional[str]:
        """
        Generate a word from root and pattern.

        Args:
            root: The 3-letter root (e.g., 'كتب')
            pattern: The pattern (e.g., 'فاعل', 'مفعول')
            gender: 'masc' or 'fem'
            number: 'singular', 'dual', or 'plural'
            definite: Whether to add definite article

        Returns:
            The generated word
        """
        if len(root) < 3:
            return None

        r1, r2, r3 = root[0], root[1], root[2]

        # Generate base form from pattern
        base = self._apply_pattern(r1, r2, r3, pattern)

        if not base:
            return None

        # Apply gender
        if gender == 'fem':
            base = self._feminize(base)

        # Apply number
        if number == 'plural':
            base = self._pluralize(base, gender)
        elif number == 'dual':
            base = self._dualize(base, gender)

        # Apply definiteness
        if definite:
            if not base.startswith('ال'):
                base = 'ال' + base

        return base

    def _apply_pattern(self, r1: str, r2: str, r3: str, pattern: str) -> Optional[str]:
        """Apply a morphological pattern to root consonants."""
        # Common patterns
        patterns = {
            # Active participles
            'فاعل': f'{r1}ا{r2}{r3}',
            'مفعل': f'م{r1}{r2}{r3}',
            'مفعّل': f'م{r1}{r2}ّ{r3}',
            'متفعّل': f'مت{r1}{r2}ّ{r3}',
            'مستفعل': f'مست{r1}{r2}{r3}',

            # Passive participles
            'مفعول': f'م{r1}{r2}و{r3}',
            'مفعّل_passive': f'م{r1}{r2}َّ{r3}',

            # Nouns of place/instrument
            'مفعل_place': f'م{r1}{r2}{r3}',
            'مفعلة': f'م{r1}{r2}{r3}ة',

            # Verbal nouns (masdar)
            'فعل': f'{r1}{r2}{r3}',
            'فعال': f'{r1}{r2}ا{r3}',
            'فعالة': f'{r1}{r2}ا{r3}ة',
            'تفعيل': f'ت{r1}{r2}ي{r3}',
            'افتعال': f'ا{r1}ت{r2}ا{r3}',
            'استفعال': f'است{r1}{r2}ا{r3}',

            # Adjective patterns
            'فعيل': f'{r1}{r2}ي{r3}',
            'فعول': f'{r1}{r2}و{r3}',
            'أفعل': f'أ{r1}{r2}{r3}',  # Comparative/superlative

            # Basic forms
            'فعل_verb': f'{r1}{r2}{r3}',
        }

        return patterns.get(pattern, f'{r1}{r2}{r3}')

    def _feminize(self, word: str) -> str:
        """Convert word to feminine form."""
        # Check if in known pairs
        if word in self.adj_masc_to_fem:
            return self.adj_masc_to_fem[word]

        # Standard feminization: add ة
        if not word.endswith('ة') and not word.endswith('اء') and not word.endswith('ى'):
            return word + 'ة'

        return word

    def _masculinize(self, word: str) -> str:
        """Convert word to masculine form."""
        if word in self.adj_fem_to_masc:
            return self.adj_fem_to_masc[word]

        if word.endswith('ة'):
            return word[:-1]

        return word

    def _pluralize(self, word: str, gender: str) -> str:
        """Convert word to plural form."""
        # Check broken plurals
        if word in self.broken_plurals:
            return self.broken_plurals[word]

        # Check sound plurals
        if word in self.sound_masc_plurals:
            return self.sound_masc_plurals[word]
        if word in self.sound_fem_plurals:
            return self.sound_fem_plurals[word]

        # Default sound plural
        if gender == 'fem' or word.endswith('ة'):
            if word.endswith('ة'):
                return word[:-1] + 'ات'
            return word + 'ات'
        else:
            return word + 'ون'

    def _dualize(self, word: str, gender: str) -> str:
        """Convert word to dual form."""
        if word.endswith('ة'):
            return word[:-1] + 'تان'
        return word + 'ان'

    def feminize_adjective(self, adj: str) -> str:
        """Convert an adjective to feminine."""
        # Handle definite article
        prefix = ''
        base = adj
        if adj.startswith('ال'):
            prefix = 'ال'
            base = adj[2:]

        if base in self.adj_masc_to_fem:
            return prefix + self.adj_masc_to_fem[base]

        if not base.endswith('ة'):
            return prefix + base + 'ة'

        return adj

    def masculinize_adjective(self, adj: str) -> str:
        """Convert an adjective to masculine."""
        prefix = ''
        base = adj
        if adj.startswith('ال'):
            prefix = 'ال'
            base = adj[2:]

        if base in self.adj_fem_to_masc:
            return prefix + self.adj_fem_to_masc[base]

        if base.endswith('ة'):
            return prefix + base[:-1]

        return adj

    def pluralize_noun(self, noun: str, gender: str = 'masc') -> str:
        """Pluralize a noun."""
        # Check known plurals
        if noun in self.broken_plurals:
            return self.broken_plurals[noun]
        if noun in self.sound_masc_plurals:
            return self.sound_masc_plurals[noun]
        if noun in self.sound_fem_plurals:
            return self.sound_fem_plurals[noun]

        # Strip definite article
        prefix = ''
        base = noun
        if noun.startswith('ال'):
            prefix = 'ال'
            base = noun[2:]

        # Default pluralization
        if gender == 'fem' or base.endswith('ة'):
            if base.endswith('ة'):
                return prefix + base[:-1] + 'ات'
            return prefix + base + 'ات'
        else:
            return prefix + base + 'ون'

    def singularize_noun(self, noun: str) -> Optional[str]:
        """Get singular form of a noun."""
        # Check reverse mappings
        for sing, plur in self.broken_plurals.items():
            if noun == plur or noun == 'ال' + plur:
                prefix = 'ال' if noun.startswith('ال') else ''
                return prefix + sing

        # Sound plurals
        base = noun[2:] if noun.startswith('ال') else noun
        prefix = 'ال' if noun.startswith('ال') else ''

        if base.endswith('ات'):
            return prefix + base[:-2] + 'ة'
        if base.endswith('ون') or base.endswith('ين'):
            return prefix + base[:-2]

        return None
