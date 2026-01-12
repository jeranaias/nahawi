"""
Nahawi Arabic Morphology Analyzer

Elite-grade Arabic morphological analysis tool for:
- Word analysis (root, pattern, gender, number, POS)
- Agreement checking (subject-verb, noun-adjective)
- Word generation from roots and patterns
- Verb conjugation
- Error injection for training data generation
"""

from .analyzer import ArabicAnalyzer, MorphInfo
from .agreement import AgreementChecker, AgreementError
from .conjugator import VerbConjugator
from .generator import WordGenerator
from .error_injector import ErrorInjector
from .postprocessor import MorphologyPostProcessor, create_postprocessor

__version__ = "1.0.0"
__all__ = [
    'ArabicMorphology',
    'MorphInfo',
    'AgreementError',
    'MorphologyPostProcessor',
    'create_postprocessor',
]


class ArabicMorphology:
    """
    Main interface for Arabic morphological analysis.

    Usage:
        morph = ArabicMorphology()

        # Analyze a word
        info = morph.analyze("الطالبات")
        print(info.gender, info.number)  # feminine, plural

        # Check agreement
        errors = morph.check_agreement("الطالبة", "المجتهد")

        # Generate word
        word = morph.generate(root='كتب', pattern='فاعل', gender='fem')

        # Conjugate verb
        verb = morph.conjugate(root='ذهب', tense='past', person='3fs')

        # Inject error
        error_word = morph.inject_error("ذهبت", "gender")
    """

    def __init__(self):
        self.analyzer = ArabicAnalyzer()
        self.agreement_checker = AgreementChecker(self.analyzer)
        self.conjugator = VerbConjugator()
        self.generator = WordGenerator()
        self.error_injector = ErrorInjector(self.analyzer, self.conjugator)
        self.postprocessor = MorphologyPostProcessor(conservative=True)

    def analyze(self, word: str) -> MorphInfo:
        """Analyze a word and return morphological information."""
        return self.analyzer.analyze(word)

    def check_agreement(self, word1: str, word2: str) -> list:
        """Check agreement between two words."""
        return self.agreement_checker.check(word1, word2)

    def fix_agreement(self, sentence: str) -> str:
        """Fix agreement errors in a sentence."""
        return self.agreement_checker.fix_sentence(sentence)

    def conjugate(self, root: str, tense: str = 'past', person: str = '3ms',
                  form: int = 1) -> str:
        """Conjugate a verb from its root."""
        return self.conjugator.conjugate(root, tense, person, form)

    def generate(self, root: str, pattern: str, gender: str = 'masc',
                 number: str = 'singular', definite: bool = False) -> str:
        """Generate a word from root and pattern."""
        return self.generator.generate(root, pattern, gender, number, definite)

    def inject_error(self, word: str, error_type: str) -> str:
        """Inject a specific type of error into a word."""
        return self.error_injector.inject(word, error_type)

    def inject_random_error(self, word: str) -> tuple:
        """Inject a random applicable error, return (error_word, error_type)."""
        return self.error_injector.inject_random(word)

    def postprocess(self, text: str) -> str:
        """Apply morphology-based post-processing to fix agreement errors."""
        return self.postprocessor.process(text)

    def get_postprocess_fixes(self, text: str) -> list:
        """Get list of fixes that post-processing would apply."""
        return self.postprocessor.get_fixes(text)
