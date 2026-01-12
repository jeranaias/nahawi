# Arabic Conjugation Engine
# Comprehensive verb conjugation for all root types and forms

from .root_types import RootType, classify_root
from .verb_forms import VerbForm, VERB_FORMS
from .conjugator import ArabicConjugator
from .paradigms import get_paradigm

__all__ = [
    'RootType',
    'classify_root',
    'VerbForm',
    'VERB_FORMS',
    'ArabicConjugator',
    'get_paradigm',
]
