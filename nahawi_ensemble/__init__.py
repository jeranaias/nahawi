"""
Nahawi Ensemble - Arabic GEC System
====================================

A 10-model ensemble for Arabic Grammatical Error Correction.

Models:
1. HamzaFixer - Hamza confusion (أ/إ/ا/آ/ء/ؤ/ئ)
2. SpaceFixer - Merge/split errors
3. TaaMarbutaFixer - ة ↔ ه
4. AlifMaksuraFixer - ى ↔ ي
5. PunctuationFixer - Arabic punctuation
6. DeletedWordFixer - Missing words
7. RepeatedWordFixer - Duplicate words
8. SpellingFixer - General spelling
9. MorphologyFixer - Agreement errors
10. GeneralGEC - Catch-all

Usage:
    from nahawi_ensemble import NahawiEnsemble

    ensemble = NahawiEnsemble()
    corrected = ensemble.correct("النص الخاطئ")
"""

__version__ = "0.1.0"

from .orchestrator import NahawiEnsemble
from .config import Config
