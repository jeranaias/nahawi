"""
Nahawi - Arabic Grammatical Error Correction

A 124M parameter transformer with LoRA achieving 78.84% F0.5 on QALB-2014.
95.4% of SOTA (82.63% by ArbESC+) with 6x fewer parameters.
"""

from .model import NahawiModel

__version__ = "1.1.0"
__all__ = ["NahawiModel"]
