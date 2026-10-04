"""
AVA LDM Studio — Linguistic Normalization Package
"""
from .slang_mapper import SlangMapper, PhraseMatch, TokenGloss
from .semantic_preserver import SemanticPreserver, ExtractionResult, SemanticAnchor, PreservationReport
from .dialect_adapter import DialectAdapter, DialectAdaptation
from .normalizer import LinguisticNormalizer, NormalizationResult, NormalizationTrace

__all__ = [
    "LinguisticNormalizer",
    "NormalizationResult",
    "NormalizationTrace",
    "SlangMapper",
    "PhraseMatch",
    "TokenGloss",
    "SemanticPreserver",
    "ExtractionResult",
    "SemanticAnchor",
    "PreservationReport",
    "DialectAdapter",
    "DialectAdaptation",
]
