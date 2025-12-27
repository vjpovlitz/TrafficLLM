"""
SAM 3 Annotation Module

Provides intelligent annotation capabilities using Meta's Segment Anything Model 3.
"""

from .sam3_annotator import SAM3Annotator, AnnotationResult
from .exporter import AnnotationExporter, ExportFormat
from .verifier import AnnotationVerifier, QualityMetrics

__all__ = [
    'SAM3Annotator',
    'AnnotationResult',
    'AnnotationExporter',
    'ExportFormat',
    'AnnotationVerifier',
    'QualityMetrics'
]
