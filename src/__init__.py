# GNN Road Criticality API
from .impact_propagation import (
    RoadDamageAnalyzer,
    RoadCondition,
    DamageAnalysisResult,
    AffectedRoad,
    CONDITION_SEVERITY_MAP,
    CONDITION_DESCRIPTION,
)

from .cnn_classifier import RoadDamageClassifier
from .unified_pipeline import UnifiedRoadAnalyzer

__all__ = [
    # GNN
    "RoadDamageAnalyzer",
    "RoadCondition", 
    "DamageAnalysisResult",
    "AffectedRoad",
    "CONDITION_SEVERITY_MAP",
    "CONDITION_DESCRIPTION",
    # CNN
    "RoadDamageClassifier",
    # Unified Pipeline
    "UnifiedRoadAnalyzer",
]
