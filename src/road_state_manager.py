"""
road_state_manager.py - State management untuk tracking damaged roads.

Fitur:
- Singleton pattern untuk shared state antar request
- Report damage dengan cascade effects ke neighboring roads
- Mark fixed dengan recalculation
- Accumulative impact dari multiple damaged roads
"""

from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from collections import deque
import numpy as np

from .impact_propagation import RoadDamageAnalyzer, CONDITION_SEVERITY_MAP, RoadCondition


# Condition string to severity mapping
CONDITION_TO_SEVERITY = {
    "good": 0.0,
    "fair": 0.2,
    "poor": 0.5,
    "very_poor": 0.9,
}


@dataclass
class DamageRecord:
    """Record of a damaged road."""
    edge_idx: int
    u: int
    v: int
    lat: float
    lon: float
    condition: str
    severity: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> dict:
        return {
            "edge": f"{self.u}→{self.v}",
            "edge_idx": self.edge_idx,
            "lat": self.lat,
            "lon": self.lon,
            "condition": self.condition,
            "severity": self.severity,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class RoadStatus:
    """Current status of a road segment."""
    edge_idx: int
    u: int
    v: int
    is_damaged: bool
    original_score: float
    current_score: float
    impact_from_others: float
    impact_sources: List[dict] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "edge": f"{self.u}→{self.v}",
            "is_damaged": self.is_damaged,
            "original_score": round(self.original_score, 4),
            "current_score": round(self.current_score, 4),
            "impact_from_others": round(self.impact_from_others, 4),
            "impact_sources": self.impact_sources
        }


class RoadStateManager:
    """
    Singleton state manager untuk tracking damaged roads.
    
    Manages:
    - damaged_roads: Currently damaged roads
    - impact_sources: Impact on each road from damaged roads
    - baseline_scores: Original model scores
    - current_scores: Scores after cascade calculation
    """
    
    _instance: Optional["RoadStateManager"] = None
    
    def __new__(cls, analyzer: RoadDamageAnalyzer = None):
        if cls._instance is None:
            if analyzer is None:
                raise ValueError("Analyzer required for first initialization")
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, analyzer: RoadDamageAnalyzer = None):
        if self._initialized:
            return
            
        self.analyzer = analyzer
        
        # State
        self.damaged_roads: Dict[int, DamageRecord] = {}
        self.impact_sources: Dict[int, List[Tuple[int, float]]] = {}
        
        # Scores
        self.baseline_scores = analyzer.baseline_scores.copy()
        self.current_scores = analyzer.baseline_scores.copy()
        
        # Cache edge info
        self.edge_index = analyzer.graph.edge_index.cpu().numpy()
        
        self._initialized = True
        print("RoadStateManager initialized!")
    
    def report_damage(self, lat: float, lon: float, condition: str) -> DamageRecord:
        """
        Report damage at a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            condition: good/fair/poor/very_poor
            
        Returns:
            DamageRecord for the damaged road
        """
        # Find nearest edge
        edge_idx, _ = self.analyzer.find_nearest_edge(lat, lon)
        
        # Get edge nodes
        u = int(self.edge_index[0, edge_idx])
        v = int(self.edge_index[1, edge_idx])
        
        # Get severity
        severity = CONDITION_TO_SEVERITY.get(condition.lower(), 0.0)
        
        # Create record
        record = DamageRecord(
            edge_idx=edge_idx,
            u=u,
            v=v,
            lat=lat,
            lon=lon,
            condition=condition,
            severity=severity
        )
        
        # Add to state
        self.damaged_roads[edge_idx] = record
        
        # Recalculate cascade
        self._recalculate_with_cascade()
        
        return record
    
    def mark_fixed(self, lat: float, lon: float) -> bool:
        """
        Mark a road as fixed.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            True if road was found and fixed, False if not found
        """
        edge_idx, _ = self.analyzer.find_nearest_edge(lat, lon)
        
        if edge_idx not in self.damaged_roads:
            return False
        
        # Remove from state
        del self.damaged_roads[edge_idx]
        
        # Recalculate cascade
        self._recalculate_with_cascade()
        
        return True
    
    def _recalculate_with_cascade(self):
        """
        Recalculate all scores with cascade effects from all damaged roads.
        
        Logic:
        1. Reset current_scores to baseline
        2. For each damaged road:
           - Degrade damaged road: score = baseline * (1 - severity * 0.6)
           - BFS propagate impact to neighbors (max 3 hops)
           - Accumulate impact sources
        3. Apply accumulated impacts
        """
        # Reset
        self.current_scores = self.baseline_scores.copy()
        self.impact_sources = {}
        
        if not self.damaged_roads:
            return
        
        # Collect all impacts: edge_idx -> list of (source_idx, impact_value)
        all_impacts: Dict[int, List[Tuple[int, float]]] = {}
        
        for edge_idx, record in self.damaged_roads.items():
            severity = record.severity
            
            # 1. Degrade the damaged road itself
            degradation = severity * 0.6
            self.current_scores[edge_idx] = self.baseline_scores[edge_idx] * (1 - degradation)
            
            # 2. BFS propagation to neighbors
            visited = {edge_idx}
            current_level = [edge_idx]
            
            for hop in range(1, 4):  # max 3 hops
                next_level = []
                
                for e_idx in current_level:
                    for neighbor_idx in self.analyzer.edge_neighbors.get(e_idx, []):
                        if neighbor_idx not in visited and neighbor_idx not in self.damaged_roads:
                            visited.add(neighbor_idx)
                            next_level.append(neighbor_idx)
                            
                            # Calculate impact (positive = traffic increase)
                            # Neighbors get more traffic as alternative routes
                            original_score = self.baseline_scores[neighbor_idx]
                            impact = severity * (1 / hop) * (0.3 + original_score * 0.7)
                            
                            # Accumulate
                            if neighbor_idx not in all_impacts:
                                all_impacts[neighbor_idx] = []
                            all_impacts[neighbor_idx].append((edge_idx, float(impact)))
                
                current_level = next_level
                if not current_level:
                    break
        
        # 3. Apply accumulated impacts
        for edge_idx, sources in all_impacts.items():
            total_impact = sum(impact for _, impact in sources)
            self.current_scores[edge_idx] = min(1.0, self.baseline_scores[edge_idx] + total_impact)
            self.impact_sources[edge_idx] = sources
    
    def get_road_status(self, lat: float, lon: float) -> RoadStatus:
        """
        Get current status of a road.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            RoadStatus with current and original scores
        """
        edge_idx, _ = self.analyzer.find_nearest_edge(lat, lon)
        
        u = int(self.edge_index[0, edge_idx])
        v = int(self.edge_index[1, edge_idx])
        
        is_damaged = edge_idx in self.damaged_roads
        original_score = float(self.baseline_scores[edge_idx])
        current_score = float(self.current_scores[edge_idx])
        
        # Impact from others
        sources = self.impact_sources.get(edge_idx, [])
        impact_from_others = sum(impact for _, impact in sources)
        
        # Format sources
        impact_sources_formatted = []
        for src_idx, impact in sources:
            if src_idx in self.damaged_roads:
                record = self.damaged_roads[src_idx]
                impact_sources_formatted.append({
                    "from_edge": f"{record.u}→{record.v}",
                    "condition": record.condition,
                    "impact": round(impact, 4)
                })
        
        return RoadStatus(
            edge_idx=edge_idx,
            u=u,
            v=v,
            is_damaged=is_damaged,
            original_score=original_score,
            current_score=current_score,
            impact_from_others=impact_from_others,
            impact_sources=impact_sources_formatted
        )
    
    def get_damaged_list(self) -> List[DamageRecord]:
        """Get list of all damaged roads."""
        return list(self.damaged_roads.values())
    
    def get_summary(self) -> dict:
        """Get summary of current state."""
        damaged_list = [r.to_dict() for r in self.damaged_roads.values()]
        
        # Count affected roads (roads receiving impact from damaged roads)
        affected_count = len([
            idx for idx in self.impact_sources
            if idx not in self.damaged_roads
        ])
        
        return {
            "total_damaged": len(self.damaged_roads),
            "total_affected": affected_count,
            "damaged_roads": damaged_list
        }
    
    def clear_all(self):
        """Clear all damage records and reset to baseline."""
        self.damaged_roads.clear()
        self.impact_sources.clear()
        self.current_scores = self.baseline_scores.copy()
        print("All damage records cleared!")
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (for testing)."""
        cls._instance = None


def get_state_manager(analyzer: RoadDamageAnalyzer = None) -> RoadStateManager:
    """
    Get or create the RoadStateManager singleton.
    
    Args:
        analyzer: Required for first call
        
    Returns:
        RoadStateManager instance
    """
    return RoadStateManager(analyzer)
