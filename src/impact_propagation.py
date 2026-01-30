"""
impact_propagation.py - Analisis dampak kerusakan jalan ke jalan sekitar.

Input: Koordinat jalan rusak + severity
Output: 
  - Skor kekritisan jalan rusak
  - Dampak ke jalan-jalan lain
  - Rekomendasi
"""

import torch
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from scipy.spatial import KDTree
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum


class RoadCondition(Enum):
    """Klasifikasi kondisi jalan dari Vision Model."""
    GOOD = "good"           # Mulus, tidak ada masalah
    FAIR = "fair"           # Bergetar saat dilalui
    POOR = "poor"           # Ada retakan
    VERY_POOR = "very_poor" # Berlubang


# Mapping kondisi ke severity (0-1)
CONDITION_SEVERITY_MAP = {
    RoadCondition.GOOD: 0.0,       # Tidak ada efek
    RoadCondition.FAIR: 0.2,       # Sedikit lambat
    RoadCondition.POOR: 0.5,       # Perlu hati-hati
    RoadCondition.VERY_POOR: 0.9,  # Hampir tidak bisa lewat
}

# Deskripsi untuk output
CONDITION_DESCRIPTION = {
    RoadCondition.GOOD: "Kondisi baik, tidak ada masalah",
    RoadCondition.FAIR: "Sedikit bergetar, kendaraan melambat",
    RoadCondition.POOR: "Ada retakan, perlu hati-hati",
    RoadCondition.VERY_POOR: "Berlubang, hampir tidak bisa dilalui",
}
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, BatchNorm


# --- Model Definition (V4) - sama dengan training ---
class DeepGraphSAGE(nn.Module):
    def __init__(self, in_channels, hidden_dim=128, out_dim=128, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, hidden_dim)
        self.conv1 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.conv2 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.conv3 = SAGEConv(hidden_dim, hidden_dim, aggr='mean')
        self.conv4 = SAGEConv(hidden_dim, out_dim, aggr='mean')
        self.bn1 = BatchNorm(hidden_dim)
        self.bn2 = BatchNorm(hidden_dim)
        self.bn3 = BatchNorm(hidden_dim)
        self.bn4 = BatchNorm(out_dim)
        self.dropout = dropout
        
    def forward(self, x, edge_index):
        x = F.relu(self.input_proj(x))
        h = self.conv1(x, edge_index); h = self.bn1(h); h = F.relu(h); h = F.dropout(h, p=self.dropout, training=self.training); x = x + h
        h = self.conv2(x, edge_index); h = self.bn2(h); h = F.relu(h); h = F.dropout(h, p=self.dropout, training=self.training); x = x + h
        h = self.conv3(x, edge_index); h = self.bn3(h); h = F.relu(h); h = F.dropout(h, p=self.dropout, training=self.training); x = x + h
        x = self.conv4(x, edge_index); x = self.bn4(x); x = F.relu(x)
        return x


class MultiHeadScorer(nn.Module):
    def __init__(self, node_dim, edge_dim, hidden_dim=256, num_heads=3, dropout=0.2):
        super().__init__()
        input_dim = 3 * node_dim + edge_dim
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_heads)
        ])
        self.head_attention = nn.Linear(input_dim, num_heads)
        self.edge_attention = nn.Sequential(nn.Linear(edge_dim, edge_dim), nn.Sigmoid())
        
    def forward(self, node_u, node_v, edge_attr):
        edge_weight = self.edge_attention(edge_attr)
        weighted_edge = edge_attr * edge_weight
        node_product = node_u * node_v
        features = torch.cat([node_u, node_v, node_product, weighted_edge], dim=1)
        head_outputs = torch.stack([head(features) for head in self.heads], dim=1).squeeze(-1)
        attention = F.softmax(self.head_attention(features), dim=-1)
        combined = (attention * head_outputs).sum(dim=-1, keepdim=True)
        return torch.sigmoid(combined)


class EdgeCriticalityGNNV4(nn.Module):
    def __init__(self, num_node_features, num_edge_features, hidden_dim=128, 
                 scorer_hidden=256, num_heads=3, dropout=0.2):
        super().__init__()
        self.encoder = DeepGraphSAGE(num_node_features, hidden_dim, hidden_dim, dropout)
        self.scorer = MultiHeadScorer(hidden_dim, num_edge_features, scorer_hidden, num_heads, dropout)
        
    def forward(self, x, edge_index, edge_attr):
        node_emb = self.encoder(x, edge_index)
        src, tgt = edge_index
        return self.scorer(node_emb[src], node_emb[tgt], edge_attr).squeeze(-1)


@dataclass
class AffectedRoad:
    """Jalan yang terdampak kerusakan."""
    edge_idx: int
    u: int
    v: int
    original_score: float
    new_score: float
    impact_increase: float  # Persentase kenaikan
    distance_from_damage: int  # Hop distance dari jalan rusak
    
    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization."""
        return {
            "edge": f"{self.u}→{self.v}",
            "impact": round(self.impact_increase, 1),
            "distance": self.distance_from_damage
        }


@dataclass  
class DamageAnalysisResult:
    """Hasil analisis kerusakan jalan."""
    # Input
    latitude: float
    longitude: float
    severity: float
    condition: Optional[str] = None  # good/fair/poor/very_poor
    condition_description: Optional[str] = None
    
    # Jalan yang rusak
    damaged_edge_idx: int = 0
    damaged_road_u: int = 0
    damaged_road_v: int = 0
    damaged_road_score: float = 0.0
    damaged_road_percentile: float = 0.0
    criticality_level: str = ""
    
    # Dampak
    affected_roads: List[AffectedRoad] = None
    total_affected: int = 0
    max_impact: float = 0.0
    avg_impact: float = 0.0
    
    # Rekomendasi
    priority_level: str = ""
    recommendation: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization (minimal response).
        
        Priority Score Formula:
        = (0.6 × Percentile) + (0.4 × Severity × 100) + (Impact_Max × 0.2)
        """
        priority_score = (
            (0.6 * self.damaged_road_percentile) + 
            (0.4 * self.severity * 100) + 
            (self.max_impact * 0.2)
        )
        
        return {
            "metadata": {
                "latitude": self.latitude,
                "longitude": self.longitude,
                "condition": self.condition
            },
            "priority_score": round(priority_score, 2),
            "percentile": round(self.damaged_road_percentile, 1),
            "max_impact": round(self.max_impact, 1),
            "affected_count": self.total_affected,
            "affected_roads": [r.to_dict() for r in (self.affected_roads or [])]
        }


class RoadDamageAnalyzer:
    """
    Analyzer untuk menghitung dampak kerusakan jalan.
    """
    
    def __init__(
        self,
        model_dir: Path = Path("models"),
        data_dir: Path = Path("data/processed"),
        device: str = "cpu"
    ):
        self.model_dir = Path(model_dir)
        self.data_dir = Path(data_dir)
        self.device = torch.device(device)
        
        # Load semua komponen
        self._load_graph()
        self._load_model()
        self._build_kdtree()
        self._precompute_scores()
        self._build_adjacency()
        
    def _load_graph(self):
        """Load graph data."""
        print("Loading graph...")
        self.graph = torch.load(self.data_dir / "graph.pt", weights_only=False)
        self.graph = self.graph.to(self.device)
        
        # Load edge dataframe untuk info tambahan
        self.edges_df = pd.read_parquet(self.data_dir / "edges_processed.parquet")
        
        # Load node coordinates
        self.nodes_raw = pd.read_parquet(self.data_dir / "nodes_raw_coords.parquet")
        if 'osmid' in self.nodes_raw.columns:
            self.nodes_raw = self.nodes_raw.rename(columns={'osmid': 'node_id', 'y': 'latitude', 'x': 'longitude'})
        
        print(f"Graph loaded: {self.graph.num_nodes} nodes, {self.graph.num_edges} edges")
        
    def _load_model(self):
        """Load trained model."""
        print("Loading model...")
        config = joblib.load(self.model_dir / "model_config.pkl")
        
        self.model = EdgeCriticalityGNNV4(
            num_node_features=config['num_node_features'],
            num_edge_features=config['num_edge_features'],
            hidden_dim=config['hidden_dim'],
            scorer_hidden=config['scorer_hidden'],
            num_heads=config['num_heads'],
            dropout=config['dropout']
        ).to(self.device)
        
        self.model.load_state_dict(
            torch.load(self.model_dir / "model_best.pt", map_location=self.device, weights_only=True)
        )
        self.model.eval()
        print("Model loaded!")
        
    def _build_kdtree(self):
        """Build KDTree untuk cari edge terdekat dari koordinat."""
        print("Building KDTree...")
        node_coords = self.nodes_raw.set_index('node_id')[['latitude', 'longitude']].to_dict('index')
        
        midpoints = []
        valid_edges = []
        
        for idx, row in self.edges_df.iterrows():
            u, v = int(row['u']), int(row['v'])
            if u in node_coords and v in node_coords:
                lat_u, lon_u = node_coords[u]['latitude'], node_coords[u]['longitude']
                lat_v, lon_v = node_coords[v]['latitude'], node_coords[v]['longitude']
                midpoints.append([(lat_u + lat_v) / 2, (lon_u + lon_v) / 2])
                valid_edges.append(idx)
        
        self.midpoints = np.array(midpoints)
        self.valid_edge_indices = np.array(valid_edges)
        self.kdtree = KDTree(self.midpoints)
        print(f"KDTree built with {len(self.midpoints)} edges")
        
    def _precompute_scores(self):
        """Precompute skor untuk semua edges."""
        print("Precomputing scores...")
        with torch.no_grad():
            self.baseline_scores = self.model(
                self.graph.x, self.graph.edge_index, self.graph.edge_attr
            ).cpu().numpy()
        print(f"Scores computed: min={self.baseline_scores.min():.4f}, max={self.baseline_scores.max():.4f}")
        
    def _build_adjacency(self):
        """Build adjacency list untuk neighbor lookup."""
        print("Building adjacency list...")
        self.edge_neighbors = {}  # edge_idx -> list of neighbor edge indices
        
        edge_index = self.graph.edge_index.cpu().numpy()
        num_edges = edge_index.shape[1]
        
        # Build node -> edges mapping
        node_to_edges = {}
        for e_idx in range(num_edges):
            u, v = edge_index[0, e_idx], edge_index[1, e_idx]
            if u not in node_to_edges:
                node_to_edges[u] = []
            if v not in node_to_edges:
                node_to_edges[v] = []
            node_to_edges[u].append(e_idx)
            node_to_edges[v].append(e_idx)
        
        # Build edge neighbors (edges yang share node)
        for e_idx in range(num_edges):
            u, v = edge_index[0, e_idx], edge_index[1, e_idx]
            neighbors = set(node_to_edges.get(u, []) + node_to_edges.get(v, []))
            neighbors.discard(e_idx)  # Remove self
            self.edge_neighbors[e_idx] = list(neighbors)
        
        print("Adjacency list built!")
    
    def _get_criticality_level(self, percentile: float) -> str:
        """Konversi percentile ke level."""
        if percentile >= 99:
            return "SANGAT KRITIS"
        elif percentile >= 95:
            return "KRITIS"
        elif percentile >= 85:
            return "TINGGI"
        elif percentile >= 70:
            return "SEDANG"
        else:
            return "RENDAH"
    
    def _get_priority_level(self, criticality: str, severity: float) -> str:
        """Hitung priority berdasarkan criticality dan severity kerusakan."""
        score = 0
        if criticality in ["SANGAT KRITIS", "KRITIS"]:
            score += 2
        elif criticality == "TINGGI":
            score += 1
            
        if severity >= 0.7:
            score += 2
        elif severity >= 0.4:
            score += 1
            
        if score >= 3:
            return "SANGAT TINGGI"
        elif score >= 2:
            return "TINGGI"
        elif score >= 1:
            return "SEDANG"
        else:
            return "RENDAH"
    
    def find_nearest_edge(self, lat: float, lon: float) -> Tuple[int, float]:
        """Cari edge terdekat dari koordinat."""
        distances, indices = self.kdtree.query([[lat, lon]], k=1)
        edge_idx = indices[0]
        distance = distances[0]
        return int(edge_idx), float(distance)
    
    def get_affected_roads(
        self, 
        damaged_edge_idx: int, 
        severity: float,
        max_hops: int = 3,
        min_impact: float = 0.01
    ) -> List[AffectedRoad]:
        """
        Hitung jalan-jalan yang terdampak kerusakan.
        
        Menggunakan BFS untuk cari neighbors sampai max_hops.
        Impact dihitung berdasarkan proximity dan original score.
        """
        affected = []
        visited = {damaged_edge_idx}
        current_level = [damaged_edge_idx]
        
        edge_index = self.graph.edge_index.cpu().numpy()
        
        for hop in range(1, max_hops + 1):
            next_level = []
            
            for e_idx in current_level:
                for neighbor_idx in self.edge_neighbors.get(e_idx, []):
                    if neighbor_idx not in visited:
                        visited.add(neighbor_idx)
                        next_level.append(neighbor_idx)
                        
                        # Hitung impact
                        original_score = self.baseline_scores[neighbor_idx]
                        
                        # Impact = severity * (1/hop distance) * neighbor's importance
                        # Jalan dengan skor tinggi lebih mudah terdampak karena sudah sibuk
                        impact_factor = severity * (1 / hop) * (0.3 + original_score * 0.7)
                        impact_increase = impact_factor * 100  # Convert to percentage
                        
                        if impact_increase >= min_impact:
                            u = int(edge_index[0, neighbor_idx])
                            v = int(edge_index[1, neighbor_idx])
                            
                            affected.append(AffectedRoad(
                                edge_idx=neighbor_idx,
                                u=u,
                                v=v,
                                original_score=original_score,
                                new_score=min(1.0, original_score + impact_factor),
                                impact_increase=impact_increase,
                                distance_from_damage=hop
                            ))
            
            current_level = next_level
            if not current_level:
                break
        
        # Sort by impact (descending)
        affected.sort(key=lambda x: x.impact_increase, reverse=True)
        return affected
    
    def analyze_damage(
        self, 
        latitude: float, 
        longitude: float, 
        severity: float = 0.5,
        max_affected: int = 20
    ) -> DamageAnalysisResult:
        """
        Analisis lengkap dampak kerusakan jalan.
        
        Args:
            latitude: Latitude jalan rusak
            longitude: Longitude jalan rusak
            severity: Tingkat keparahan kerusakan (0-1)
            max_affected: Max jumlah jalan terdampak yang dilaporkan
            
        Returns:
            DamageAnalysisResult dengan semua info analisis
        """
        # 1. Cari edge terdekat
        edge_idx, distance = self.find_nearest_edge(latitude, longitude)
        
        # 2. Ambil info jalan rusak
        edge_index = self.graph.edge_index.cpu().numpy()
        u = int(edge_index[0, edge_idx])
        v = int(edge_index[1, edge_idx])
        
        damaged_score = float(self.baseline_scores[edge_idx])
        damaged_percentile = (self.baseline_scores < damaged_score).mean() * 100
        criticality_level = self._get_criticality_level(damaged_percentile)
        
        # 3. Hitung dampak ke jalan lain
        affected_roads = self.get_affected_roads(edge_idx, severity)[:max_affected]
        
        # 4. Hitung statistik
        if affected_roads:
            impacts = [r.impact_increase for r in affected_roads]
            max_impact = max(impacts)
            avg_impact = sum(impacts) / len(impacts)
        else:
            max_impact = 0.0
            avg_impact = 0.0
        
        # 5. Generate rekomendasi
        priority = self._get_priority_level(criticality_level, severity)
        
        if priority == "SANGAT TINGGI":
            recommendation = "SEGERA perbaiki! Jalan ini kritis dan kerusakan parah. Siapkan rerouting."
        elif priority == "TINGGI":
            recommendation = "Prioritas tinggi untuk perbaikan. Monitor jalan alternatif."
        elif priority == "SEDANG":
            recommendation = "Jadwalkan perbaikan dalam waktu dekat."
        else:
            recommendation = "Dapat diperbaiki sesuai jadwal maintenance rutin."
        
        return DamageAnalysisResult(
            latitude=latitude,
            longitude=longitude,
            severity=severity,
            damaged_edge_idx=edge_idx,
            damaged_road_u=u,
            damaged_road_v=v,
            damaged_road_score=damaged_score,
            damaged_road_percentile=damaged_percentile,
            criticality_level=criticality_level,
            affected_roads=affected_roads,
            total_affected=len(affected_roads),
            max_impact=max_impact,
            avg_impact=avg_impact,
            priority_level=priority,
            recommendation=recommendation
        )
    
    def analyze_from_vision(
        self,
        latitude: float,
        longitude: float,
        condition: str,  # "good", "fair", "poor", "very_poor"
        max_affected: int = 20
    ) -> DamageAnalysisResult:
        """
        Analisis kerusakan jalan dari hasil klasifikasi Vision Model.
        
        Args:
            latitude: Latitude jalan
            longitude: Longitude jalan
            condition: Kondisi jalan dari vision model ("good", "fair", "poor", "very_poor")
            max_affected: Max jumlah jalan terdampak
            
        Returns:
            DamageAnalysisResult dengan analisis lengkap
        """
        # Parse condition
        condition_lower = condition.lower().strip()
        
        # Map string ke enum
        condition_map = {
            "good": RoadCondition.GOOD,
            "fair": RoadCondition.FAIR,
            "poor": RoadCondition.POOR,
            "very_poor": RoadCondition.VERY_POOR,
            "verypoor": RoadCondition.VERY_POOR,
            "very poor": RoadCondition.VERY_POOR,
        }
        
        if condition_lower not in condition_map:
            raise ValueError(f"Unknown condition: {condition}. Expected: good, fair, poor, very_poor")
        
        road_condition = condition_map[condition_lower]
        severity = CONDITION_SEVERITY_MAP[road_condition]
        description = CONDITION_DESCRIPTION[road_condition]
        
        # Jika kondisi GOOD, tidak perlu analisis dampak
        if road_condition == RoadCondition.GOOD:
            edge_idx, distance = self.find_nearest_edge(latitude, longitude)
            edge_index = self.graph.edge_index.cpu().numpy()
            u = int(edge_index[0, edge_idx])
            v = int(edge_index[1, edge_idx])
            damaged_score = float(self.baseline_scores[edge_idx])
            damaged_percentile = (self.baseline_scores < damaged_score).mean() * 100
            
            return DamageAnalysisResult(
                latitude=latitude,
                longitude=longitude,
                severity=severity,
                condition=condition,
                condition_description=description,
                damaged_edge_idx=edge_idx,
                damaged_road_u=u,
                damaged_road_v=v,
                damaged_road_score=damaged_score,
                damaged_road_percentile=damaged_percentile,
                criticality_level=self._get_criticality_level(damaged_percentile),
                affected_roads=[],
                total_affected=0,
                max_impact=0.0,
                avg_impact=0.0,
                priority_level="TIDAK ADA KERUSAKAN",
                recommendation="Kondisi jalan baik. Tidak perlu tindakan."
            )
        
        # Lakukan analisis normal
        result = self.analyze_damage(latitude, longitude, severity, max_affected)
        
        # Tambahkan info kondisi
        result.condition = condition
        result.condition_description = description
        
        return result
    
    def print_analysis(self, result: DamageAnalysisResult):
        """Print hasil analisis dengan format yang readable."""
        print("\n" + "=" * 70)
        print("🚧 ANALISIS KERUSAKAN JALAN")
        print("=" * 70)
        
        print(f"\n📍 LOKASI KERUSAKAN")
        print(f"   Koordinat: ({result.latitude}, {result.longitude})")
        
        # Show condition if available (from Vision model)
        if result.condition:
            print(f"   Kondisi: {result.condition.upper()}")
            print(f"   Deskripsi: {result.condition_description}")
        
        print(f"   Severity: {result.severity * 100:.0f}%")
        
        print(f"\n🛣️  JALAN YANG DIANALISIS")
        print(f"   Edge: {result.damaged_road_u} → {result.damaged_road_v}")
        print(f"   Skor Kekritisan: {result.damaged_road_score:.4f}")
        print(f"   Percentile: {result.damaged_road_percentile:.1f}%")
        print(f"   Level Kepentingan: {result.criticality_level}")
        
        print(f"\n⚠️  DAMPAK KE JALAN LAIN ({result.total_affected} jalan)")
        print("-" * 70)
        
        if result.affected_roads:
            for i, road in enumerate(result.affected_roads[:10], 1):
                level_emoji = "🔴" if road.impact_increase > 20 else "🟠" if road.impact_increase > 10 else "🟡"
                print(f"   {i}. Edge {road.u}→{road.v} | "
                      f"Impact: +{road.impact_increase:.1f}% {level_emoji} | "
                      f"Jarak: {road.distance_from_damage} hop")
        else:
            print("   Tidak ada jalan yang terdampak signifikan.")
        
        print(f"\n📊 STATISTIK DAMPAK")
        print(f"   Total jalan terdampak: {result.total_affected}")
        print(f"   Impact maksimum: +{result.max_impact:.1f}%")
        print(f"   Impact rata-rata: +{result.avg_impact:.1f}%")
        
        print(f"\n📌 REKOMENDASI")
        print(f"   Prioritas: {result.priority_level}")
        print(f"   {result.recommendation}")
        
        print("\n" + "=" * 70)


def main():
    """Test impact propagation dengan kondisi dari Vision Model."""
    analyzer = RoadDamageAnalyzer(
        model_dir=Path("models"),
        data_dir=Path("data/processed")
    )
    
    # Test cases - simulasi input dari Vision Model
    # Format: (lat, lon, condition, desc)
    test_cases = [
        (-6.9932, 110.4203, "very_poor", "Simpang Lima - Berlubang"),
        (-6.9708, 110.4292, "poor", "Jl. Pandanaran - Retak"),
        (-7.0489, 110.4378, "very_poor", "UNDIP Tembalang - Berlubang"),
        (-6.9506, 110.3978, "fair", "Bandara Ahmad Yani - Bergetar"),
        (-6.9847, 110.4097, "good", "Jl. Gajah Mada - Mulus"),
    ]
    
    print("\n" + "=" * 70)
    print("🔗 INTEGRASI VISION MODEL → GNN IMPACT ANALYSIS")
    print("=" * 70)
    print("\nMapping Kondisi → Severity:")
    print("  • good      → 0%   (tidak ada masalah)")
    print("  • fair      → 20%  (bergetar)")
    print("  • poor      → 50%  (retak)")
    print("  • very_poor → 90%  (berlubang)")
    
    for lat, lon, condition, desc in test_cases:
        print(f"\n\n{'#' * 70}")
        print(f"# INPUT DARI VISION: {desc}")
        print(f"# Koordinat: ({lat}, {lon})")
        print(f"# Kondisi: {condition}")
        print(f"{'#' * 70}")
        
        result = analyzer.analyze_from_vision(lat, lon, condition)
        analyzer.print_analysis(result)


if __name__ == "__main__":
    main()

