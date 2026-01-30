"""
unified_pipeline.py - Pipeline terintegrasi CNN → GNN untuk analisis kerusakan jalan.

Pipeline Flow:
1. Image → CNN Classifier → condition (good/fair/poor/very_poor)
2. condition + (lat, lon) → GNN Analyzer → impact analysis
3. Return full analysis result

NO API BOTTLENECK: Direct Python function calls
"""

from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass

from .cnn_classifier import RoadDamageClassifier
from .impact_propagation import RoadDamageAnalyzer, DamageAnalysisResult


@dataclass
class FullAnalysisResult:
    """Hasil analisis lengkap dari pipeline CNN-GNN."""
    
    # CNN Classification
    cnn_condition: str
    cnn_confidence: float
    cnn_probabilities: dict
    
    # GNN Analysis
    gnn_result: DamageAnalysisResult
    
    def to_dict(self) -> dict:
        """Convert ke dict untuk JSON response."""
        gnn_dict = self.gnn_result.to_dict()
        
        return {
            "classification": {
                "condition": self.cnn_condition,
                "confidence": round(self.cnn_confidence, 4),
                "probabilities": {
                    k: round(v, 4) for k, v in self.cnn_probabilities.items()
                }
            },
            "metadata": gnn_dict["metadata"],
            "priority_score": gnn_dict["priority_score"],
            "percentile": gnn_dict["percentile"],
            "max_impact": gnn_dict["max_impact"],
            "affected_count": gnn_dict["affected_count"],
            "affected_roads": gnn_dict["affected_roads"]
        }


class UnifiedRoadAnalyzer:
    """
    Pipeline terintegrasi CNN → GNN untuk analisis kerusakan jalan.
    
    Flow:
    1. Image → CNN Classifier → condition (good/fair/poor/very_poor)
    2. condition + (lat, lon) → GNN Analyzer → impact analysis
    3. Return full analysis result
    
    NO API BOTTLENECK: Direct Python function calls between models
    
    Usage:
        analyzer = UnifiedRoadAnalyzer(
            cnn_model_path=Path("models/97.14_modif_resnet18_checkpoint.pth"),
            gnn_model_dir=Path("models"),
            gnn_data_dir=Path("data")
        )
        
        result = analyzer.analyze(
            lan=-6.9932, 
            lon=110.4203, 
            img_location="path/to/image.jpg"
        )
    """
    
    def __init__(
        self,
        cnn_model_path: Union[str, Path],
        gnn_model_dir: Union[str, Path],
        gnn_data_dir: Union[str, Path],
        device: str = "cpu"
    ):
        """
        Initialize unified pipeline.
        
        Args:
            cnn_model_path: Path ke CNN checkpoint (.pth)
            gnn_model_dir: Directory containing GNN model files
            gnn_data_dir: Directory containing GNN data files
            device: Device untuk inference
        """
        self.device = device
        
        # Initialize CNN Classifier
        print("=" * 60)
        print("Initializing Unified Road Analyzer Pipeline")
        print("=" * 60)
        
        print("\n[1/2] Loading CNN Classifier...")
        self.cnn_classifier = RoadDamageClassifier(
            model_path=cnn_model_path,
            device=device
        )
        
        # Initialize GNN Analyzer
        print("\n[2/2] Loading GNN Analyzer...")
        self.gnn_analyzer = RoadDamageAnalyzer(
            model_dir=Path(gnn_model_dir),
            data_dir=Path(gnn_data_dir),
            device=device
        )
        
        print("\n" + "=" * 60)
        print("[OK] Unified Pipeline Ready!")
        print("=" * 60)
    
    def analyze(
        self, 
        lan: float, 
        lon: float, 
        img_location: Union[str, Path],
        max_affected: int = 20
    ) -> dict:
        """
        Full pipeline: Image → CNN → GNN → Result
        
        Args:
            lan: Latitude koordinat jalan
            lon: Longitude koordinat jalan
            img_location: Path ke file gambar
            max_affected: Max jumlah affected roads dalam output
            
        Returns:
            Dict dengan classification dan impact analysis
        """
        img_path = Path(img_location)
        
        # Validate image exists
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # Step 1: CNN Classification (DIRECT CALL - NO API)
        print(f"\n[CNN] Analyzing image: {img_path}")
        cnn_result = self.cnn_classifier.classify_with_confidence(img_path)
        condition = cnn_result["condition"]
        confidence = cnn_result["confidence"]
        probabilities = cnn_result["probabilities"]
        
        print(f"   CNN Result: {condition} ({confidence:.2%})")
        
        # Step 2: GNN Impact Analysis (DIRECT CALL - NO API)
        print(f"   Coordinates: ({lan}, {lon})")
        gnn_result = self.gnn_analyzer.analyze_from_vision(
            latitude=lan,
            longitude=lon,
            condition=condition,
            max_affected=max_affected
        )
        
        print(f"   GNN Result: Priority={gnn_result.priority_level}, Affected={gnn_result.total_affected}")
        
        # Step 3: Combine Results
        full_result = FullAnalysisResult(
            cnn_condition=condition,
            cnn_confidence=confidence,
            cnn_probabilities=probabilities,
            gnn_result=gnn_result
        )
        
        return full_result.to_dict()
    
    def analyze_with_full_details(
        self, 
        lan: float, 
        lon: float, 
        img_location: Union[str, Path]
    ) -> FullAnalysisResult:
        """
        Full pipeline dengan return FullAnalysisResult object.
        
        Berguna untuk akses detail lebih lanjut (bukan untuk API response).
        """
        img_path = Path(img_location)
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        # CNN Classification
        cnn_result = self.cnn_classifier.classify_with_confidence(img_path)
        
        # GNN Analysis
        gnn_result = self.gnn_analyzer.analyze_from_vision(
            latitude=lan,
            longitude=lon,
            condition=cnn_result["condition"]
        )
        
        return FullAnalysisResult(
            cnn_condition=cnn_result["condition"],
            cnn_confidence=cnn_result["confidence"],
            cnn_probabilities=cnn_result["probabilities"],
            gnn_result=gnn_result
        )


# === Main for testing ===

def main():
    """Test unified pipeline."""
    from pathlib import Path
    
    # Initialize pipeline
    analyzer = UnifiedRoadAnalyzer(
        cnn_model_path=Path("models/97.14_modif_resnet18_checkpoint.pth"),
        gnn_model_dir=Path("models"),
        gnn_data_dir=Path("data")
    )
    
    # Test with sample image
    test_image = Path("d/Test-001.jpg")
    test_lat = -6.9932
    test_lon = 110.4203
    
    if test_image.exists():
        print("\n" + "=" * 60)
        print("TESTING UNIFIED PIPELINE")
        print("=" * 60)
        
        result = analyzer.analyze(
            lan=test_lat,
            lon=test_lon,
            img_location=test_image
        )
        
        print("\n[RESULT] FULL RESULT:")
        import json
        print(json.dumps(result, indent=2))
    else:
        print(f"Test image not found: {test_image}")


if __name__ == "__main__":
    main()
