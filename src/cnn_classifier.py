"""
cnn_classifier.py - CNN Classifier untuk klasifikasi kerusakan jalan.

Model: Modified ResNet18 dengan attention modules
Classes: good, fair, poor, very_poor
Accuracy: 97.14%
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path
from typing import Union, List


# === Model Architecture ===

class RoadAttention(nn.Module):
    """Spatial attention untuk fokus pada area jalan."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial_weight = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mask = self.sigmoid(self.spatial_weight(x))
        return x * mask


class PotholeRefinement(nn.Module):
    """Channel gating untuk fokus pada pola lubang jalan."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.spatial_refine = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.spatial_refine(x)
        gate = self.channel_gate(feat)
        return feat * gate


class PotholeClassifier(nn.Module):
    """
    Road Damage Classifier berbasis ResNet18.
    
    Architecture:
    1. ResNet18 backbone (pretrained)
    2. RoadAttention - fokus pada area jalan
    3. PotholeRefinement - fokus pada pola kerusakan
    4. Classifier head
    """
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        # ResNet18 backbone tanpa fully connected layers
        self.backbone = nn.Sequential(
            *list(models.resnet18(weights='DEFAULT').children())[:-2]
        )
        
        # Attention modules
        self.road_attention = RoadAttention(in_channels=512)
        self.pothole_refine = PotholeRefinement(in_channels=512)
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)
        x = self.road_attention(x)
        x = self.pothole_refine(x)
        x = self.classifier(x)
        return x


# === Classifier Wrapper ===

class RoadDamageClassifier:
    """
    High-level classifier untuk klasifikasi kerusakan jalan.
    
    Usage:
        classifier = RoadDamageClassifier(model_path="models/97.14_modif_resnet18_checkpoint.pth")
        condition = classifier.classify("path/to/image.jpg")
        # Returns: "good", "fair", "poor", or "very_poor"
    """
    
    # Class names sesuai urutan training
    CLASS_NAMES: List[str] = ["fair", "good", "poor", "very_poor"]
    
    def __init__(
        self, 
        model_path: Union[str, Path],
        device: str = "cpu"
    ):
        """
        Initialize classifier.
        
        Args:
            model_path: Path ke checkpoint model (.pth)
            device: Device untuk inference ("cpu" atau "cuda")
        """
        self.device = torch.device(device)
        self.model_path = Path(model_path)
        
        # Initialize model
        self._load_model()
        
        # Preprocessing pipeline
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        print(f"RoadDamageClassifier loaded from {self.model_path}")
        print(f"Classes: {self.CLASS_NAMES}")
    
    def _load_model(self):
        """Load model dari checkpoint."""
        self.model = PotholeClassifier(num_classes=len(self.CLASS_NAMES))
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume checkpoint is the state dict itself
                self.model.load_state_dict(checkpoint)
        else:
            # Direct state dict
            self.model.load_state_dict(checkpoint)
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    def preprocess_image(self, image_path: Union[str, Path]) -> torch.Tensor:
        """
        Load dan preprocess image.
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Preprocessed tensor (1, 3, 224, 224)
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = Image.open(image_path).convert("RGB")
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        return tensor.unsqueeze(0)
    
    def classify(self, image_path: Union[str, Path]) -> str:
        """
        Klasifikasi kondisi jalan dari gambar.
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Kondisi jalan: "good", "fair", "poor", atau "very_poor"
        """
        # Preprocess
        tensor = self.preprocess_image(image_path).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            class_idx = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, class_idx].item()
        
        condition = self.CLASS_NAMES[class_idx]
        
        print(f"Classification: {condition} (confidence: {confidence:.2%})")
        
        return condition
    
    def classify_with_confidence(self, image_path: Union[str, Path]) -> dict:
        """
        Klasifikasi dengan detail confidence score.
        
        Args:
            image_path: Path ke file gambar
            
        Returns:
            Dict dengan condition, confidence, dan all_probabilities
        """
        # Preprocess
        tensor = self.preprocess_image(image_path).to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probabilities = torch.softmax(outputs, dim=1)
            class_idx = probabilities.argmax(dim=1).item()
            confidence = probabilities[0, class_idx].item()
        
        # Build probability dict
        all_probs = {
            name: probabilities[0, i].item() 
            for i, name in enumerate(self.CLASS_NAMES)
        }
        
        return {
            "condition": self.CLASS_NAMES[class_idx],
            "confidence": confidence,
            "probabilities": all_probs
        }


# === Main for testing ===

def main():
    """Test CNN classifier."""
    from pathlib import Path
    
    # Initialize classifier
    classifier = RoadDamageClassifier(
        model_path=Path("models/97.14_modif_resnet18_checkpoint.pth")
    )
    
    # Test with sample image
    test_image = Path("d/Test-001.jpg")
    
    if test_image.exists():
        result = classifier.classify_with_confidence(test_image)
        print(f"\nResult: {result}")
    else:
        print(f"Test image not found: {test_image}")


if __name__ == "__main__":
    main()
