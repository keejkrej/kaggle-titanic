from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch import Tensor

from src.model import TitanicNet


class EnsembleModel:
    """Ensemble of multiple TitanicNet models for improved accuracy."""
    
    def __init__(self, models: List[TitanicNet], device: torch.device):
        self.models = models
        self.device = device
        for model in self.models:
            model.to(device)
            model.eval()
    
    def predict(self, features: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Make predictions by averaging outputs from all models.
        
        Args:
            features: Input features
            weights: Optional weights for each model (default: equal weights)
        """
        feature_tensor = torch.from_numpy(features).float().to(self.device)
        
        predictions = []
        with torch.no_grad():
            for model in self.models:
                logits = model(feature_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                predictions.append(probs)
        
        predictions = np.array(predictions)
        
        # Weighted average if weights provided, otherwise simple average
        if weights is not None:
            weights = np.array(weights)
            weights = weights / weights.sum()  # Normalize
            avg_probs = np.average(predictions, axis=0, weights=weights)
        else:
            avg_probs = np.mean(predictions, axis=0)
        
        return (avg_probs >= 0.5).astype(int)
    
    def predict_proba(self, features: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
        """Return average probabilities from all models."""
        feature_tensor = torch.from_numpy(features).float().to(self.device)
        
        predictions = []
        with torch.no_grad():
            for model in self.models:
                logits = model(feature_tensor)
                probs = torch.sigmoid(logits).cpu().numpy().ravel()
                predictions.append(probs)
        
        predictions = np.array(predictions)
        
        if weights is not None:
            weights = np.array(weights)
            weights = weights / weights.sum()
            return np.average(predictions, axis=0, weights=weights)
        return np.mean(predictions, axis=0)
    
    @classmethod
    def load_ensemble(
        cls,
        model_paths: List[Path],
        input_dim: int,
        device: torch.device,
    ) -> EnsembleModel:
        """Load multiple models and create an ensemble (assumes default architecture)."""
        models = []
        for model_path in model_paths:
            model = TitanicNet(input_dim=input_dim)
            model.load_state_dict(torch.load(model_path, map_location=device))
            models.append(model)
        return cls(models, device)
    
    @classmethod
    def load_ensemble_from_info(
        cls,
        model_info_list: List[dict],
        input_dim: int,
        device: torch.device,
    ) -> EnsembleModel:
        """Load multiple models with their saved architectures."""
        import json
        models = []
        for info in model_info_list:
            model_path = Path(info["model_path"])
            arch_path = Path(info["arch_path"])
            
            # Load architecture info
            if arch_path.exists():
                with arch_path.open() as f:
                    arch_data = json.load(f)
                hidden_units = tuple(arch_data["hidden_units"])
            else:
                # Fallback to default if arch file doesn't exist
                hidden_units = (256, 128, 64, 32)
            
            # Create model with correct architecture
            model = TitanicNet(input_dim=input_dim, hidden_units=hidden_units)
            model.load_state_dict(torch.load(model_path, map_location=device))
            models.append(model)
        return cls(models, device)
    
    @classmethod
    def load_ensemble_from_json(
        cls,
        ensemble_info_path: Path,
        input_dim: int,
        device: torch.device,
    ) -> EnsembleModel:
        """Load ensemble from saved ensemble_info.json file."""
        import json
        with ensemble_info_path.open() as f:
            ensemble_info = json.load(f)
        
        model_info_list = ensemble_info["model_info"]
        return cls.load_ensemble_from_info(model_info_list, input_dim, device)
