import os
import torch
import torchaudio
import joblib
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

class AcousticDeepfakeDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_name = "microsoft/wavlm-base-plus"
        self.classifier_path = "models/acoustic_classifier.pkl"
        
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.backbone = WavLMModel.from_pretrained(self.model_name).to(self.device)
        self.backbone.eval()
        
        # Cargar el clasificador entrenado con scikit-learn
        if os.path.exists(self.classifier_path):
            self.classifier = joblib.load(self.classifier_path)
            print(f"[AcousticModel] Cargado clasificador entrenado desde {self.classifier_path}")
        else:
            self.classifier = None
            print("[AcousticModel Warning] No se encontró acoustic_classifier.pkl. Usando score base.")

    def predict(self, waveform: torch.Tensor, sample_rate: int = 16000) -> float:
        if waveform.numel() == 0 or waveform.shape[-1] < (sample_rate * 0.2):
            return 0.10

        try:
            inputs = waveform.squeeze(0).cpu().numpy()
            inputs_processed = self.feature_extractor(
                inputs, 
                sampling_rate=16000, 
                return_tensors="pt"
            ).input_values.to(self.device)

            with torch.no_grad():
                outputs = self.backbone(inputs_processed)
                embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()

            if self.classifier is not None:
                # Predecir probabilidad de la clase 1 (sintético)
                proba = self.classifier.predict_proba(embedding)[0][1]
                return float(proba)
            else:
                return 0.50

        except Exception as e:
            print(f"[AcousticModel Inference Error]: {e}")
            return 0.20

_detector_instance = None

def predict_acoustic(ch0_waveform: torch.Tensor) -> float:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AcousticDeepfakeDetector()
    return _detector_instance.predict(ch0_waveform)