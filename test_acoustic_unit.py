# test_acoustic_unit.py
import torch
from models.acoustic import predict_acoustic

print("Inicializando detector acústico y cargando modelo...")
# Generar un tensor de prueba (2 segundos de señal a 16kHz)
dummy_tensor = torch.randn((1, 32000))

score = predict_acoustic(dummy_tensor)
print(f"✅ Inferencia exitosa. Probabilidad acústica de Fake: {score:.4f}")