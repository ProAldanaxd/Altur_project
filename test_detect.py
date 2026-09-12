import base64
import requests

# 1. Crear un audio de prueba pequeño en memoria (WAV en silenciador/dummy)
# O bien puedes leer un archivo .wav real si tienes uno a la mano:
# with open("tu_audio_de_prueba.wav", "rb") as f:
#     audio_b64 = base64.b64encode(f.read()).decode("utf-8")

# Payload dummy de prueba con la estructura exacta de Altur
payload = {
    # Un header Base64 de prueba WAV
    "audio": "UklGRiQAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQAAAAA=",
    "turns": [
        {"channel": 1, "start": 0.0, "end": 1.0},
        {"channel": 0, "start": 1.0, "end": 2.0}
    ]
}

print("Enviando petición al backend local...")
url = "http://127.0.0.1:8000/detect"
response = requests.post(url, json=payload)

print("Status Code:", response.status_code)
print("Respuesta del servidor:", response.json())