import base64
import json
import os
import requests

# 1. Rutas del dataset local
DATASET_TURNS = "./hackmty26/turns"
DATASET_AUDIO = "./hackmty26/audio"

# Buscar el primer JSON disponible
json_files = [f for f in os.listdir(DATASET_TURNS) if f.endswith('.json')]
if not json_files:
    print(f"Error: No se encontraron archivos JSON en {DATASET_TURNS}")
    exit(1)

target_json = json_files[0]
base_name = os.path.splitext(target_json)[0]

# 2. Cargar turnos
json_path = os.path.join(DATASET_TURNS, target_json)
with open(json_path, 'r', encoding='utf-8') as f:
    turns_data = json.load(f)

turns = turns_data.get("turns", []) if isinstance(turns_data, dict) else turns_data

# 3. Buscar y codificar el archivo de audio correspondiente en Base64
audio_extensions = ['.wav', '.mp3', '.flac', '.ogg']
audio_path = None
for ext in audio_extensions:
    candidate = os.path.join(DATASET_AUDIO, base_name + ext)
    if os.path.exists(candidate):
        audio_path = candidate
        break

if not audio_path:
    print(f"Error: No se encontró el archivo de audio para {base_name} en {DATASET_AUDIO}")
    exit(1)

with open(audio_path, 'rb') as af:
    b64_audio = base64.b64encode(af.read()).decode('utf-8')

# 4. Construir el payload con el esquema esperado por la API
payload = {
    "audio": b64_audio,
    "turns": turns
}

# 5. Enviar petición al servidor FastAPI
url = "http://127.0.0.1:8000/detect"
print(f"Probando endpoint {url} con la muestra: {base_name}...")

try:
    response = requests.post(url, json=payload)
    print("\n--- Respuesta del Servidor ---")
    print(f"Status Code: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
except Exception as e:
    print(f"\nError al conectar con la API: {e}")
    print("Asegúrate de que uvicorn esté corriendo en otra terminal con: python -m uvicorn main:app --reload")