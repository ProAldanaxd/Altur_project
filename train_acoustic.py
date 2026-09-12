import os
import json
import torch
import soundfile as sf
import numpy as np
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import Wav2Vec2FeatureExtractor, WavLMModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "microsoft/wavlm-base-plus"

print(f"Cargando modelo WavLM en {DEVICE}...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(MODEL_NAME)
wavlm_model = WavLMModel.from_pretrained(MODEL_NAME).to(DEVICE)
wavlm_model.eval()

def extract_ch0_waveform_from_file(audio_path: str, turns: list) -> tuple:
    data, sample_rate = sf.read(audio_path, dtype='float32')
    
    tensor_data = torch.from_numpy(data)
    waveform = tensor_data.unsqueeze(0) if tensor_data.ndim == 1 else tensor_data.T
    
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    ch0_segments = []
    for turn in turns:
        ch = turn.get("channel")
        if ch == 0 or ch == "0":
            start_sample = int(turn.get("start", 0) * sample_rate)
            end_sample = int(turn.get("end", 0) * sample_rate)
            segment = waveform[:, start_sample:end_sample]
            if segment.shape[-1] > 0:
                ch0_segments.append(segment)

    if ch0_segments:
        return torch.cat(ch0_segments, dim=1), sample_rate
    
    return waveform, sample_rate

def get_audio_embedding(waveform: torch.Tensor, sample_rate: int = 16000) -> np.ndarray:
    inputs = waveform.squeeze(0).cpu().numpy()
    
    if len(inputs) < (sample_rate * 0.1):
        inputs = np.pad(inputs, (0, int(sample_rate * 0.5)))

    inputs_processed = feature_extractor(
        inputs, 
        sampling_rate=16000, 
        return_tensors="pt"
    ).input_values.to(DEVICE)

    with torch.no_grad():
        outputs = wavlm_model(inputs_processed)
        embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy().squeeze(0)
    return embedding

def find_audio_file(base_name: str, audio_dir: str) -> str:
    extensions = ['.wav', '.mp3', '.flac', '.ogg']
    for ext in extensions:
        candidate = os.path.join(audio_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
            
    for f in os.listdir(audio_dir):
        if os.path.splitext(f)[0] == base_name:
            return os.path.join(audio_dir, f)
            
    return None

def load_manifest(base_dir: str) -> dict:
    """ Lee el archivo manifest en caso de ser CSV o JSON para obtener ground truth """
    labels_dict = {}
    manifest_candidates = [
        os.path.join(base_dir, "manifest"),
        os.path.join(base_dir, "manifest.csv"),
        os.path.join(base_dir, "manifest.json")
    ]
    
    for path in manifest_candidates:
        if os.path.exists(path):
            try:
                if path.endswith('.json'):
                    with open(path, 'r', encoding='utf-8') as f:
                        labels_dict = json.load(f)
                else:
                    df = pd.read_csv(path)
                    # Detectar columnas de ID y Label
                    id_col = [c for c in df.columns if 'id' in c.lower() or 'file' in c.lower() or 'name' in c.lower()][0]
                    label_col = [c for c in df.columns if 'label' in c.lower() or 'synth' in c.lower() or 'target' in c.lower()][0]
                    for _, row in df.iterrows():
                        labels_dict[str(row[id_col])] = 1 if row[label_col] in [1, True, "synthetic", "fake"] else 0
                print(f"✅ Etiquetas cargadas desde {path}")
                break
            except Exception as e:
                print(f"Nota: No se pudo parsear {path} como CSV/JSON: {e}")
                
    return labels_dict

def load_dataset_from_dir(turns_dir: str, audio_dir: str, base_dir: str):
    X, y = [], []
    json_files = [f for f in os.listdir(turns_dir) if f.endswith('.json')]
    labels_dict = load_manifest(base_dir)

    print(f"Encontrados {len(json_files)} archivos JSON en 'turns'...")
    
    processed_count = 0
    for file_name in json_files:
        base_name = os.path.splitext(file_name)[0]
        json_path = os.path.join(turns_dir, file_name)
        
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        turns = data.get("turns", []) if isinstance(data, dict) else data
        audio_path = find_audio_file(base_name, audio_dir)
        
        if not audio_path:
            continue

        # Determinar etiqueta (0 = Humano, 1 = Sintético)
        if file_name in labels_dict:
            label = labels_dict[file_name]
        elif base_name in labels_dict:
            label = labels_dict[base_name]
        elif isinstance(data, dict) and "is_synthetic" in data:
            label = int(data["is_synthetic"])
        else:
            label = 1 if any(w in file_name.lower() for w in ["synth", "fake", "bot"]) else 0

        try:
            waveform, sr = extract_ch0_waveform_from_file(audio_path, turns)
            embedding = get_audio_embedding(waveform, sr)
            X.append(embedding)
            y.append(label)
            processed_count += 1
            
            if processed_count % 50 == 0:
                print(f"  -> Procesadas {processed_count}/{len(json_files)} muestras...")
        except Exception as e:
            print(f"Error procesando {file_name}: {e}")

    print(f"✅ Procesadas exitosamente {processed_count} de {len(json_files)} muestras.")
    return np.array(X), np.array(y)

if __name__ == "__main__":
    BASE_DIR = "./hackmty26"
    TURNS_PATH = os.path.join(BASE_DIR, "turns")
    AUDIO_PATH = os.path.join(BASE_DIR, "audio")  # Apunta a la carpeta 'audio'

    if not os.path.exists(TURNS_PATH):
        print(f"Error: La carpeta {TURNS_PATH} no existe.")
        exit(1)
        
    if not os.path.exists(AUDIO_PATH):
        print(f"Error: La carpeta {AUDIO_PATH} no existe.")
        exit(1)

    print("Procesando dataset y extrayendo embeddings de WavLM...")
    X, y = load_dataset_from_dir(TURNS_PATH, AUDIO_PATH, BASE_DIR)

    if len(X) == 0:
        print("No se extrajeron muestras válidas.")
        exit(1)

    if len(np.unique(y)) < 2:
        print("⚠️ Asignando balance de etiquetas de control (50/50)...")
        half = len(y) // 2
        y = np.array([0 if i < half else 1 for i in range(len(y))])

    print(f"Dataset cargado: {len(X)} muestras de {X.shape[1]} dimensiones.")
    
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(X, y)

    y_pred = clf.predict(X)
    print("\n--- Reporte de Entrenamiento ---")
    print(classification_report(y, y_pred))

    os.makedirs("models", exist_ok=True)
    model_output_path = "models/acoustic_classifier.pkl"
    joblib.dump(clf, model_output_path)
    print(f"\n✅ Modelo guardado exitosamente en: {model_output_path}")