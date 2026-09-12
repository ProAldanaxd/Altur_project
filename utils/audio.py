import base64
import io
import torch
import soundfile as sf

def process_audio_channels(base64_string: str, turns: list):
    """
    Decodifica el audio Base64 directamente con soundfile a Tensor sin depender de torchaudio.load.
    """
    raw_bytes = base64.b64decode(base64_string)
    
    # Decodificar directo a numpy array usando soundfile
    data, sample_rate = sf.read(io.BytesIO(raw_bytes), dtype='float32')
    
    # Convertir a Tensor de PyTorch (Canales x Muestras)
    # soundfile entrega formato (muestras, canales)
    tensor_data = torch.from_numpy(data)
    if tensor_data.ndim == 1:
        waveform = tensor_data.unsqueeze(0)  # Convertir (Muestras,) a (1, Muestras)
    else:
        waveform = tensor_data.T  # Transponer a (Canales, Muestras)
    
    # Asegurar mono si tiene múltiples canales físicos
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    ch0_segments = []
    ch1_segments = []

    for turn in turns:
        channel = turn.get("channel")
        start_sec = turn.get("start", 0)
        end_sec = turn.get("end", 0)

        start_sample = int(start_sec * sample_rate)
        end_sample = int(end_sec * sample_rate)

        segment = waveform[:, start_sample:end_sample]

        if channel == 0:
            ch0_segments.append(segment)
        elif channel == 1:
            ch1_segments.append(segment)

    ch0_combined = torch.cat(ch0_segments, dim=1) if ch0_segments else torch.zeros((1, sample_rate))
    ch1_combined = torch.cat(ch1_segments, dim=1) if ch1_segments else torch.zeros((1, sample_rate))

    return ch0_combined, ch1_combined