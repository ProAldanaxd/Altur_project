import base64
import binascii
import io

import numpy as np
import soundfile as sf

MAX_AUDIO_BYTES = 12 * 1024 * 1024
MAX_BASE64_CHARS = 4 * ((MAX_AUDIO_BYTES + 2) // 3)
# Límite local de recursos; Altur no publica un máximo de duración.
MAX_FRAMES = MAX_AUDIO_BYTES // (2 * 2)  # Dos canales, PCM de 16 bits.


def decode_channels(encoded: str):
    """Formato del dataset oficial: WAV PCM16 estéreo, 8 kHz."""
    if len(encoded) > MAX_BASE64_CHARS:
        raise ValueError("Audio demasiado grande")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise ValueError("audio_base64 debe contener base64 válido") from None
    if len(raw) > MAX_AUDIO_BYTES:
        raise ValueError("Audio demasiado grande")
    try:
        with sf.SoundFile(io.BytesIO(raw)) as audio:
            if audio.format != "WAV":
                raise ValueError("Se requiere un archivo WAV")
            if audio.subtype != "PCM_16":
                raise ValueError("Se requiere PCM de 16 bits")
            if audio.channels != 2:
                raise ValueError("Se requieren dos canales: cliente=0, agente=1")
            if audio.samplerate != 8000:
                raise ValueError("Se requieren 8000 Hz; no se remuestrea automáticamente")
            if not 0 < audio.frames <= MAX_FRAMES:
                raise ValueError("Audio vacío o demasiado grande")
            samples = audio.read(dtype="float32", always_2d=True)
    except (sf.LibsndfileError, RuntimeError):
        raise ValueError("No se pudo decodificar el archivo WAV") from None
    if samples.shape[0] == 0 or not np.isfinite(samples).all():
        raise ValueError("Audio vacío o con muestras no finitas")
    return samples[:, 0].copy(), samples[:, 1].copy(), 8000
