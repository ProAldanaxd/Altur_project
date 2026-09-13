"""ElevenLabs alert generation is explicit and separate from /detect."""
import os
import re
from contextlib import nullcontext
from threading import BoundedSemaphore

import httpx

ALERT_TEXT_SYNTHETIC = "Atención. Se detectaron señales de voz sintética. Solicita una verificación adicional de identidad antes de continuar. Esta alerta no confirma fraude."
ALERT_TEXT_HUMAN = "Atención. La voz de quien llama fue clasificada como humana según el modelo. Este resultado no sustituye una verificación de identidad."
ALERT_TEXT = ALERT_TEXT_SYNTHETIC  # alias retrocompatible; usado por /voice/status
_slot = BoundedSemaphore(1)


def status():
    if not os.environ.get("ELEVENLABS_API_KEY"):
        return {"status": "unavailable", "reason": "missing_api_key"}
    if not re.fullmatch(r"[A-Za-z0-9_-]+", os.environ.get("ELEVENLABS_VOICE_ID", "")):
        return {"status": "unavailable", "reason": "missing_or_invalid_voice_id"}
    return {"status": "configured", "live_verified": False}


def generate_alert(*, client=None, is_synthetic):
    if is_synthetic:
        text = ALERT_TEXT_SYNTHETIC 
    else:
        text = ALERT_TEXT_HUMAN
    state = status()
    if state["status"] != "configured":
        return state
    if not _slot.acquire(blocking=False):
        return {"status": "unavailable", "reason": "voice_busy", "retryable": True}
    try:
        context = nullcontext(client) if client is not None else httpx.Client(timeout=20, trust_env=False)
        with context as transport:
            response = transport.post(
                "https://api.elevenlabs.io/v1/text-to-speech/" + os.environ["ELEVENLABS_VOICE_ID"],
                params={"output_format": "mp3_44100_128"},
                headers={"xi-api-key": os.environ["ELEVENLABS_API_KEY"], "Accept": "audio/mpeg"},
                json={"text": text, "model_id": os.environ.get("ELEVENLABS_MODEL_ID", "eleven_multilingual_v2")},
                timeout=20,
            )
        if response.status_code != 200:
            return {"status": "unavailable", "reason": f"provider_http_{response.status_code}",
                    "retryable": response.status_code == 429 or response.status_code >= 500}
        if "audio/" not in response.headers.get("content-type", "") or not 32 <= len(response.content) <= 2 * 1024 * 1024:
            return {"status": "unavailable", "reason": "invalid_audio_response"}
        return {"status": "ok", "audio": response.content, "content_type": "audio/mpeg", "text": text}
    except httpx.TimeoutException:
        return {"status": "unavailable", "reason": "provider_timeout", "retryable": True}
    except httpx.HTTPError:
        return {"status": "unavailable", "reason": "provider_connection_error", "retryable": True}
    finally:
        _slot.release()
