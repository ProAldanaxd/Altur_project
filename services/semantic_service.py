import os
import io
import torch
import soundfile as sf
from google import genai
from google.genai import types
from faster_whisper import WhisperModel

stt_model = WhisperModel("tiny", device="cpu")

def transcribe_segment(waveform: torch.Tensor, sample_rate: int = 16000) -> str:
    if waveform.numel() == 0 or waveform.shape[-1] < (sample_rate * 0.1):
        return ""
    try:
        buffer = io.BytesIO()
        audio_np = waveform.squeeze(0).cpu().numpy()
        sf.write(buffer, audio_np, sample_rate, format='WAV')
        buffer.seek(0)
        segments, _ = stt_model.transcribe(buffer, beam_size=1)
        return " ".join([seg.text for seg in segments]).strip()
    except Exception:
        return ""

def predict_semantic(ch0_waveform: torch.Tensor, ch1_waveform: torch.Tensor) -> float:
    transcript_ch0 = transcribe_segment(ch0_waveform)
    transcript_ch1 = transcribe_segment(ch1_waveform)

    if not transcript_ch0 and not transcript_ch1:
        return 0.10

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return 0.10

    client = genai.Client(api_key=api_key)

    # 1. AGENTE EVALUADOR (Detecta anomalías y emite primera hipótesis)
    evaluator_prompt = f"""
    Eres un analista de seguridad telefónica. Analiza esta transcripción entre Agente (Ch 1) y Cliente (Ch 0).
    Identifica anomalías semánticas, respuestas automáticas, alucinaciones o comportamientos típicos de un bot sintético.

    [Agente - Ch 1]: {transcript_ch1}
    [Cliente - Ch 0]: {transcript_ch0}

    Devuelve un breve análisis de 2 oraciones y concluye con una sospecha de bot estimada entre 0.0 y 1.0.
    """

    try:
        eval_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=evaluator_prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )
        first_analysis = eval_response.text.strip()

        # 2. AGENTE CRÍTICO (Verifica y busca explicaciones alternativas de comportamiento humano)
        critic_prompt = f"""
        Eres un auditor crítico de control de calidad. Tu trabajo es evitar falsos positivos.
        Revisa la transcripción original y el primer análisis realizado por tu colega.

        [Transcripción Original]:
        Agente (Ch 1): {transcript_ch1}
        Cliente (Ch 0): {transcript_ch0}

        [Análisis del Evaluador]:
        {first_analysis}

        Instrucciones:
        1. Cuestiona la evaluación: ¿Podría esta respuesta pertenecer a un humano distraído, molesto o con acento?
        2. Determina la puntuación final refinada.

        Responde ÚNICAMENTE con el valor flotante final entre 0.0 (humano confirmado) y 1.0 (bot sintético confirmado).
        """

        critic_response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=critic_prompt,
            config=types.GenerateContentConfig(temperature=0.1)
        )

        final_score = float(critic_response.text.strip())
        return final_score

    except Exception:
        return 0.20