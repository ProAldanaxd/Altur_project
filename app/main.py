from contextlib import asynccontextmanager
from datetime import datetime, timezone
from uuid import uuid4
import os
import secrets
import sqlite3
from pathlib import Path
from time import perf_counter
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.responses import HTMLResponse
from starlette.concurrency import run_in_threadpool
from pydantic import AliasChoices, BaseModel, Field, model_validator

from app.audio import MAX_BASE64_CHARS, decode_channels
from app.model import Detector, InsufficientSpeechError
from app.limits import RequestSizeLimit
from conversation.temporal import analyze_channels, extract_model_features
from conversation.semantics import Turn, Trap, analyze_transcript, transcribe_channels, configuration_status
from ops.audit import AuditStore
from ops.voice import generate_alert, status as voice_status, ALERT_TEXT_SYNTHETIC, ALERT_TEXT_HUMAN


@asynccontextmanager
async def lifespan(app):
    app.state.detector = Detector()
    try:
        app.state.audit = AuditStore()
    except (OSError, RuntimeError, sqlite3.Error):
        app.state.audit = None
    try:
        yield
    finally:
        if app.state.audit is not None:
            await run_in_threadpool(app.state.audit.close)


app = FastAPI(title="Altur — detección de voz sintética", lifespan=lifespan)
app.add_middleware(RequestSizeLimit)


class DetectRequest(BaseModel):
    audio_base64: str = Field(min_length=1, max_length=MAX_BASE64_CHARS,
                             validation_alias=AliasChoices("audio", "audio_base64"))

    @model_validator(mode="before")
    @classmethod
    def reject_conflicting_audio(cls, data):
        if isinstance(data, dict) and "audio" in data and "audio_base64" in data:
            if data["audio"] != data["audio_base64"]:
                raise ValueError("audio y audio_base64 no pueden diferir")
        return data


class DetectResponse(BaseModel):
    is_synthetic: bool = Field(strict=True)
    confidence: Optional[float] = Field(default=None, ge=0, le=1, allow_inf_nan=False)


class ConversationRequest(DetectRequest):
    include_semantics: bool = False
    transcript: Optional[List[Turn]] = Field(default=None, max_length=400)
    traps: List[Trap] = Field(default_factory=list, max_length=40)


class SemanticRequest(BaseModel):
    turns: List[Turn] = Field(min_length=1, max_length=400)
    traps: List[Trap] = Field(default_factory=list, max_length=40)


@app.get("/conversation/status")
def conversation_status():
    return {"temporal": "ready", "schema_version": "temporal-v1",
            "semantic": configuration_status(), "affects_detect": False}


@app.post("/conversation/semantic")
def semantic_endpoint(payload: SemanticRequest):
    try:
        return analyze_transcript([t.model_dump() for t in payload.turns],
                                  [t.model_dump() for t in payload.traps])
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.post("/conversation/analyze")
def conversation_endpoint(payload: ConversationRequest):
    try:
        caller, agent, sr = decode_channels(payload.audio_base64)
        temporal = analyze_channels(caller, agent, sr)
        semantic = {"status": "not_requested", "affects_verdict": False}
        transcription = None
        if payload.include_semantics:
            transcript = [t.model_dump() for t in payload.transcript] if payload.transcript else None
            if transcript is not None and any(t["end"] > len(caller) / sr for t in transcript):
                raise ValueError("La transcripción excede la duración del audio")
            if transcript is None:
                if payload.traps:
                    raise ValueError("Los índices de trampas requieren una transcripción proporcionada")
                transcription = transcribe_channels(caller, agent, sr)
                if transcription["status"] == "ok":
                    transcript = transcription["turns"]
                else:
                    semantic = transcription
            if transcript:
                semantic = analyze_transcript(transcript, [t.model_dump() for t in payload.traps])
            elif transcription and transcription["status"] == "ok":
                semantic = {"status": "insufficient_transcript", "affects_verdict": False}
        return {"temporal": temporal, "features": extract_model_features(temporal),
                "semantic": semantic, "transcription": transcription, "affects_detect": False}
    except ValueError as error:
        raise HTTPException(422, str(error)) from error


@app.middleware("http")
async def timing(request, call_next):
    started = perf_counter()
    request_id = str(uuid4())
    status_code = 500
    try:
        response = await call_next(request)
        status_code = response.status_code
    finally:
        elapsed = (perf_counter() - started) * 1000
        audit = getattr(request.app.state, "audit", None)
        if request.url.path == "/detect" and request.method == "POST" and audit is not None:
            detector = request.app.state.detector
            result = getattr(request.state, "detection", {})
            audit.enqueue({"request_id": request_id, "created_at": datetime.now(timezone.utc).isoformat(),
                "status_code": status_code, "is_synthetic": result.get("is_synthetic") if status_code == 200 else None,
                "p_synthetic": result.get("p_synthetic") if status_code == 200 else None,
                "latency_ms": elapsed, "duration_s": getattr(request.state, "duration_s", None),
                "model_variant": getattr(detector, "variant", "test"), "model_sha256": getattr(detector, "sha256", None)})
    response.headers["X-Process-Time-Ms"] = f"{elapsed:.2f}"
    response.headers["X-Model-Variant"] = getattr(request.app.state.detector, "variant", "test")
    response.headers["X-Request-ID"] = request_id
    return response


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/ready")
def ready(request: Request):
    if not request.app.state.detector.ready:
        raise HTTPException(503, "Falta integrar el modelo de Dev 2")
    return {"status": "ready", "model": request.app.state.detector.variant,
            "sha256": request.app.state.detector.sha256}


@app.get("/model")
def model_info(request: Request):
    detector = request.app.state.detector
    if not detector.ready:
        raise HTTPException(503, "Modelo no disponible")
    return {"variant": detector.variant, "sha256": detector.sha256,
            "feature_count": len(detector.columns),
            "training_source": detector.metadata["training_source"],
            "holdout_preserved": detector.metadata["holdout_preserved"]}


@app.post("/detect", response_model=DetectResponse, response_model_exclude_none=True)
def detect(payload: DetectRequest, request: Request):
    try:
        client, agent, sample_rate = decode_channels(payload.audio_base64)
        request.state.duration_s = len(client) / sample_rate
    except ValueError as error:
        raise HTTPException(422, str(error)) from error
    detector = request.app.state.detector
    if not detector.ready:
        raise HTTPException(503, "Falta integrar el modelo de Dev 2")
    try:
        if hasattr(detector, "predict_details"):
            result = detector.predict_details(client, agent, sample_rate)
        else:
            result = detector.predict(client, agent, sample_rate)
        request.state.detection = result
        return {key: value for key, value in result.items() if key in {"is_synthetic", "confidence"}}
    except InsufficientSpeechError as error:
        raise HTTPException(422, str(error)) from error


def require_operator(request: Request):
    token = os.environ.get("ADMIN_TOKEN")
    if token:
        if not secrets.compare_digest(request.headers.get("Authorization", ""), "Bearer " + token):
            raise HTTPException(401, "Se requiere token de operador")
    elif not request.client or request.client.host not in {"127.0.0.1", "::1", "testclient"}:
        raise HTTPException(403, "Configura ADMIN_TOKEN para acceso remoto")


@app.get("/audit/stats", dependencies=[Depends(require_operator)])
def audit_stats(request: Request):
    if request.app.state.audit is None:
        raise HTTPException(503, "Auditoría no disponible")
    return request.app.state.audit.stats()


@app.get("/audit/calls", dependencies=[Depends(require_operator)])
def audit_calls(request: Request, limit: int = 50):
    if request.app.state.audit is None:
        raise HTTPException(503, "Auditoría no disponible")
    return {"calls": request.app.state.audit.recent(limit)}


@app.get("/voice/status", dependencies=[Depends(require_operator)])
def alert_status():
    return {**voice_status(), "text_synthetic": ALERT_TEXT_SYNTHETIC, "text_human": ALERT_TEXT_HUMAN,
            "automatic_outbound_calls": False}


class VoiceRequest(BaseModel):
    demo: bool
    is_synthetic: bool = True


@app.post("/voice/alert", dependencies=[Depends(require_operator)])
def voice_alert(payload: VoiceRequest):
    if not payload.demo:
        raise HTTPException(422, "Este endpoint genera la alerta para la demo")
    result = generate_alert(is_synthetic=payload.is_synthetic)
    if result["status"] != "ok":
        raise HTTPException(503, result)
    return Response(result["audio"], media_type="audio/mpeg", headers={"Cache-Control": "no-store"})


@app.get("/demo", response_class=HTMLResponse)
def demo_page():
    # read_text() sin encoding usa la codificación local del sistema (cp1252 en
    # Windows), lo que corrompe los acentos UTF-8 del HTML. Forzar utf-8 explícito.
    return (Path(__file__).resolve().parents[1] / "ops/demo.html").read_text(encoding="utf-8")
