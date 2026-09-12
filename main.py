from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

from utils.audio import process_audio_channels
from models.acoustic import predict_acoustic
from services.semantic_service import predict_semantic

app = FastAPI()

class Turn(BaseModel):
    channel: int
    start: float
    end: float

class DetectPayload(BaseModel):
    audio: str 
    turns: List[Turn]

@app.post("/detect")
async def detect(payload: DetectPayload):
    try:
        turns_dict = [turn.model_dump() for turn in payload.turns]
        ch0, ch1 = process_audio_channels(payload.audio, turns_dict)

        score_acoust = predict_acoustic(ch0)
        score_sem = predict_semantic(ch0,ch1)

        final_confidence = (score_acoust * 0.6) + (score_sem *0.4)
        is_synt = final_confidence > 0.50

        return {
            "is_synthetic": bool(is_synt),
            "confidence": float(final_confidence)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
