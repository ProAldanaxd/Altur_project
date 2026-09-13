"""Run one fictional semantic check using the current terminal's credentials."""
import json
import os
import re
from pathlib import Path

import httpx

from dev3.semantics import analyze_transcript, configuration_status


def sanitize(value):
    message = str(value)
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY"):
        secret = os.environ.get(name)
        if secret:
            message = message.replace(secret, "[CLAVE OCULTA]")
    message = re.sub(r"AIza[\w-]+|AQ\.[\w.-]+", "[CLAVE OCULTA]", message)
    return message[:1200]


class DiagnosticClient:
    def __init__(self, client):
        self.client = client

    def post(self, *args, **kwargs):
        response = self.client.post(*args, **kwargs)
        print("HTTP de Gemini:", response.status_code)
        if response.status_code != 200:
            try:
                error = response.json().get("error", {})
                print("Estado:", sanitize(error.get("status", "unknown")))
                print("Detalle:", sanitize(error.get("message", "Sin detalle")))
            except (ValueError, AttributeError):
                print("El proveedor no devolvió un error JSON legible.")
        return response


def isolate(client, config):
    """Stop at the first rejected stage; no model switching or real call data."""
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    headers = {"x-goog-api-key": key}
    root = "https://generativelanguage.googleapis.com/v1beta/models/"
    print("Etapa 1: consultar el modelo configurado", flush=True)
    response = client.get(root + config["model"], headers=headers)
    print("HTTP modelo:", response.status_code, flush=True)
    if response.status_code != 200:
        try:
            print("Detalle:", sanitize(response.json().get("error", {})))
        except ValueError:
            print("Detalle: respuesta no JSON")
        print("No se pudo consultar el modelo; no se enviaron solicitudes de generación.")
        return
    info = response.json()
    print("Métodos:", sanitize(info.get("supportedGenerationMethods", [])))
    url = root + config["model"] + ":generateContent"
    transport = DiagnosticClient(client)
    body = {"contents": [{"role": "user", "parts": [{"text": "Responde solamente OK."}]}]}
    print("Etapa 2: texto mínimo, sin esquema ni parámetros de generación", flush=True)
    response = transport.post(url, headers=headers, json=body)
    if response.status_code != 200:
        print("La petición mínima también falla. El esquema de Altur no es necesario para reproducir el error.")
        return
    print("Etapa 3: JSON con un esquema sencillo", flush=True)
    body["contents"][0]["parts"][0]["text"] = 'Devuelve un objeto JSON con ok igual a true.'
    body["generationConfig"] = {
        "responseMimeType": "application/json",
        "responseJsonSchema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]},
    }
    response = transport.post(url, headers=headers, json=body)
    if response.status_code != 200:
        print("Texto mínimo aceptado; el fallo aparece al solicitar JSON estructurado.")
        return
    print("Etapa 4: análisis completo del diálogo ficticio de Altur", flush=True)
    payload = json.loads((Path(__file__).parent / "examples/semantic_request.json").read_text())
    result = analyze_transcript(payload["turns"], payload["traps"], client=transport)
    print("Resultado:", sanitize(json.dumps(result, ensure_ascii=False)))


if __name__ == "__main__":
    config = configuration_status()
    print("Configuración:", json.dumps(config, ensure_ascii=False), flush=True)
    if config["status"] == "configured":
        try:
            with httpx.Client(timeout=30, trust_env=False) as client:
                isolate(client, config)
        except httpx.TimeoutException:
            print("Resultado: timeout del proveedor en la etapa indicada. No se hicieron reintentos.")
        except httpx.HTTPError:
            print("Resultado: error de conexión con el proveedor. No se hicieron reintentos.")
    else:
        print("Ejecuta esta prueba en la misma Terminal donde exportaste las variables de Gemini.")
