# Dev 1 — API de detección

Base inicial para conectar el modelo de Dev 2. No incluye un clasificador ni está desplegada. Ajustada al README público de Altur; el cuerpo exacto de la solicitud sigue pendiente porque el repositorio no lo especifica.

## Arrancar en Mac

Desde esta carpeta, ejecutar una línea a la vez:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m uvicorn app.main:app --reload
```

Abrir http://127.0.0.1:8000/docs para ver y probar la API. La terminal permanece ocupada mientras corre el servidor; Ctrl+C lo detiene.

## Contrato revisado con Altur

Nuestra API usa provisionalmente el campo `audio_base64`: POST /detect recibe `{"audio_base64": "<base64 de un archivo WAV completo>"}`. WAV PCM de 16 bits, estéreo de 8000 Hz, canal 0 cliente, canal 1 agente. Límite local: 12 MiB decodificados (aproximadamente 393 segundos de PCM16 estéreo de 8 kHz, menos cabecera). Se eliminó el límite de 180 segundos: el manifest incluye llamadas de hasta 273 segundos. Este límite de bytes no es un requisito de Altur y deberá ajustarse si los jueces envían clips más largos. No admite PCM crudo ni prefijos data URI. El README oficial confirma el audio base64 y los canales, pero no publica el nombre del campo de entrada, timeout, concurrencia ni límites. Confirmarlos con Altur.

Con modelo integrado, responde `{"is_synthetic": true, "confidence": 0.9}`. Solo `is_synthetic` es obligatorio. `confidence` puede omitirse y entonces no aparece en la respuesta. Altur lo usa para desempates y calibración, pero no especifica si representa probabilidad de voz sintética o confianza en el veredicto. Confirmar antes de integrarlo. La duración del procesamiento se expone en `X-Process-Time-Ms`; no mide latencia de red.

Sin modelo, un audio válido devuelve HTTP 503. Entradas inválidas devuelven 422. GET /health comprueba que la API vive; GET /ready devuelve 503 hasta conectar el modelo. No hay predicciones inventadas para la demo.

## Enviar audio real

En otra terminal, con el entorno virtual activado, sustituir llamada.wav por la ruta del archivo:

```bash
python -c 'import base64,json,pathlib; print(json.dumps({"audio_base64":base64.b64encode(pathlib.Path("llamada.wav").read_bytes()).decode()}))' > /tmp/altur-request.json
curl -i http://127.0.0.1:8000/detect -H 'Content-Type: application/json' --data-binary @/tmp/altur-request.json
```

## Acuerdo con Dev 2 y Dev 3

Dev 2 implementa `Detector` en app/model.py: carga pesos una vez en `__init__`, establece `ready = True` después de cargarlos y devuelve is_synthetic y, opcionalmente, confidence desde `predict(client, agent, sample_rate)`. Cada canal es un array NumPy mono float32. El predictor debe soportar llamadas concurrentes o proteger el acceso al modelo. Dev 3 entrega sus características del canal agente para integrarlas en ese predictor.

## Pruebas

```bash
python -m pytest -q
```

Verifican errores de entrada, orden de canales, audio de 273 segundos, confidence opcional y conexión del predictor usando un doble de prueba. No evalúan precisión del modelo ni rendimiento del benchmark.

## Preparación de despliegue

Con Docker instalado:

```bash
docker build -t altur-api .
docker run --name altur-api --restart unless-stopped -p 8000:8000 -d altur-api
```

El Dockerfile usa un proceso Uvicorn, sin recarga automática, y usuario sin privilegios. Es una base para un servidor como Vultr; no aprovisiona nube, DNS ni TLS. Antes de publicar: integrar pesos y dependencias del modelo, fijar versiones resueltas, configurar HTTPS y un límite de cuerpo HTTP en el proxy (el límite del esquema se aplica después de recibir el JSON), verificar /ready y medir p50/p95 con audios reales y la concurrencia oficial. Ajustar límites de duración y concurrencia al benchmark.

Referencias: https://fastapi.tiangolo.com/deployment/docker/ y https://python-soundfile.readthedocs.io/en/0.13.1/

## Material oficial y siguientes pasos

Fuente: https://github.com/alturio/hackmty26

- El manifest contiene etiquetas y splits train/val; no mezclar hablantes entre entrenamiento y validación. El conjunto oculto tiene voces y personas nuevas.
- Los JSON de turns contienen segmentos por canal; Dev 3 puede usarlos como punto de partida. El endpoint debe funcionar solo con el audio recibido: el README no promete turns durante la evaluación.
- Los audios se distribuyen aparte en altur-challenge-audio.zip dentro de Releases: https://github.com/alturio/hackmty26/releases/tag/v1.0 . Clonar el repositorio por sí solo no descarga los WAV.
- Uso del dataset limitado a HackMTY 2026; no redistribuir. Mantener audios y datos fuera del repositorio de la API.

Pregunta concreta para los organizadores: “¿Nos comparten un curl real a /detect con el JSON de entrada, el timeout, la concurrencia, el tamaño máximo y la definición de confidence?”

Después: conectar el predictor de Dev 2, probar audios oficiales y desplegar. Esta base todavía no demuestra precisión, latencia del benchmark ni uptime en nube.
