# Altur — Dev 1 + Dev 2 + Dev 3 + Dev 4

Guía vigente del proyecto. Los documentos `DEV1-DEV2-REFERENCIA.md`, `INTEGRACION.md`, `ALFA-REVISION.md`, `DEV3-ENTREGA.md` y `DEV4-ENTREGA.md` describen el estado de cada entrega histórica (con sus propios puertos y comandos de esa época) y se conservan sin modificar como evidencia; para arrancar y operar el proyecto hoy, usa este README. `REVISION-TECNICA.md`, `CAMBIOS-Y-VALIDACION.md` y `BENCHMARK-ANTES-DESPUES.md` documentan la revisión más reciente.

Dev 3 temporal está implementado y probado con WAV oficiales. La conexión semántica Gemini está implementada y probada con un proveedor simulado; falta API key/modelo habilitado y prueba real — **pendiente por decisión del usuario**, no se busca ni configura en esta fase. No es un detector semántico validado.

## Nuestro enfoque (para jueces y evaluación)

El clasificador desplegado (`models/model.pkl`) **no es un clasificador acústico "de librería"** sobre espectro/MFCC/prosodia. Sus 87 características vienen enteramente de la **dinámica de turnos de la conversación**: cuánto dura cada intervención, cuánto tarda el cliente en responder al agente (con signo), cuánto se solapan, cuántos silencios hay, la entropía de esos patrones. Es el enfoque de "comportamiento conversacional" del reto — cómo reacciona quien llama a las interrupciones y silencios del agente, no cómo suena su voz — combinado con un ensemble simple (regresión logística + random forest + gradient boosting), sin deep learning ni espectrogramas.

- **Profundidad antes que bulto:** probamos agregar 26 características temporales adicionales (`conversation/temporal.py`, turnos con latencias firmadas, solapamientos, reinicios) y las **rechazamos con evidencia**: mejora de AUC de solo 0.0015 en CV de 5 folds sobre train, por debajo del umbral de 0.005 fijado *antes* de medir. La misma disciplina se aplicó esta semana con una corrección real al cálculo de latencias (`latency_pairing="signed_v2"`, ver `REVISION-TECNICA.md` hallazgo #6): probada contra el dataset oficial completo, dio el mismo balanced accuracy y solo +0.0008 de AUC — tampoco se promovió a producción. No agregamos señales que no demuestren aportar.
- **Viable en producción:** sin GPU ni modelo pesado — CPU, ensemble de scikit-learn, 100-150 ms por llamada en promedio (máximo observado 228 ms) medido con `scripts/check_endpoint.py`, el mismo script del juez, muy por debajo del límite de 30 s.
- **Enfoque semántico, listo pero no forzado:** `conversation/semantics.py` implementa el tercer enfoque del reto (detectar si quien llama inventa respuestas sobre datos que no existen) vía Gemini, probado exhaustivamente con proveedor simulado; nunca afirma que algo "no existe" sin una premisa de prueba explícita. Pendiente solo de credenciales reales, por decisión del equipo — no de código.
- **Honestos sobre lo que no sabemos:** no hay forma de medir el desempeño real contra el conjunto oculto (voces y personas nuevas) hasta la evaluación en vivo. La única evidencia disponible es sobre las 71 llamadas de validación (hablantes fuera de train): `balanced_accuracy: 0.945`, `auc: 0.980`, verificado con el script oficial del juez, no solo con herramientas propias (ver `CAMBIOS-Y-VALIDACION.md`).

## Qué funciona hoy sin ninguna cuenta externa

- `POST /detect`: clasificación local (baseline o alfa), sin llamar a ningún proveedor.
- `POST /conversation/analyze` y `POST /conversation/semantic` (sin `include_semantics`): análisis temporal 100% local.
- Auditoría SQLite local (`/audit/stats`, `/audit/calls`) y panel `/demo`.
- Todo lo anterior está cubierto por la suite de pruebas (`pytest`), que no requiere red ni credenciales.

## Qué está implementado pero no verificado con una cuenta real

- Gemini (`conversation/semantics.py`): adaptador probado con proveedor simulado (mocks). Pendiente por decisión del usuario.
- ElevenLabs (`ops/voice.py`): **verificado con cuenta real** (generación de MP3 real exitosa, ver `DEV4-ENTREGA.md`). Requiere `ELEVENLABS_API_KEY`/`ELEVENLABS_VOICE_ID` propios configurados en el entorno para reproducirlo.
- PostgreSQL/Tiger Data (`ops/postgres.py`): exportación idempotente probada con conexión simulada. Falta `DATABASE_URL` real.
- Docker/Linux: la imagen está definida (`Dockerfile`, Python 3.12) pero no se ha construido ni probado en este entorno (no hay Docker instalado en la máquina donde se hizo la última revisión).

## Arrancar

Desde la raíz del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8025
```

Abrir http://127.0.0.1:8025/docs y http://127.0.0.1:8025/demo. El modelo baseline de Dev 2 es el predeterminado. Alfa sigue seleccionable con `MODEL_VARIANT=alfa` (ver `ALFA-REVISION.md`: alfa no supera a baseline en la comparación disponible y no se recomienda como predeterminado).

Para arrancar sin ninguna variable de entorno externa heredada de la terminal (recomendado para reproducir la línea base):

```bash
env -u GEMINI_API_KEY -u GOOGLE_API_KEY -u GEMINI_MODEL \
    -u ELEVENLABS_API_KEY -u ELEVENLABS_VOICE_ID -u DATABASE_URL \
    -u MODEL_PATH \
    MODEL_VARIANT=baseline AUDIT_DB_PATH=work/audit-test.sqlite3 \
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8025
```

`.env.example` documenta las variables aceptadas; **la aplicación no carga `.env` automáticamente**, hay que exportarlas en la terminal.

## API

| Ruta | Método | Entrada | Resultado |
| --- | --- | --- | --- |
| `/health` | GET | — | Proceso vivo |
| `/ready` | GET | — | 200 solo si el modelo cargó, verificó identidad/versión/columnas/clases y pasó una predicción numérica de prueba |
| `/model` | GET | — | Variante, SHA-256, número de features, origen del entrenamiento, si conservó holdout |
| `/detect` | POST | WAV base64 en `audio` o `audio_base64` | `{"is_synthetic": true|false}` |
| `/conversation/status` | GET | — | Disponibilidad temporal y estado de configuración Gemini |
| `/conversation/analyze` | POST | Igual que `/detect`, más `include_semantics` opcional | Turnos, silencios, solapamientos, latencias con signo, 26 features, `affects_detect: false` |
| `/conversation/semantic` | POST | `turns` transcritos y `traps` opcionales | Reacciones con citas e índices verificados |
| `/audit/stats`, `/audit/calls` | GET | — | Requiere token de operador o loopback |
| `/voice/status`, `/voice/alert` | GET/POST | — | Requiere token de operador o loopback |
| `/demo` | GET | — | Panel HTML de la demo |

### Entrada de `/detect`

```json
{"audio": "BASE64_DEL_WAV_COMPLETO"}
```

WAV estéreo PCM16 de 8 kHz: canal 0 cliente, canal 1 agente. No se remuestrea ni recorta silencios. `audio` y `audio_base64` son alias del mismo campo; si se envían ambos deben coincidir byte a byte o la solicitud se rechaza con 422.

### Respuesta

```json
{"is_synthetic": true, "confidence": 0.89}
```

`confidence` (0.0–1.0) es la **confianza en el veredicto devuelto** — no la probabilidad cruda de que sea sintético: si `is_synthetic=true`, es `P(sintético)`; si `is_synthetic=false`, es `P(humano) = 1 - P(sintético)`. Así lo espera el contrato oficial de Altur (`scripts/check_endpoint.py` del juez reconstruye `P(sintético)` con esa misma fórmula). El umbral interno es 0.5 sobre la probabilidad cruda del modelo, que **no es una garantía de fraude**.

### Errores

| Código | Causa |
| --- | --- |
| 422 | WAV inválido, formato incorrecto, canal/tasa de muestreo distinta a la esperada, sin habla detectable en el canal del cliente, o `audio`/`audio_base64` contradictorios |
| 413 | Cuerpo de la solicitud excede el límite (≈16 MiB antes de decodificar JSON, 12 MiB de WAV ya decodificado) |
| 503 | Falta el modelo o la auditoría no está disponible (para las rutas de auditoría; `/detect` sigue funcionando aunque falle la auditoría) |

Un error nunca produce un veredicto ficticio: no hay una ruta que devuelva `is_synthetic`/`confidence` por defecto ante una excepción.

### Ejemplo completo (Python)

```python
import base64, json, pathlib, httpx

wav_bytes = pathlib.Path("work/synth_wav/normal_30s.wav").read_bytes()  # o un WAV propio
payload = {"audio": base64.b64encode(wav_bytes).decode()}
response = httpx.post("http://127.0.0.1:8025/detect", json=payload, timeout=30)
print(response.status_code, response.json())
# 200 {'is_synthetic': False, 'confidence': 0.94}
```

`work/gen_synth_wav.py` genera WAV sintéticos (senoides, no habla real) útiles para probar el formato del endpoint sin el dataset oficial; nunca deben usarse para afirmar exactitud de detección.

### Análisis de una llamada (Dev 3)

```bash
python -c 'import base64,json,pathlib; print(json.dumps({"audio":base64.b64encode(pathlib.Path("llamada.wav").read_bytes()).decode()}))' > work/call.json
curl -s http://127.0.0.1:8025/conversation/analyze -H 'Content-Type: application/json' --data-binary @work/call.json
```

Respuesta: `temporal` (resumen y eventos), `features` (26 valores), `semantic` (`not_requested` por defecto), `transcription` y `affects_detect: false`.

### Gemini opcional (pendiente por decisión del usuario)

Configurar en el entorno del servidor `GEMINI_API_KEY` (o `GOOGLE_API_KEY`) y `GEMINI_MODEL` con un modelo disponible en la cuenta con entrada de audio y salida JSON estructurada. Sin key o modelo, `/conversation/analyze?include_semantics=true` devuelve `unavailable/missing_api_key` sin contactar al proveedor. Ejemplo ficticio de `/conversation/semantic` (sin llamar a ningún proveedor, la transcripción ya viene dada):

```bash
curl -s http://127.0.0.1:8025/conversation/semantic -H 'Content-Type: application/json' --data-binary @examples/semantic_request.json
```

## Verificación y evaluación

```bash
python -m pytest -q
python verify_real_http.py --data-root /ruta/al/altur-data --url http://127.0.0.1:8025 --report verification-new-http.json
python verify_dev3_http.py --data-root /ruta/al/altur-data --url http://127.0.0.1:8025
```

El dataset debe contener `manifest.csv`, `turns/` y `audio/`; no se incluye en este paquete (ver https://github.com/alturio/hackmty26 y sus Releases). `verify_real_http.py` y `verify_dev3_http.py` aceptan `--report` para no sobrescribir la evidencia histórica (`verification-*.json` ya presentes en la raíz).

Sin dataset, `work/bench_detect.py` mide latencia y robustez con WAV sintéticos (ver `BENCHMARK-ANTES-DESPUES.md`).

### Verificación con el script oficial del juez

Altur publicó `scripts/check_endpoint.py` en `alturio/hackmty26` — el mismo cliente HTTP que usa el juez para evaluar, no una aproximación nuestra. Una copia queda en `work/altur_official/` para referencia (bajarla de nuevo por si Altur la actualiza):

```bash
python work/altur_official/check_endpoint.py \
  --url http://127.0.0.1:8025/detect \
  --manifest /ruta/al/altur-data/manifest.csv \
  --audio-dir /ruta/al/altur-data/audio \
  --split val --n 0 --out work/altur_official/check_endpoint_result.json
```

Última corrida sobre las 71 llamadas de val: `balanced_accuracy: 0.945`, `auc: 0.980`, `brier: 0.046`, 0 errores, latencia máxima 131 ms (límite del juez: 30 s). Ver `CAMBIOS-Y-VALIDACION.md` para el detalle completo, incluyendo un error de semántica en `confidence` que este mismo script detectó y que ya está corregido.

## Comparación de modelos

| Modelo | Aciertos históricos en 71 llamadas val (HTTP) | Predeterminado |
| --- | --- | --- |
| baseline (`models/model.pkl`) | 67/71 (94.4%) | Sí — conserva validación separada |
| alfa (`models/dev2Alfa.pkl`) | 64/71 (90.1%) | No — evidencia de scaler ajustado con train+val, ver `ALFA-REVISION.md` |

Estas cifras son históricas (macOS ARM64, Python 3.9.6); no se repitieron en esta revisión por falta del dataset. `models/registry.json` y `models/model.json` documentan hash, versión de scikit-learn y procedencia de cada variante.

## Límites y despliegue

- Cuerpo HTTP: ≈16 MiB antes de parsear JSON, aplicado con y sin `Content-Length`. WAV decodificado: 12 MiB.
- `/conversation/semantic`: hasta 512 KiB de cuerpo, 400 turnos, 60000 caracteres de texto total.
- Proveedores externos: máximo 2 solicitudes simultáneas a Gemini, 1 a ElevenLabs; timeouts de 20-30 s sin reintentos automáticos. Ninguno de los tres agrega latencia al camino de `/detect`.
- Docker: `docker build -t altur-hackmty .` y `docker run -p 8000:8000 -e MODEL_VARIANT=baseline altur-hackmty` — **no probado en este entorno** (sin Docker instalado). La imagen usa Python 3.12 mientras que las pruebas locales de esta revisión usaron Python 3.11.9 y las históricas Python 3.9.6; verificar antes de depender de ella para el benchmark oculto.
- Nube, HTTPS, dominio y validación de Gemini/PostgreSQL reales: pendientes, fuera de alcance de esta revisión (ElevenLabs ya se verificó, ver arriba).

## Variables de entorno

`MODEL_VARIANT`, `MODEL_PATH`, `AUDIT_DB_PATH`, `ADMIN_TOKEN`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`, `GEMINI_MODEL`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL_ID`, `DATABASE_URL`. Ninguna es obligatoria salvo para la integración correspondiente. Ver `.env.example`.

## Documentos de esta entrega

- `REVISION-TECNICA.md`: hallazgos confirmados, severidad, causa y evidencia.
- `CAMBIOS-Y-VALIDACION.md`: qué se corrigió, qué pruebas nuevas se agregaron y sus resultados.
- `BENCHMARK-ANTES-DESPUES.md`: metodología y resultados de latencia, con lo que no se pudo medir explícito.
