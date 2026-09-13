# Benchmark antes/después

## Resumen honesto primero

Esta revisión **no aplicó optimizaciones de rendimiento al código de inferencia**: los defectos confirmados (ver `REVISION-TECNICA.md`) fueron de corrección/robustez (scripts de verificación, contadores de auditoría, exportación a Postgres, un botón de demo), no de latencia. Por lo tanto no hay un "antes/después" de código para `/detect` en esta entrega — hay una **línea base nueva**, medida en una máquina y sistema operativo distintos a los históricos, que sirve de referencia para trabajo futuro.

No se pudo usar el dataset oficial (no está presente en este entorno), así que ninguna cifra de aquí abajo mide exactitud. Todo lo medido es latencia y comportamiento HTTP con WAV **sintéticos** generados en `work/gen_synth_wav.py` (senoides con ruido de fondo, 8 kHz, estéreo, distintas duraciones y patrones de solapamiento). Los generadores y el script de benchmark (`work/bench_detect.py`) quedan en el repositorio para que Altur o el equipo los reutilicen contra el dataset real cuando esté disponible.

## Condiciones de medición

- Máquina: Windows, Python 3.11.9, dependencias de `requirements-lock.txt` (scikit-learn 1.6.1).
- Servidor: `uvicorn app.main:app --host 127.0.0.1 --port 8025`, `MODEL_VARIANT=baseline`, sin credenciales externas, base de auditoría de prueba separada (`AUDIT_DB_PATH=work/audit-test.sqlite3`, borrada después).
- Cliente: `httpx.Client` en el mismo proceso Python, sin proxy, `localhost`.
- Referencia histórica (no comparable directamente): macOS ARM64, Python 3.9.6, WAV reales de val, ver `verification-dev4-http.json` (media 47.2 ms, p95 59.1 ms sobre 71 llamadas reales).

## Latencia por forma de clip (30 repeticiones cada uno, tras un warm-up descartado)

| Clip sintético | Duración | Resultado | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| `short_3s.wav` | 3 s | 200 (humano) | 47.5 | 48.3 | 48.7 | 48.9 |
| `normal_30s.wav` | 30 s | 200 (humano) | 61.6 | 63.5 | 64.8 | 65.0 |
| `cropped_fragment_10s.wav` | 10 s | 200 (humano) | 50.6 | 51.4 | 51.6 | 51.7 |
| `heavy_overlap_20s.wav` | 20 s | 200 (humano) | 55.8 | 57.2 | 57.6 | 57.8 |
| `long_120s.wav` | 120 s | 200 (humano) | 99.9 | 103.8 | 107.4 | 108.5 |
| `silent_client_15s.wav` | 15 s | **422** (sin habla del cliente) | 9.4 | 10.9 | 13.2 | 14.1 |

Todos los clips sintéticos se clasificaron como "humano" — esperable, ya que son senoides con ruido y no voz sintética real de TTS; esto no es una medición de exactitud, solo confirma que el endpoint procesa y responde con clips de duración/patrón variados.

Observaciones:
- La latencia crece con la duración del audio (dominado por el VAD y la extracción de 87 features sobre más muestras), de ~48 ms en un clip de 3 s a ~100 ms en uno de 120 s. Es un crecimiento sublineal razonable, no hay evidencia de una operación cuadrática evidente en clips de esta escala.
- El rechazo temprano por falta de habla (`422`) es notablemente más rápido (~9-14 ms) porque corta antes de extraer características del canal del agente ni de correr el modelo — el comportamiento de `Detector.predict_details` (falla si `caller_turns` está vacío antes de tocar el modelo) ya está bien encaminado para el caso de error.
- No se puede comparar directamente contra los 47.2 ms / 59.1 ms p95 históricos: son máquinas, sistemas operativos y (sobre todo) audios distintos. Como referencia de orden de magnitud, los clips sintéticos de duración similar a las llamadas reales (~30 s) están en el mismo rango (61-65 ms aquí vs. 47-59 ms histórico), lo cual es razonable considerando que es hardware distinto.

## Concurrencia (clip `normal_30s.wav`, 1/2/4/8 hilos)

| Hilos | Solicitudes | p50 (ms) | p95 (ms) | Throughput (req/s) |
| --- | --- | --- | --- | --- |
| 1 | 8 | 61.9 | 62.3 | 16.1 |
| 2 | 8 | 109.5 | 126.5 | 17.7 |
| 4 | 16 | 218.0 | 231.2 | 17.7 |
| 8 | 32 | 424.0 | 433.8 | 18.5 |

**Hallazgo (no corregido en esta fase):** el throughput no crece con más hilos (~16-18 req/s en todos los niveles) y la latencia p50 crece casi linealmente con la concurrencia. Esto es consistente con el diseño actual de `app/model.py`: el acceso a `model.predict_proba` está serializado con un `Lock` explícito (`with self.lock, threadpool_limits(limits=1):`), documentado como control deliberado de thread-safety para los estimadores de scikit-learn. La extracción de características (VAD, `extract_features_from_turns`) sí corre fuera del lock y puede paralelizarse.

No se retiró ni se relajó ese lock en esta revisión: hacerlo requeriría antes confirmar que los tres estimadores del ensemble (`LogisticRegression`, `RandomForestClassifier` con `n_jobs=1`, `HistGradientBoostingClassifier`) son seguros para invocación concurrente de `predict_proba` sobre la misma instancia — no hay evidencia reunida en esta revisión de que lo sean, y las instrucciones del encargo prohíben quitar controles de concurrencia solo para bajar milisegundos sin justificarlo. Queda como candidato de optimización futura, con este benchmark como línea base: si se investiga y se confirma que es seguro paralelizar la inferencia (o cargar una copia del modelo por hilo/worker), debería volver a medirse con esta misma metodología antes/después.

## Cómo reproducir

```bash
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
python -m pip install -r requirements-lock.txt -r requirements-dev.txt
python work/gen_synth_wav.py
env -u GEMINI_API_KEY -u GOOGLE_API_KEY -u GEMINI_MODEL -u ELEVENLABS_API_KEY -u ELEVENLABS_VOICE_ID -u DATABASE_URL -u MODEL_PATH \
    MODEL_VARIANT=baseline AUDIT_DB_PATH=work/audit-bench.sqlite3 \
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8025 &
python work/bench_detect.py --url http://127.0.0.1:8025 --repeats 30
```

El reporte completo de esta corrida queda en `work/bench-detect-report.json`.

## Con el dataset oficial disponible

Cuando se tenga acceso al dataset (`altur-data/`), el mismo `verify_real_http.py` (ya actualizado a puerto 8025) da la medición comparable con exactitud real:

```bash
python verify_real_http.py --data-root /ruta/al/altur-data --url http://127.0.0.1:8025 --report verification-new-http.json
```

No sobrescribe `verification-dev4-http.json` ni `verification-baseline-http.json` (ya usa `--report` con nombre distinto por defecto).
