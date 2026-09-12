# Cambios implementados y su validación

Ver `REVISION-TECNICA.md` para el detalle de causa/evidencia de cada hallazgo. Este documento se centra en qué cambió en el código y cómo se comprobó.

## Entorno de esta validación

- Windows, Python 3.11.9 (la entrega anterior se probó en macOS ARM64/Python 3.9.6; no se afirma equivalencia de rendimiento entre ambos, solo de comportamiento funcional).
- Entorno virtual `.venv` creado con `python -m venv .venv` e instalado desde `requirements-lock.txt` + `requirements-dev.txt`.
- Variables externas (`GEMINI_API_KEY`, `GOOGLE_API_KEY`, `GEMINI_MODEL`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `DATABASE_URL`, `MODEL_PATH`, `ADMIN_TOKEN`) retiradas explícitamente del entorno antes de cada ejecución, para que la evaluación no dependa de credenciales.

## Cambios de código

| Archivo | Cambio | Motivo |
| --- | --- | --- |
| `dev4/audit.py` | `written` ahora solo cuenta filas realmente insertadas (`cursor.rowcount == 1`); se agregó contador `duplicate_ignored` | Hallazgo #3 |
| `dev4/postgres.py` | Cierre explícito de la conexión SQLite local (`try/finally: local.close()`); `sqlite3.OperationalError` por tabla ausente se traduce a `status: unavailable, reason: audit_db_not_initialized` en vez de propagar la excepción | Hallazgo #4 |
| `dev4/demo.html` | Bandera `voiceBusy` para que el refresco automático (cada 3 s) no reactive el botón de generar alerta mientras una solicitud sigue en curso | Hallazgo #5 |
| `verify_dev3_http.py` | Antes de probar `include_semantics=true` sin credenciales, se comprueba `/conversation/status`; si el servidor ya tiene Gemini configurado, la verificación se omite en vez de llamar al proveedor real y fallar. Se agregó `--report` (por defecto `verification-dev3-http-new.json`) para no sobrescribir la evidencia histórica. Puerto por defecto actualizado a 8025 | Hallazgos #1, #2, #7 |
| `verify_real_http.py` | Puerto por defecto actualizado a 8025 para consistencia con el README vigente | Hallazgo #7 |
| `README.md` | Consolidado como guía única y vigente: arranque, puerto 8025, variables, estado real de cada integración, ejemplos de petición/respuesta, y qué sigue pendiente. Los documentos DEV*-ENTREGA.md/INTEGRACION.md/ALFA-REVISION.md se mantienen intactos como historial | Hallazgo #7 |
| `ml/features.py` | Se agregó el parámetro opt-in `latency_pairing` (`"legacy"` por defecto, `"signed_v2"` corregido) a `extract_features_from_turns()`. Con el valor por defecto el comportamiento es idéntico byte a byte al anterior; `app/model.py` sigue llamando sin este argumento, así que el modelo desplegado no se ve afectado | Hallazgo #6 |
| `train_validate.py` | Acepta `--latency-pairing {legacy,signed_v2}` (por defecto `legacy`) y registra la opción usada en `models/model.json`, para poder producir la próxima generación de pesos con la fórmula corregida en cuanto exista el dataset oficial | Hallazgo #6 |

No se tocaron: `app/main.py`, `app/audio.py`, `app/limits.py`, `app/model.py`, `ml/ensemble.py`, `dev3/temporal.py`, `dev3/semantics.py`, `dev4/voice.py`, `models/*.pkl`, `models/registry.json`, `models/model.json`. El contrato de `/detect`, las predicciones del modelo baseline y la variante alfa quedan exactamente iguales a como estaban.

## Pruebas nuevas

Se agregaron 7 pruebas nuevas, todas dentro de la suite existente (`pytest`):

1. `tests/test_dev4.py::test_postgres_sync_on_uninitialized_db_reports_status_not_crash` — reproduce el hallazgo #4 con una base SQLite vacía nunca inicializada por `AuditStore`, confirma que `sync_batch()` responde `audit_db_not_initialized` en vez de lanzar `sqlite3.OperationalError`, y que nunca intenta contactar al "proveedor" (se le pasa un `connect` que hace `pytest.fail` si se invoca).
2. `tests/test_dev4.py::test_audit_write_counter_distinguishes_duplicate_request_id` — encola el mismo evento dos veces con el mismo `request_id`; confirma `written == 1`, `duplicate_ignored == 1` y `persisted.total == 1` (hallazgo #3).
3. `tests/test_features.py::test_default_call_is_legacy_and_latency_frac_negative_is_structurally_always_zero` — fija el comportamiento actual del hallazgo #6 (llamando la función tal como la usa `Detector`, sin el argumento nuevo).
4. `tests/test_features.py::test_explicit_legacy_matches_default` — confirma que pasar `latency_pairing="legacy"` explícito es idéntico a no pasar el argumento, para que quede documentado que el default no cambió.
5. `tests/test_features.py::test_signed_v2_produces_negative_latency_on_real_overlap` — confirma que la corrección (`signed_v2`) sí produce latencias negativas (-3.0 s y -0.5 s) en el mismo caso de solapamiento donde `legacy` da 0.
6. `tests/test_features.py::test_signed_v2_keeps_the_same_87_column_names_as_legacy` — confirma que ambas variantes devuelven exactamente las mismas columnas en el mismo orden, precondición para que un reentrenamiento futuro siga siendo compatible con el chequeo de identidad de `Detector`.
7. `tests/test_features.py::test_invalid_latency_pairing_is_rejected` — valida el guard del parámetro nuevo.

## Resultados de pytest

**Antes de esta revisión (línea base reproducida, sin modificar nada):**

```
81 passed in 63.99s
```

**Después de los cambios de esta revisión:**

```
88 passed in 11.64s
```

81 pruebas heredadas siguen pasando sin cambios de comportamiento (ninguna aserción existente se modificó); se suman las 7 pruebas nuevas. La diferencia de tiempo (64 s → ~10 s) se observó entre corridas en la misma máquina y probablemente se debe a caché de disco/antivirus en la primera ejecución tras crear el entorno virtual, no a una optimización deliberada; no se afirma como mejora de rendimiento del código. Ver `BENCHMARK-ANTES-DESPUES.md` para mediciones de latencia con metodología explícita.

**Sobre el hallazgo #6 en particular:** el arreglo de `latency_frac_negative` (ver tabla de cambios de código arriba) es opt-in y no cambia ninguna predicción del modelo desplegado — se verificó que las 88 pruebas pasan, incluyendo las que cargan el `Detector` real contra `models/model.pkl` sin ningún cambio de comportamiento. Falta el paso final (reentrenar con el dataset oficial usando `--latency-pairing signed_v2`, comparar contra el baseline actual en val, y solo promoverlo si mejora) — ver el procedimiento exacto en `REVISION-TECNICA.md`.

## Verificación manual adicional (HTTP)

Con el servidor corriendo localmente (`MODEL_VARIANT=baseline`, sin credenciales externas, puerto 8025):

- `GET /ready` → `200`, hash del baseline coincide con `models/registry.json`.
- `GET /model` → `feature_count: 87`, `training_source: audio_vad`, `holdout_preserved: true`, sin cambios respecto al registro.
- `POST /detect` con cuerpo vacío, base64 inválido, y `audio`/`audio_base64` contradictorios → los tres devuelven `422`, como documentado.
- `POST /detect` con cuerpo de 17 MiB, con y sin `Content-Length` declarado → `413` en ambos casos.
- `POST /detect` con cuerpo de 13 MiB de base64 "basura" (bajo el límite de bytes pero no un WAV válido) → `422` (correcto: el límite de tamaño no dispara, pero la decodificación de WAV sí falla).
- `GET /conversation/status`, `GET /audit/stats`, `GET /voice/status` desde loopback sin `ADMIN_TOKEN` → `200` (acceso de operador local permitido, como documentado).
- Tras ~250 solicitudes de prueba, `/audit/stats` reportó `process_queue.written == persisted.total` y `duplicate_ignored: 0`, confirmando que el contador corregido cuadra con las filas reales en SQLite.

Esta verificación manual usó WAV **sintéticos** generados en `work/gen_synth_wav.py` (senoides con ruido, no habla real) porque el dataset oficial no está disponible en este entorno. Sirven para probar formato, límites y comportamiento del servidor — **no para afirmar exactitud de clasificación**, que solo puede medirse con el dataset oficial o el benchmark oculto de Altur.

## ElevenLabs: verificación con cuenta real (2026-09-12)

A diferencia del resto de esta revisión, esta parte sí se probó con una cuenta y credenciales reales del equipo (nunca vistas ni manejadas por el asistente; el usuario las configuró directamente en su propia terminal como variables de entorno). Resultado final: `POST /voice/alert` devolvió `200` con un MP3 real y válido (ID3 v2.4.0, MPEG layer III, 128 kbps, 44.1 kHz, ~188 KB), generado por la API real de ElevenLabs a partir del texto fijo de alerta.

El camino hasta llegar ahí pasó por cuatro causas de error distintas, cada una diagnosticada leyendo la razón que ya devuelve `dev4/voice.py` (y, cuando la razón agregada no bastaba, con un `print` de depuración temporal agregado solo para esta sesión de trabajo, que imprimía el cuerpo de la respuesta del proveedor únicamente en la terminal del servidor — nunca en la respuesta HTTP al cliente — y que ya se quitó del código antes de este commit):

1. **`provider_http_401`, `"status":"api_key_id_used_as_api_key"`** — se usó el *ID* de la API key (visible en la tabla del dashboard de ElevenLabs) en vez de la key secreta real (que siempre empieza con `sk_`).
2. **`provider_http_401`, `"status":"missing_permissions"`** — una key con formato correcto pero creada con permisos restringidos, sin el scope de *Text to Speech* habilitado.
3. **`provider_http_401`, `"status":"detected_unusual_activity"`** — ElevenLabs deshabilitó el acceso Free Tier de la cuenta por "actividad inusual" (su sistema antiabuso; puede activarse por VPN/proxy o por varios intentos seguidos). Se resolvió solo, sin cambiar nada, después de un rato.
4. **`provider_http_402`, `"status":"payment_required"`, código `paid_plan_required`** — cuentas Free Tier no pueden usar voces de la librería pública de ElevenLabs vía API. Se resolvió agregando una voz a "My Voices" en el dashboard y usando ese Voice ID.

En ningún momento hizo falta cambiar `dev4/voice.py`: el adaptador ya distinguía y exponía correctamente cada código de estado (`provider_http_401`/`402`, con `retryable` correcto) desde antes de esta sesión. El diagnóstico fue enteramente de configuración de cuenta, no de código. Ver también la nota de actualización en `DEV4-ENTREGA.md`.

**Qué significa esto para el estado del proyecto:** `dev4/voice.py` deja de ser "solo probado con mocks" y pasa a tener una verificación end-to-end real, aunque puntual (una sola cuenta, una sola vez, no automatizada en CI). `status()` sigue devolviendo `live_verified: false` siempre por diseño — esa bandera describe si el proceso actual ya confirmó el proveedor en esta ejecución, no si alguna vez funcionó; no se cambió ese comportamiento.

## Qué no se validó en esta fase (y por qué)

| Área | Motivo |
| --- | --- |
| Exactitud del modelo contra las 71 llamadas de val | Dataset oficial no presente en esta máquina |
| Reentrenamiento con `latency_frac_negative` corregido | Requiere dataset oficial |
| Build/despliegue Docker en Linux | Docker no instalado en este entorno |
| Gemini con cuenta real | Decisión expresa del usuario: pendiente |
| PostgreSQL/Tiger Data con base real | Sin credenciales disponibles |
| Despliegue público / HTTPS / Vultr | Fuera de alcance de esta revisión, no solicitado |
