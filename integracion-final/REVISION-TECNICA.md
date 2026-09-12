# Revisión técnica — Altur_Hackmty

Fecha de esta revisión: 2026-09-12. Entorno: Windows 10/11, Python 3.11.9, dependencias fijadas en `requirements-lock.txt` (scikit-learn 1.6.1, numpy 2.0.2, fastapi 0.128.8). La entrega anterior se probó en macOS ARM64 con Python 3.9.6; esta revisión reproduce la suite en un entorno distinto y documenta cualquier diferencia.

No se contó con el dataset oficial (`altur-data/manifest.csv`, `audio/`, `turns/`) en esta máquina: se buscó en las rutas conocidas y no está presente. Esto limita la Fase D (precisión) a análisis de código y pruebas con WAV sintéticos generados localmente (ver `BENCHMARK-ANTES-DESPUES.md`), nunca a una medición de exactitud nueva. Tampoco se dispuso de Docker instalado, ni de credenciales de Gemini/ElevenLabs/PostgreSQL — esto es una decisión expresa del usuario para Gemini y una limitación real de este equipo para el resto.

## Línea base reproducida

- `python -m pytest -q` con variables externas retiradas del entorno: **81/81 pruebas aprobadas**, igual al resultado histórico documentado en `CLAUDE-CODE-ENTREGA.md`. Confirma que el estado heredado es reproducible en un entorno nuevo.
- El modelo baseline carga con el SHA-256 registrado (`9aef6ea7a5efc0a4ca3488054f914c2c69acaaa1cba0f7535fdd0342e7472f2f`) y pasa la comprobación numérica de arranque.
- El servidor local expone `/health`, `/ready`, `/model`, `/detect`, `/conversation/*`, `/audit/*`, `/voice/*`, `/demo` como documentado.

## Hallazgos confirmados y corregidos

### 1. `verify_dev3_http.py` podía contactar a Gemini real y fallar por eso (Alto)

- **Archivo:** [verify_dev3_http.py:62-63](verify_dev3_http.py) (versión previa a esta revisión).
- **Causa:** el script llamaba `/conversation/analyze` con `include_semantics=true` y asumía sin comprobar que el servidor no tenía credenciales de Gemini configuradas (`assert unavailable["reason"] == "missing_api_key"`). Si alguien ejecuta esta verificación contra un servidor con `GEMINI_API_KEY` configurada, el script dispara una solicitud real al proveedor y luego el assert falla porque la razón ya no es `missing_api_key`.
- **Evidencia:** lectura del código; el propio `GUION-DEMO.md`/`DEV3-ENTREGA.md` advierten que Gemini no debe llamarse sin decisión explícita.
- **Corrección:** el script ahora lee `semantic` desde `/conversation/status` primero y solo ejecuta la verificación offline si el estado es `unavailable/missing_api_key`. Si el servidor aparenta tener Gemini configurado, la verificación se **omite explícitamente** (`skipped: true`) en vez de contactar al proveedor o fallar de forma confusa.
- **Estado:** corregido. Prueba: no hay test automatizado de este script (es un script de integración HTTP manual), pero la lógica se revisó línea por línea y se ejecutó manualmente contra el servidor local sin `GEMINI_API_KEY` (ver `BENCHMARK-ANTES-DESPUES.md`).

### 2. `verify_dev3_http.py` sobrescribía la evidencia histórica (Medio)

- **Archivo:** [verify_dev3_http.py:69](verify_dev3_http.py) (versión previa).
- **Causa:** escribía siempre en la ruta fija `verification-dev3-http.json`, que ya contiene evidencia histórica de una corrida anterior. Cualquier ejecución nueva destruye ese reporte sin posibilidad de comparar antes/después.
- **Corrección:** se agregó `--report`, con valor por defecto `verification-dev3-http-new.json` (no colisiona con el archivo histórico). `verification-dev3-http.json` original queda intacto en el repositorio.
- **Estado:** corregido.

### 3. `dev4/audit.py`: el contador `written` no distinguía inserciones reales de duplicados ignorados (Bajo/Medio)

- **Archivo:** [dev4/audit.py:70-75](dev4/audit.py) (versión previa).
- **Causa:** `INSERT OR IGNORE` no lanza excepción cuando `request_id` ya existe; el código incrementaba `written` sin mirar `cursor.rowcount`, así que un intento ignorado por colisión de clave primaria se contaba igual que una fila nueva persistida. Con UUID4 la colisión es extremadamente improbable en operación normal, pero el contador no reflejaba la realidad si ocurriera un reintento con el mismo `request_id` (por ejemplo, un cliente HTTP que reintenta con el mismo header de idempotencia en el futuro).
- **Corrección:** se usa `cursor.rowcount` para incrementar `written` solo si la fila se insertó de verdad, y se agregó un contador nuevo `duplicate_ignored` para que la distinción sea visible en `/audit/stats`.
- **Prueba nueva:** `tests/test_dev4.py::test_audit_write_counter_distinguishes_duplicate_request_id`.
- **Estado:** corregido y probado.

### 4. `dev4/postgres.py`: conexión SQLite local sin cierre explícito y sin manejo de tabla ausente (Medio)

- **Archivo:** [dev4/postgres.py:27-51](dev4/postgres.py) (versión previa).
- **Causa doble:**
  - `with sqlite3.connect(path, timeout=2) as local:` usa `Connection` como gestor de contexto, pero en el módulo estándar `sqlite3` eso solo controla la transacción (commit/rollback); **no cierra la conexión**. El archivo se mantenía abierto tras `sync_batch()`, lo que en un proceso de larga vida acumula descriptores de archivo.
  - Si `sync_batch()` se ejecuta contra una ruta SQLite que nunca fue inicializada por `AuditStore` (archivo vacío o inexistente creado por `sqlite3.connect`), `SELECT * FROM calls ...` lanza `sqlite3.OperationalError: no such table: calls` sin capturarse, y el script termina con una traza no controlada en vez de un estado explícito.
- **Corrección:** conexión con `try/finally: local.close()`, y `sqlite3.OperationalError` en la consulta se traduce a `{"status": "unavailable", "reason": "audit_db_not_initialized"}` sin tocar la red.
- **Prueba nueva:** `tests/test_dev4.py::test_postgres_sync_on_uninitialized_db_reports_status_not_crash`.
- **Estado:** corregido y probado.

### 5. `dev4/demo.html`: el refresco periódico podía reactivar "Generar alerta" durante una solicitud en curso (Bajo)

- **Archivo:** [dev4/demo.html](dev4/demo.html) (versión previa).
- **Causa:** `el('voice').onclick` deshabilita el botón al iniciar y lo reactiva en `finally`, pero `setInterval(refresh, 3000)` corría en paralelo y hacía `el('voice').disabled = voice.status !== 'configured'` sin mirar si ya había una generación en curso. Si el proveedor está `configured`, el refresco de 3 segundos podía reactivar el botón mientras la primera solicitud seguía pendiente, permitiendo un doble clic que dispara una segunda solicitud a ElevenLabs (con costo real de crédito) mientras la primera todavía no responde.
- **Corrección:** se añadió una bandera JS `voiceBusy` que el refresco respeta (`if(!voiceBusy) el('voice').disabled = ...`); el botón permanece deshabilitado desde el clic hasta que la promesa de `/voice/alert` resuelve o falla.
- **Estado:** corregido. Es un cambio solo de JavaScript de página estática; se revisó manualmente el flujo (no hay test automatizado de UI en este proyecto). Recomendado: prueba manual con DevTools abiertas simulando latencia de red antes de la demo en vivo.

## Hallazgos confirmados con corrección lista pero no aplicada al modelo desplegado

### 6. `latency_frac_negative` en el modelo de 87 características es estructuralmente siempre 0 (Medio)

- **Archivo:** [ml/features.py](ml/features.py).
- **Causa:** la lista `latencies` solo empareja turnos del cliente con turnos del agente que **ya terminaron** antes de que el cliente empiece (`a["end"] <= ct["start"]`). Por construcción, `ct["start"] - max(prev) >= 0` siempre, así que `latency_frac_negative = mean(l < 0 for l in latencies)` es 0.0 para cualquier entrada válida, incluyendo casos con solapamiento real cliente/agente (ese turno del agente queda excluido del cálculo, no incluido con signo negativo).
- **Por qué no se aplicó directamente al modelo desplegado:** `models/model.pkl` y `models/dev2Alfa.pkl` fueron entrenados exactamente contra esta definición (columna `latency_frac_negative` siempre 0 en el set de entrenamiento). Cambiar la fórmula que consume `Detector` sin reentrenar alimentaría al modelo desplegado con una distribución de entrada que nunca vio, lo cual el propio encargo prohíbe explícitamente ("no alimentes pesos antiguos con una definición nueva silenciosamente"). Reentrenar requiere el dataset oficial, que sigue sin estar disponible en este entorno.
- **Corrección implementada, en modo opt-in (sin efecto en producción hoy):** `extract_features_from_turns()` ahora acepta `latency_pairing` (`"legacy"` por defecto, `"signed_v2"` explícito):
  - `"legacy"`: exactamente el cálculo anterior, byte a byte. `app/model.py`'s `Detector` sigue llamando a la función sin este argumento, así que **su comportamiento no cambió ni un bit** — mismo `Detector.__init__` que compara `self.columns == list(extract_features_from_turns([], 1).keys())` sigue pasando porque los nombres de columnas son idénticos.
  - `"signed_v2"`: empareja cada turno de agente con el primer turno del cliente que empieza después de ese agente y antes del siguiente (la misma definición ya probada en `dev3/temporal.py`, ver `tests/test_temporal.py::test_response_candidates_are_signed_unique_and_belong_to_latest_agent`), y calcula `latencia = inicio_cliente - fin_agente`, que sí puede ser negativa ante solapamiento real.
  - `train_validate.py` ahora acepta `--latency-pairing {legacy,signed_v2}` (por defecto `legacy`, para no cambiar accidentalmente el comportamiento de reproducción del baseline actual) y registra qué opción se usó en `models/model.json`.
- **Pruebas nuevas:** `tests/test_features.py` (5 pruebas): confirman que la llamada por defecto (sin el argumento nuevo) es idéntica a `"legacy"` explícito; que `"signed_v2"` sí produce latencias negativas en un caso de solapamiento real (verificado a mano: -3.0 y -0.5 s en el ejemplo de la prueba); que ambas variantes devuelven exactamente los mismos 87 nombres de columna en el mismo orden (precondición para que el chequeo de identidad de `Detector` siga siendo significativo tras un reentrenamiento); y que un valor inválido de `latency_pairing` se rechaza.
- **Verificación de que no afecta el modelo desplegado:** la suite completa (incluyendo `tests/test_model.py`, que carga el `Detector` real y compara su salida contra una suma explícita independiente) sigue pasando sin cambios: 88/88 pruebas.
- **Cómo terminar la corrección cuando llegue el dataset oficial (procedimiento exacto):**
  1. `python train_validate.py --data-root /ruta/al/altur-data --latency-pairing signed_v2 --out models/model_v2_signed_latency.pkl` — produce un artefacto nuevo, sin tocar `models/model.pkl`.
  2. Comparar `models/model_v2_signed_latency.json` contra `models/model.json` (mismo protocolo: train/val ya separados por el propio manifest, sin tocar val durante el ajuste). Si `roc_auc`/`balanced_accuracy` en val no mejoran o empeoran, **no promover** el modelo nuevo — se documenta como experimento negativo, igual que se hizo con las 26 features de Dev 3 (ver `INTEGRACION.md`).
  3. Si mejora (o al menos no empeora) de forma justificable: calcular su SHA-256, añadir una entrada nueva en `models/registry.json` (por ejemplo `"baseline_v2"`) sin borrar la entrada `"baseline"` actual, y solo entonces considerar cambiar el valor por defecto de `MODEL_VARIANT` — con su propia validación HTTP (`verify_real_http.py`) antes de promoverlo.
  4. Nunca sobrescribir `models/model.pkl` in place: el registro debe permitir volver atrás a la versión anterior en cualquier momento (ver criterio de aceptación "procedimiento para volver al baseline").
- **Estado:** corrección de código implementada, probada y lista para usar; **no aplicada al modelo en producción** porque eso requiere el dataset oficial y una nueva validación, que no se pueden hacer en este entorno.

## Contrato oficial de Altur actualizado (2026-09-12)

El organizador publicó en `alturio/hackmty26` el contrato exacto de `/detect` (campos `call_id`/`audio_base64`/`sample_rate`/`channels` en la petición, `is_synthetic`/`confidence` en la respuesta, límite de 30 s por llamada, y dos scripts de referencia: `scripts/check_endpoint.py` — el mismo cliente que usa el juez — y `scripts/example_server.py`). Se descargaron ambos (`work/altur_official/`) y se usó `check_endpoint.py` para evaluar nuestro servidor real con las 71 llamadas de val del dataset oficial, exactamente como lo haría el juez.

### 7. `confidence` invertía el AUC según el script oficial del juez (Alto, corregido)

- **Archivo:** [app/model.py:64-77](app/model.py).
- **Causa:** al activar `confidence` en la respuesta de `/detect` (antes se omitía por completo), se envió directamente `p_synthetic` sin ajustar. Pero `scripts/check_endpoint.py` del juez reconstruye la probabilidad de sintético como `confidence` si `is_synthetic=true`, o `1 - confidence` si `is_synthetic=false` — es decir, el contrato espera que `confidence` sea la **confianza en el veredicto devuelto**, no `P(sintético)` cruda.
- **Evidencia medida:** primera corrida de `check_endpoint.py --split val` (71 llamadas) con `confidence = p_synthetic` sin ajustar: `auc: 0.442` (peor que azar), `brier: 0.448`. `balanced_accuracy` seguía en 0.945 porque `is_synthetic` no depende de este cálculo.
- **Corrección:** `confidence = p_synthetic si is_synthetic, si no 1 - p_synthetic`. Reevaluado con el mismo script y las mismas 71 llamadas: `auc: 0.980`, `brier: 0.046` — coincide con las métricas internas ya conocidas en `models/model.json` (ROC-AUC 0.9801, Brier 0.0461), confirmando que el modelo siempre fue bueno; el defecto era solo de exposición en la API pública.
- **Prueba nueva:** `tests/test_dev4.py::test_classification_contract_and_internal_probability` ahora verifica que `confidence >= 0.5` y que coincide con `p_synthetic` (auditoría interna) solo cuando el veredicto es "sintético", con `1 - p_synthetic` en el caso contrario.
- **Estado:** corregido y verificado con el harness oficial del juez, no solo con nuestros propios scripts. Este hallazgo se detectó únicamente porque se probó con `check_endpoint.py` real antes de dar por buena la activación de `confidence` — una prueba contra nuestro propio `verify_real_http.py` (que no valida `confidence`) nunca lo habría revelado.

### 8. Un `422` por audio sin habla del cliente cuenta como respuesta incorrecta en el conjunto oculto (Medio, riesgo confirmado, sin resolver)

- **Archivo:** [app/model.py:65-67](app/model.py) (`InsufficientSpeechError`), [app/main.py:182-183](app/main.py).
- **Causa:** el contrato del juez es explícito: "Un timeout, un status distinto de 200 o una respuesta sin `is_synthetic` booleano cuenta como respuesta incorrecta." Nuestro `/detect` devuelve `422` cuando el VAD no detecta habla del cliente. En las 71 llamadas de val esto no ocurrió nunca (`errors: 0` en ambas corridas de `check_endpoint.py`), pero el conjunto oculto usa voces y hablantes nuevos — un audio con el canal del cliente muy silencioso o con un patrón que el VAD energético no reconozca produciría un `422`, que cuenta como fallo total en vez de tener aunque sea una oportunidad de acertar al azar.
- **Por qué no se decidió aquí:** corregirlo implica una decisión de producto, no solo de código — devolver un veredicto por defecto (¿cuál? ¿con qué justificación?) cambia el comportamiento documentado ("un error nunca produce un veredicto ficticio") y podría ocultar audio genuinamente inválido en vez de degradar con gracia. Es una decisión que le corresponde al equipo, no algo para resolver unilateralmente.
- **Estado:** riesgo confirmado y documentado, sin mitigar. Pendiente de decisión del equipo.

## Puntos revisados sin defecto confirmado

- **Límite de cuerpo HTTP con y sin `Content-Length`:** probado manualmente contra el servidor local con cuerpos de 13 MiB (pasa validación de tamaño, falla luego por WAV inválido → 422, correcto) y 17 MiB (rechazado con 413 tanto con `Content-Length` declarado como con `Transfer-Encoding: chunked`). Comportamiento documentado se sostiene.
- **Contrato `audio`/`audio_base64` contradictorios:** produce 422 como está documentado.
- **Modelo, versión de scikit-learn, orden de columnas y clases:** las comprobaciones de `app/model.py` en el arranque (`Detector.__init__`) se ejecutaron y pasaron con el entorno de esta revisión (Python 3.11.9, scikit-learn 1.6.1 vía `requirements-lock.txt`).
- **Cierre de `AuditStore` y comportamiento ante fallo de apertura de SQLite:** cubierto por pruebas existentes (`test_db_startup_failure_does_not_disable_model`) y se confirmó manualmente que `/detect` sigue disponible aunque la auditoría falle.
- **Docker:** no se pudo construir ni probar; no hay `docker` instalado en este entorno. Sigue como pendiente heredado, ahora explícito también aquí.
- **Gemini/PostgreSQL con credenciales reales:** fuera de alcance por decisión del usuario (Gemini) y por falta de cuentas (PostgreSQL). Los adaptadores y sus pruebas con mocks no se modificaron en su lógica de negocio, solo el cierre de recursos en `dev4/postgres.py`. **ElevenLabs sí se verificó con cuenta real** (ver `DEV4-ENTREGA.md` y `CAMBIOS-Y-VALIDACION.md`): generación de MP3 real exitosa tras resolver varios problemas de configuración de cuenta (no de código).

## Referencias de puertos y documentación desactualizada (Bajo)

`README.md`, `DEV1-DEV2-REFERENCIA.md` y `verify_real_http.py`/`verify_dev3_http.py` usaban puertos heredados de distintas etapas (8018–8023). Esto no es un defecto de comportamiento, solo inconsistencia documental que puede confundir a quien arranque el proyecto. Se actualizó `README.md` (la guía principal vigente) para usar el puerto 8025 sugerido en `CLAUDE-CODE-ENTREGA.md` de forma consistente, y los valores por defecto de `verify_real_http.py`/`verify_dev3_http.py` para que apunten ahí también. Los documentos históricos (`DEV1-DEV2-REFERENCIA.md`, `INTEGRACION.md`, `ALFA-REVISION.md`, `DEV3-ENTREGA.md`, `DEV4-ENTREGA.md`) se dejaron sin tocar: describen el estado de cada entrega en su momento y no deben reescribirse para parecer la guía actual.

## Resumen de severidad

| # | Hallazgo | Severidad | Estado |
| - | --- | --- | --- |
| 1 | Verificación Dev3 podía llamar a Gemini real | Alto | Corregido |
| 2 | Verificación Dev3 sobrescribía evidencia histórica | Medio | Corregido |
| 3 | Contador `written` no distinguía duplicados | Bajo/Medio | Corregido + test |
| 4 | Conexión SQLite sin cerrar / tabla ausente en `postgres.py` | Medio | Corregido + test |
| 5 | Botón de voz reactivable durante solicitud en curso | Bajo | Corregido |
| 6 | `latency_frac_negative` siempre 0 | Medio | Corrección lista y probada (`latency_pairing="signed_v2"`), opt-in; falta reentrenar+validar con dataset oficial para promoverla |
| 7 | `confidence` invertía el AUC (semántica de "confianza en veredicto" mal interpretada) | Alto | Corregido y verificado con el script oficial del juez (AUC 0.442 → 0.980) |
| 8 | `422` por falta de habla del cliente cuenta como respuesta incorrecta en el conjunto oculto | Medio | Riesgo confirmado, documentado, pendiente de decisión del equipo |
| 9 | Puertos inconsistentes en documentación | Bajo | Consolidado en README.md vigente |
