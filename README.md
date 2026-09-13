# Altur — Detección de voz sintética en llamadas bancarias (HackMTY26)

Documento único: reúne todo el historial del proyecto (encargo, Dev 1-4, revisión técnica, benchmarks, verificación con cuenta real, y el reto oficial de Altur) en orden cronológico. Antes existían 12 documentos separados (`CLAUDE-CODE-ENTREGA.md`, `DEV1-DEV2-REFERENCIA.md`, `INTEGRACION.md`, `ALFA-REVISION.md`, `DEV3-ENTREGA.md`, `DEV4-ENTREGA.md`, `REVISION-TECNICA.md`, `CAMBIOS-Y-VALIDACION.md`, `BENCHMARK-ANTES-DESPUES.md`, `LISTO-PARA-EL-JUEZ.md`, `GUION-DEMO.md` y este mismo README); se fusionaron aquí y se eliminaron para no duplicar información.

---

## Arranque rápido (estado actual)

```powershell
cd "ruta\a\Altur_project"
.\.venv\Scripts\Activate.ps1
python -m pytest -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8025
```

Abrir `http://127.0.0.1:8025/demo` (panel visual) o `http://127.0.0.1:8025/docs` (API interactiva). Si no existe `.venv` todavía:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-lock.txt -r requirements-dev.txt
```

Para que un juez o un dispositivo en la misma red pueda alcanzar el servidor, usar `--host 0.0.0.0` en vez de `127.0.0.1` (ver sección 12).

**Contrato de `/detect`** (coincide con el oficial de Altur):

```json
// Petición
{"call_id": "opcional", "audio_base64": "<WAV completo en base64>", "sample_rate": 8000, "channels": 2}
// también acepta {"audio": "..."} como alias de audio_base64
```
```json
// Respuesta
{"is_synthetic": true, "confidence": 0.87}
```

WAV estéreo PCM16 de 8 kHz: canal 0 = cliente (a clasificar), canal 1 = agente. `confidence` es la **confianza en el veredicto devuelto** (si `is_synthetic=true`, es P(sintético); si es `false`, es P(humano) = 1 − P(sintético)) — no la probabilidad cruda del modelo sin ajustar (ver sección 7.7, fue un error real que se corrigió). `call_id`/`sample_rate`/`channels` se ignoran de forma segura si se envían (Pydantic descarta campos extra); el formato real del WAV siempre se valida contra el archivo decodificado, nunca contra esos metadatos declarados.

**Últimas cifras verificadas** (71 llamadas de val, con el script oficial del juez): `balanced_accuracy: 0.945`, `auc: 0.980`, `brier: 0.046`, 0 errores, latencia media 100-150 ms (máxima 228 ms, límite del juez: 30 s). Esto es val, no el conjunto oculto de evaluación — ver sección 11.

**Módulos:** `app/` (API y clasificador, Dev 1+2), `ml/` (features y ensemble, Dev 2), `conversation/` (análisis temporal y semántico, Dev 3 — antes `dev3/`), `ops/` (auditoría, demo, voz, Postgres, Dev 4 — antes `dev4/`). Ver sección 12 sobre el renombrado.

---

## 1. El encargo y el reto de Altur

Altur (Tecnologías Altur S.A.P.I. de C.V.) construye agentes de voz con IA que atienden llamadas para bancos de América Latina. El reto de HackMTY26 (*"Defend the Bank Against Voice Deepfakes"*) pide un sistema que detecte si quien llama a un banco es una persona real o una voz sintética, a partir de una llamada grabada entre el cliente y el agente. El material oficial sugiere tres enfoques posibles —no obligatorios—: detección acústica (la voz en sí), comportamiento conversacional (cómo reacciona el cliente a interrupciones y silencios del agente) y semántico (si el cliente inventa respuestas sobre datos que no existen).

El equipo (Dev 1 a Dev 4) recibió el dataset oficial (llamadas grabadas, español, 8 kHz estéreo, canal 0 = cliente/canal 1 = agente) y construyó el sistema en las secciones 2-5. Más adelante se hizo una revisión técnica a fondo para depurar y dejar reproducible ese trabajo (sección 6 en adelante), con una condición explícita del equipo: la conexión real con Gemini queda pendiente por decisión propia, sin buscar credenciales para esa fase.

Repositorio oficial del reto: https://github.com/alturio/hackmty26. Dataset oficial: `manifest.csv` + `turns/` (en ese mismo repo) + `audio/` (release `v1.0`, `altur-challenge-audio.zip`, ~640 MB) — no se redistribuye, es exclusivo de HackMTY26.

## 2. Dev 1 + Dev 2 — API y clasificador baseline

Backend en FastAPI. `app/main.py` expone `/health`, `/ready`, `/model`, `/detect`, `/docs`; `app/audio.py` decodifica y valida el WAV (base64 → estéreo PCM16 8kHz, límite de 12 MiB decodificado); `app/limits.py` acota el cuerpo HTTP (~16 MiB) antes de parsear JSON, incluso sin `Content-Length`.

El clasificador (`app/model.py`, `ml/features.py`, `ml/ensemble.py`) **no es un detector acústico/espectral**: extrae 87 características derivadas únicamente de la *dinámica de turnos* de la conversación (duración de intervenciones, latencias de respuesta, solapamientos, silencios, entropía de esos patrones), y las alimenta a un ensemble de voto suave (regresión logística + random forest + gradient boosting). Se entrenó con 282 llamadas de train; las 71 de val nunca entraron al ajuste.

**Hallazgo clave de esta etapa:** entrenar con los turnos "oficiales" del dataset y predecir con turnos detectados por el propio VAD del proyecto genera un desajuste de distribución (comprobado con datos reales):

| Entrenamiento / validación | AUC val | Aciertos |
| --- | --- | --- |
| Turnos oficiales / turnos oficiales | 0.9968 | 69/71 |
| Turnos oficiales / VAD de WAV | 0.9428 | 65/71 |
| VAD de WAV / VAD de WAV | 0.9801 | 67/71 |

Por eso el modelo entregado se entrenó y evalúa con el mismo VAD que usa la API en producción (última fila), no con los turnos oficiales — mismo preprocesamiento en entrenamiento e inferencia. Resultado: **67/71 (94.37%)** por HTTP, 0 errores, ~47 ms de latencia media / 59 ms p95 (macOS ARM64, Python 3.9.6, medición histórica).

Otras decisiones de esa etapa: `confidence` se omitió del contrato hasta aclarar su semántica con Altur (se resolvió después, sección 7.7); un error nunca produce un veredicto ficticio (se eliminó una ruta previa que devolvía sintético/confidence=0.5 ante cualquier excepción); el VAD sin habla en el canal del cliente devuelve 422; `models/model.pkl` y `models/model.json` guardan el modelo y su procedencia; `models/registry.json` guarda las identidades (hash, versión de sklearn) verificadas al cargar.

## 3. Revisión del modelo Alfa

Un compañero de equipo (Dev 2) entregó una variante alternativa, `models/dev2Alfa.pkl` (SHA-256 `4a187789ecf0b93a16416810d8b8fc86f1351cdd7861114380b2fbb82dae09c1`). Al auditarla (`audit_alfa.py`, `audit-alfa.json`) se encontró evidencia fuerte de que su `scaler` interno se ajustó con estadísticas de las 353 llamadas completas (train+val): su vector de medias coincide, con tolerancia 1e-10, con las features de *todos* los turnos oficiales — no con las extraídas del audio real. Esto hace que su resultado sobre val sea diagnóstico, no una evaluación independiente (marca 71/71 en turnos oficiales, pero ese número no demuestra generalización).

| Variante | Aciertos (WAV real, HTTP) | Humanos→sintético | Sintéticos→humano | Media | p95 |
| --- | --- | --- | --- | --- | --- |
| baseline | 67/71 | 3 | 1 | 47.1 ms | 60.0 ms |
| alfa | 64/71 | 6 | 1 | 47.0 ms | 59.2 ms |

**Decisión:** baseline permanece como predeterminado (`MODEL_VARIANT=baseline`) porque conserva un holdout real y obtuvo mejor resultado con audio real; Alfa queda integrada y seleccionable (`MODEL_VARIANT=alfa`) para comparación, sin representar una mejora comprobada. `models/registry.json` documenta ambas identidades; el registro no debe editarse para aceptar pesos nuevos sin auditarlos primero.

## 4. Dev 3 — análisis temporal y semántico

Módulo `conversation/` (originalmente `dev3/`): `temporal.py` normaliza intervalos de actividad de voz por canal (semiabiertos `[start,end)`), calcula latencias de respuesta **con signo** (negativas si el cliente empieza antes de que el agente termine — solapamiento real), solapamientos, silencios, reinicios tras interrupción, y produce 26 características descriptivas versionadas `temporal-v1`. `semantics.py` conecta con Gemini para transcribir canales por separado o revisar transcripciones dadas, con citas verificadas literalmente contra el texto y escenarios de prueba explícitos (`traps`) para detectar si el cliente inventa respuestas sobre datos ficticios — nunca declara que algo "no existe" sin un escenario de prueba proporcionado.

Rutas nuevas: `GET /conversation/status`, `POST /conversation/analyze` (mismo audio que `/detect`, con `include_semantics` opcional), `POST /conversation/semantic` (turnos transcritos + `traps`). Ninguna de estas 26 características ni la capa semántica afecta el veredicto de `/detect` (`affects_detect: false` siempre).

**Comparación con evidencia:** se compararon las 87 features de Dev 2 contra 113 (87+26) con el mismo ensemble e hiperparámetros, 5 folds estratificados sobre train (semilla 42), sin tocar val. AUC medio: 0.987642 (87) vs. 0.989138 (113) — mejora de solo 0.0015, por debajo del umbral de 0.005 fijado *antes* de medir. **Por eso las 26 features nunca se integraron a producción** — la disciplina de "profundidad antes que bulto" se mantuvo incluso cuando el propio equipo las había construido.

Estado de Gemini en esa etapa: configuración local presente, pero ninguna respuesta real exitosa confirmada (`gemini-2.5-flash` → 404; `gemini-3.8-flash` → 503 y 400 sin causa aislada). `configured` solo verifica variables; `live_verified` seguía en `false`. Pendiente de decisión del equipo, no de código — se retomó y se resolvió en la sección 10, pero solo para ElevenLabs; Gemini sigue sin credenciales por decisión expresa del equipo.

## 5. Dev 4 — auditoría, demo, voz y PostgreSQL

Módulo `ops/` (originalmente `dev4/`): `audit.py` usa SQLite con WAL y una cola acotada de 1024 eventos en segundo plano, registrando identificador, fecha, código HTTP, veredicto, probabilidad interna, latencia, duración e identidad del modelo — nunca audio, transcripciones ni secretos. Si SQLite falla al abrir, `/detect` sigue funcionando y las rutas de auditoría devuelven 503. `demo.html` (ruta `/demo`) es el panel web: sube un WAV, ve el veredicto, los registros y controles de voz. `voice.py` genera una alerta de audio fija vía ElevenLabs (implementado y probado solo con mocks en esa etapa). `postgres.py` exporta de forma explícita e idempotente (por `request_id`) a PostgreSQL/Tiger Data, nunca desde el camino de `/detect`.

Rutas de operador (`/audit/stats`, `/audit/calls`, `/voice/status`, `/voice/alert`): requieren `ADMIN_TOKEN` si está configurado; si no, solo aceptan clientes loopback.

**Evidencia de esa entrega:** 81 pruebas automáticas aprobadas; 71 WAV de val por HTTP con 67 aciertos (3 falsos positivos, 1 falso negativo); latencia media 47.2 ms / p95 59.1 ms; 83 eventos persistidos (79 respuestas 200, 4 errores 422, 0 descartes). Material de presentación: `Pitch-Altur.pptx` (5 diapositivas) con guion de 3 minutos — problema, cómo funciona, resultados (67/71), demo en vivo, cierre — más un recorrido técnico de 15 minutos repartido entre los cuatro devs.

## 6. Revisión técnica a fondo

A partir de aquí, el trabajo pasó a revisar, depurar, optimizar y dejar reproducible todo lo anterior para el reto — con evidencia medida, no solo recomendaciones. Regla explícita del equipo: la conexión real con Gemini queda fuera de esta fase, sin buscar credenciales. Línea base reproducida primero: **81/81 pruebas** en un entorno nuevo (Windows, Python 3.11.9, mismo `requirements-lock.txt`), confirmando que el estado heredado (macOS ARM64, Python 3.9.6) es reproducible en otra máquina.

El dataset oficial no estaba disponible en ese momento (se buscó y no estaba en la máquina), lo que limitó la primera ronda de revisión a análisis de código y pruebas con WAV sintéticos generados localmente (`work/gen_synth_wav.py`) — nunca usados para afirmar exactitud, solo formato y latencia.

## 7. Revisión técnica: hallazgos y correcciones

Nueve hallazgos confirmados en total; se listan en el orden en que se encontraron.

**1. `verify_dev3_http.py` podía llamar a Gemini real y fallar por eso (Alto, corregido).** El script asumía sin comprobar que el servidor no tenía `GEMINI_API_KEY`; si la tenía, disparaba una llamada real al proveedor y el assert fallaba de forma confusa. Corrección: ahora lee `/conversation/status` primero y omite la verificación explícitamente (`skipped: true`) si detecta Gemini configurado, en vez de contactar al proveedor.

**2. El mismo script sobrescribía evidencia histórica (Medio, corregido).** Escribía siempre en `verification-dev3-http.json`, destruyendo el reporte anterior. Se agregó `--report` (por defecto `verification-dev3-http-new.json`).

**3. `ops/audit.py`: el contador `written` no distinguía inserciones reales de duplicados (Bajo/Medio, corregido + test).** `INSERT OR IGNORE` no lanza excepción ante un `request_id` repetido; el contador se incrementaba igual sin mirar `cursor.rowcount`. Ahora `written` solo cuenta filas realmente insertadas, y se agregó `duplicate_ignored` para que la distinción sea visible en `/audit/stats`. Prueba: `test_audit_write_counter_distinguishes_duplicate_request_id`.

**4. `ops/postgres.py`: conexión SQLite sin cerrar y sin manejo de tabla ausente (Medio, corregido + test).** `with sqlite3.connect(...) as local:` solo controla la transacción en el módulo estándar `sqlite3`, no cierra la conexión; y si `sync_batch()` corre contra una base nunca inicializada, `SELECT * FROM calls` lanzaba `OperationalError` sin capturar. Corrección: `try/finally: local.close()`, y ese error se traduce a `{"status": "unavailable", "reason": "audit_db_not_initialized"}`. Prueba: `test_postgres_sync_on_uninitialized_db_reports_status_not_crash`.

**5. `ops/demo.html`: el refresco periódico podía reactivar "Generar alerta" durante una solicitud en curso (Bajo, corregido).** El `setInterval` de 3 segundos podía reactivar el botón mientras una generación seguía pendiente, permitiendo un doble clic con costo real de crédito en ElevenLabs. Corrección: bandera JS `voiceBusy` que el refresco respeta.

**6. `latency_frac_negative` en las 87 features es estructuralmente siempre 0 (Medio, corrección lista pero no aplicada al modelo desplegado).** La fórmula original solo empareja un turno del cliente con turnos del agente que *ya terminaron* antes de que el cliente empiece, así que la fracción de latencias negativas nunca puede ser mayor que cero — ni siquiera con solapamiento real. No se corrigió directamente porque `models/model.pkl` y `models/dev2Alfa.pkl` se entrenaron exactamente contra esa definición; cambiarla sin reentrenar alimentaría al modelo con una distribución que nunca vio. Se implementó como parámetro opt-in: `extract_features_from_turns(..., latency_pairing="legacy"|"signed_v2")`, con `"legacy"` por defecto (idéntico byte a byte al comportamiento anterior; `Detector` sigue llamando sin este argumento, cero impacto en producción) y `"signed_v2"` con el emparejamiento correcto (misma definición que `conversation/temporal.py`). `train_validate.py` acepta `--latency-pairing` para producir la próxima generación de pesos.

  **Se probó en cuanto el dataset oficial estuvo disponible** (ver sección 8): `python train_validate.py --data-root altur-data --latency-pairing signed_v2 --out models/model_v2_signed_latency.pkl`. Resultado sobre las 71 llamadas de val, mismo protocolo que el modelo real (entrena y evalúa con VAD de audio): accuracy y matriz de confusión **idénticas** al baseline (67/71, `[[34,3],[1,33]]`), ROC-AUC 0.9801 → 0.9809 (+0.0008), Brier 0.0461 → 0.0499 (ligeramente peor). No superó el umbral de 0.005 ya establecido por el equipo — **no se promovió a producción**, igual disciplina que con las 26 features de Dev 3. El artefacto queda en `models/model_v2_signed_latency.pkl`/`.json` como evidencia reproducible, sin tocar el registro.

**7. `confidence` invertía el AUC según el script oficial del juez (Alto, corregido).** Ver sección 9 — se detectó al activar `confidence` por primera vez.

**8. Un `422` por falta de habla del cliente cuenta como respuesta incorrecta en el conjunto oculto (Medio, riesgo confirmado, sin resolver).** El contrato del juez trata cualquier status ≠ 200 como fallo. No ocurrió nunca en las 71 llamadas de val, pero el conjunto oculto tiene voces nuevas donde podría pasar. No se resolvió unilateralmente porque implica una decisión de producto (¿fallar limpio, o arriesgar un veredicto por defecto sin justificación?), no solo de código.

**9. Puertos inconsistentes en la documentación heredada (Bajo, consolidado).** Distintas etapas usaban 8018-8023; se consolidó todo en 8025.

Puntos revisados sin defecto: límite de cuerpo HTTP (probado con 13 y 17 MiB, con y sin `Content-Length`); contrato `audio`/`audio_base64` contradictorios (422 correcto); verificación de modelo/sklearn/columnas/clases al arrancar; cierre de `AuditStore` ante fallo de SQLite; Docker (no se pudo construir, sin Docker instalado en el entorno de revisión).

## 8. El dataset oficial llega, y el fix de latencia se prueba de verdad

El dataset oficial (`manifest.csv` + `turns/` de `alturio/hackmty26`, audio del release `v1.0`) se consiguió durante la revisión. Se verificó su integridad (353 filas: 282 train/71 val, 203 sintéticas/150 humanas; 0 turns o audios faltantes) con `work/check_dataset.py`, y se usó para completar el hallazgo #6 (arriba) y para correr el script oficial del juez (sección 9).

## 9. El contrato oficial se actualiza: `confidence` y el script del juez

Altur publicó en `alturio/hackmty26` el contrato exacto de `/detect` (`call_id`/`audio_base64`/`sample_rate`/`channels` en la petición; `is_synthetic`/`confidence` en la respuesta; 30 s máx. por llamada; cualquier status≠200 o respuesta sin `is_synthetic` booleano cuenta como incorrecta) y dos scripts: `scripts/check_endpoint.py` (el mismo cliente que usa el juez) y `scripts/example_server.py`. Se descargaron a `work/altur_official/`.

**Primera corrida** (servidor sin `confidence`, 71 llamadas de val): `balanced_accuracy: 0.945`, 0 errores, latencia máxima 228 ms — confirma con el harness real del juez lo que ya se sabía (67/71).

Dado que el contrato ya definía el uso de `confidence` (AUC, calibración, desempate), se activó por primera vez enviando `p_synthetic` sin ajustar. Resultado: **`auc: 0.442`, peor que azar**. La causa: `check_endpoint.py` reconstruye `P(sintético)` como `confidence` si `is_synthetic=true`, o `1 - confidence` si es `false` — el contrato espera la **confianza en el veredicto devuelto**, no la probabilidad cruda. Corrección en `app/model.py`: `confidence = p_synthetic si is_synthetic, si no 1 - p_synthetic`. Reevaluado: **`auc: 0.980`, `brier: 0.046`** — coincide exactamente con las métricas internas ya conocidas (`models/model.json`: ROC-AUC 0.9801, Brier 0.0461). El modelo siempre fue así de bueno; el defecto era solo de exposición en la respuesta pública. `balanced_accuracy` nunca cambió (0.945), porque `is_synthetic` no depende de `confidence`. Este error **solo se detectó por probar con el script real del juez** antes de dar por buena la activación — una prueba contra herramientas propias nunca lo habría revelado. Prueba nueva: `test_classification_contract_and_internal_probability` verifica `confidence >= 0.5` y su relación exacta con `p_synthetic` según el veredicto.

## 10. Interfaz de demo mejorada

El panel `/demo` se rediseñó dos veces sobre la marcha:

1. Se agregó una sección de "Análisis temporal" con una línea de tiempo SVG (barras de cliente/agente a escala real, solapamientos resaltados) y las métricas de `conversation/analyze` (turnos, % de habla, % de solapamiento, latencia de respuesta), con las mismas advertencias de no-sobreinterpretación que ya usa la API.
2. A pedido del usuario, se simplificó: ahora lo primero que se ve es un veredicto grande y claro (🤖/🧑) con solo 4 datos clave (modelo, latencia, duración, % de solapamiento), y el resto de los datos quedan detrás de un botón "Ver más datos".

De paso se encontró y corrigió un bug real: `/demo` leía el HTML sin especificar `encoding="utf-8"`, y en Windows Python usa la codificación local del sistema por defecto — corrompía todos los acentos (`"AuditorÃa"` en vez de `"Auditoría"`). Un problema de compatibilidad multiplataforma preexistente, no introducido por este cambio; corregido en `app/main.py`.

## 11. ElevenLabs: verificación con cuenta real

A diferencia del resto de la revisión, esta parte sí se probó con una cuenta y credenciales reales del equipo (configuradas directamente en la terminal, nunca compartidas fuera de ahí). Resultado final: `POST /voice/alert` devolvió `200` con un MP3 real y válido (ID3 v2.4.0, MPEG layer III, 128 kbps, 44.1 kHz, ~188 KB).

El camino pasó por cuatro causas de error de cuenta distintas, cada una diagnosticada con la razón que ya devolvía `ops/voice.py` (y, cuando hizo falta más detalle, con un `print` de depuración temporal — solo en la terminal del servidor, nunca en la respuesta HTTP, y ya retirado del código):

1. **`401`, `api_key_id_used_as_api_key`** — se usó el *ID* de la key (visible en la tabla del dashboard) en vez de la key secreta real (siempre empieza con `sk_`).
2. **`401`, `missing_permissions`** — key con formato correcto pero sin el permiso de *Text to Speech* habilitado.
3. **`401`, `detected_unusual_activity`** — ElevenLabs deshabilitó el Free Tier de la cuenta por actividad "inusual" (su sistema antiabuso); se resolvió solo, sin cambiar nada, después de un rato.
4. **`402`, `payment_required`/`paid_plan_required`** — cuentas Free Tier no pueden usar voces de la librería pública vía API; se resolvió agregando una voz a "My Voices" y usando ese ID.

En ningún momento hizo falta cambiar `ops/voice.py`: el adaptador ya distinguía y exponía correctamente cada código de estado. El diagnóstico fue enteramente de configuración de cuenta, no de código.

## 12. Revisión final contra el reto oficial y sus criterios de jueces

Se leyó el documento confidencial del reto (`hackmty26-altur-challenge.pdf`, Tecnologías Altur S.A.P.I. de C.V., agosto 2026 — no se sube al repo por ser público y el PDF estar marcado confidencial) y se comparó todo el proyecto contra los **criterios de jueces publicados**: Robustez, Originalidad, Profundidad técnica, Viabilidad y Latencia.

- **Robustez:** lo único medible es val (0.945 balanced accuracy); el desempeño en el conjunto oculto (voces y personas nuevas) es desconocido hasta la evaluación en vivo — así se debe presentar, no como garantía. El Q&A oficial (sección 14) aclaró que el conjunto oculto usa el mismo motor de síntesis que train/val, solo con voces distintas — reduce la incertidumbre (no hace falta generalizar a una técnica de deepfake nunca vista), pero no la elimina.
- **Originalidad:** punto fuerte no explicitado hasta esta revisión — el modelo usa señales de comportamiento conversacional, no un clasificador acústico convencional (ver sección 2).
- **Profundidad técnica:** dos experimentos rechazados con evidencia y un umbral fijado antes de medir (26 features de Dev 3, y `latency_pairing=signed_v2`), más un error real de semántica (`confidence`) detectado y corregido con el script oficial del juez, no con herramientas propias.
- **Viabilidad:** CPU únicamente, sin modelo pesado, formato de entrada igual al telefónico real.
- **Latencia:** 100-200 ms por llamada medido con el script del juez, muy por debajo del límite de 30 s.

**Lo que el PDF reveló y no estaba resuelto:** los jueces visitan la mesa del equipo 15 minutos y corren su benchmark en vivo contra el endpoint — **debe ser alcanzable durante ese lapso**, algo que ningún documento anterior había resuelto. En ese momento se verificó `--host 0.0.0.0` (red local) y un túnel (ngrok) como opciones, sin decidir cuál. **Esto quedó resuelto y superado por la sección 14**: el Q&A oficial del evento confirma que el benchmark corre contra un despliegue público real, no la laptop del equipo por red local.

## 13. Reorganización del repositorio

Dos limpiezas estructurales, ambas a pedido explícito del usuario:

1. **Promoción a la raíz.** El trabajo vivía en una carpeta `integracion-final/` (para no tocar el esqueleto original del equipo mientras se revisaba en una rama). Se movió todo el contenido a la raíz del repositorio, reemplazando `core/`, `services/`, `utils/`, `main.py` y `requirements.txt` (stubs vacíos del esqueleto original) y eliminando `dev1/` (la entrega individual de un compañero, ya cubierta por la integración completa). Nada se perdió: sigue disponible en el historial de git.
2. **Sin carpetas "dev".** `dev3/` → `conversation/`, `dev4/` → `ops/` (nombres que describen la función, no un número de desarrollador). Se actualizaron todas las importaciones de Python, el `Dockerfile`, y las rutas mencionadas en la documentación vigente.

Verificado en ambos casos con un clon completamente nuevo desde GitHub, instalación desde cero y **88/88 pruebas en verde**.

## 14. Q&A oficial del evento y despliegue público

Los organizadores aclararon por separado (no en el PDF ni el README de `alturio/hackmty26`, sino en una ronda de preguntas del evento) varios puntos que cambian la prioridad operativa del proyecto:

| Pregunta | Respuesta oficial | Qué implica para nosotros |
| --- | --- | --- |
| ¿El benchmark corre desde la laptop del equipo o por red desde la del juez? | Desde la del juez, contra un **despliegue público real** — de preferencia en Vercel, Render o Railway | Las opciones de red local/túnel de la sección 12 quedan descartadas como plan principal: hace falta un despliegue de verdad, no la laptop del equipo |
| ¿Corren el benchmark antes o después de la explicación? | **Antes** | El servicio debe estar arriba y probado con anticipación, no algo que se levante mientras se explica la solución |
| ¿Cuántas llamadas manda el benchmark? | ~100 llamadas | A 100-200 ms por llamada (sección 12), el benchmark completo toma segundos si el servicio ya está "caliente" — el riesgo real es el primer arranque en frío (ver abajo) |
| ¿Las voces del conjunto oculto son de otros motores de síntesis o el mismo con voces distintas? | **El mismo motor**, voces distintas | Reduce la incertidumbre de "Robustez" (sección 12): el reto no exige generalizar a técnicas de deepfake nunca vistas, solo a hablantes nuevos del mismo motor ya representado en train/val |
| ¿Qué debe incluir la entrega en Devpost? | Repositorio y la URL del endpoint desplegado | Confirma que el despliegue no es opcional: sin URL pública no hay entrega completa |

### Cómo desplegar (pendiente de que el equipo cree la cuenta y lo ejecute)

No se puede completar esta parte desde aquí: requiere crear una cuenta en la plataforma elegida y conectar el repositorio, algo que le corresponde hacer al equipo. Lo que sí se dejó listo:

- **`Dockerfile` corregido** para respetar el puerto que asigna la plataforma (`$PORT`) en vez de tener el `8000` fijo — sin este cambio, Render/Railway no habrían podido enrutar tráfico al contenedor.
- **`render.yaml`** en la raíz: un blueprint mínimo (build por Docker, healthcheck en `/health`, `MODEL_VARIANT=baseline`) para que Render detecte la configuración automáticamente al conectar el repo.

**Pasos para el equipo (Render, recomendado por ser el más directo con un `Dockerfile` ya listo):**

1. Crear cuenta en https://render.com (gratis) y conectar la cuenta de GitHub.
2. "New +" → "Blueprint" → seleccionar este repositorio → Render debería detectar `render.yaml` solo.
3. Si se prefiere sin blueprint: "New +" → "Web Service" → seleccionar el repo → Runtime "Docker" → Health Check Path `/health`.
4. Variables de entorno opcionales (`GEMINI_API_KEY`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `DATABASE_URL`, `ADMIN_TOKEN`) se agregan en el dashboard si se quieren activar — ninguna es necesaria para que `/detect` funcione.
5. Al terminar el build, Render da una URL pública (`https://algo.onrender.com`). Probarla con el script oficial del juez **antes** del evento:
   ```bash
   python work/altur_official/check_endpoint.py --url https://algo.onrender.com/detect --manifest /ruta/al/altur-data/manifest.csv --audio-dir /ruta/al/altur-data/audio --split val --n 20
   ```

**Alternativa: Railway** (https://railway.app) — conectar el repo, Railway detecta el `Dockerfile` solo, no necesita un archivo de blueprint propio.

**Riesgo real y no resuelto: arranque en frío.** Los planes gratuitos de estas plataformas duermen el servicio tras un rato sin tráfico; la primera solicitud después de dormir puede tardar 30-60+ segundos en responder — suficiente para exceder el límite de 30 s **de esa primera llamada** y contar como fallo, aunque las siguientes 99 respondan en 150 ms. Dado que el benchmark corre *antes* de la explicación (según el Q&A), no hay margen para "calentar" el servicio hablando primero. Mitigación recomendada: mandar una solicitud de prueba (`GET /health` o un `/detect` de prueba) un par de minutos antes de que el juez llegue a la mesa, y considerar si vale la pena un plan pago de arranque instantáneo dado el costo.

---

## Referencia técnica

### Rutas de la API

| Ruta | Método | Entrada | Resultado |
| --- | --- | --- | --- |
| `/health` | GET | — | Proceso vivo |
| `/ready` | GET | — | 200 solo si el modelo cargó, verificó identidad/versión/columnas/clases y pasó una predicción numérica de prueba |
| `/model` | GET | — | Variante, SHA-256, número de features, origen del entrenamiento, si conservó holdout |
| `/detect` | POST | WAV base64 en `audio`/`audio_base64` (+ `call_id`/`sample_rate`/`channels` opcionales, ignorados) | `{"is_synthetic": bool, "confidence": float}` |
| `/conversation/status` | GET | — | Disponibilidad temporal y estado de configuración Gemini |
| `/conversation/analyze` | POST | Igual que `/detect`, más `include_semantics` opcional | Turnos, silencios, solapamientos, latencias con signo, 26 features, `affects_detect: false` |
| `/conversation/semantic` | POST | `turns` transcritos y `traps` opcionales | Reacciones con citas e índices verificados |
| `/audit/stats`, `/audit/calls` | GET | — | Requiere token de operador o loopback |
| `/voice/status`, `/voice/alert` | GET/POST | — | Requiere token de operador o loopback |
| `/demo` | GET | — | Panel HTML de la demo |

Errores: `422` (WAV inválido, formato incorrecto, sin habla detectable en el canal del cliente, o `audio`/`audio_base64` contradictorios — ver hallazgo #8 sobre el riesgo de esto en el conjunto oculto), `413` (cuerpo excede ~16 MiB), `503` (falta el modelo, o auditoría no disponible — `/detect` sigue funcionando aunque falle la auditoría). Un error nunca produce un veredicto ficticio.

### Ejemplo (Python)

```python
import base64, httpx

wav_bytes = open("llamada.wav", "rb").read()
payload = {"audio": base64.b64encode(wav_bytes).decode()}
response = httpx.post("http://127.0.0.1:8025/detect", json=payload, timeout=30)
print(response.status_code, response.json())
# 200 {'is_synthetic': False, 'confidence': 0.94}
```

`work/gen_synth_wav.py` genera WAV sintéticos (senoides, no habla real) para probar el formato sin el dataset oficial — nunca para afirmar exactitud.

### Verificación

```bash
python -m pytest -q
python verify_real_http.py --data-root /ruta/al/altur-data --url http://127.0.0.1:8025 --report verification-new-http.json
python verify_dev3_http.py --data-root /ruta/al/altur-data --url http://127.0.0.1:8025
python work/altur_official/check_endpoint.py --url http://127.0.0.1:8025/detect --manifest /ruta/al/altur-data/manifest.csv --audio-dir /ruta/al/altur-data/audio --split val --n 0
```

Los scripts `verify_*` aceptan `--report` para no sobrescribir la evidencia histórica (`verification-*.json` en la raíz). `check_endpoint.py` es el script oficial del juez (`work/altur_official/`, bajado de `alturio/hackmty26` — volver a bajarlo si Altur lo actualiza).

### Variables de entorno

`MODEL_VARIANT` (`baseline`/`alfa`), `MODEL_PATH`, `AUDIT_DB_PATH`, `ADMIN_TOKEN`, `GEMINI_API_KEY`/`GOOGLE_API_KEY`, `GEMINI_MODEL`, `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `ELEVENLABS_MODEL_ID`, `DATABASE_URL`. Ninguna es obligatoria salvo para la integración correspondiente. Ver `.env.example`. **La aplicación no carga `.env` automáticamente** — hay que exportar las variables en la terminal antes de arrancar.

### Límites

- Cuerpo HTTP ≈16 MiB antes de parsear JSON (con y sin `Content-Length`); WAV decodificado 12 MiB; llamadas de 1-4 min y ~5 MB de cuerpo (contrato oficial) caben con margen.
- `/conversation/semantic`: hasta 512 KiB de cuerpo, 400 turnos, 60000 caracteres de texto.
- Proveedores externos: máx. 2 solicitudes simultáneas a Gemini, 1 a ElevenLabs; timeouts 20-30 s sin reintentos. Ninguno agrega latencia al camino de `/detect`.
- Docker: `docker build -t altur-hackmty .` — **no probado** (sin Docker en ningún entorno usado). Imagen en Python 3.12; las pruebas locales fueron en 3.11.9 (esta revisión) y 3.9.6 (histórico).
- Concurrencia: el acceso al modelo está serializado con un `Lock` (thread-safety de scikit-learn); throughput se estanca en ~16-18 req/s bajo carga concurrente independientemente del número de hilos — candidato de optimización futura, no resuelto porque requeriría confirmar primero que los tres estimadores del ensemble son seguros para invocación concurrente.

## Pendientes reales (sin inflar)

| Pendiente | Por qué sigue abierto |
| --- | --- |
| **Desplegar en Render/Railway/Vercel y probar la URL pública** | **Máxima prioridad ahora** (sección 14): el Q&A oficial confirma que el juez corre el benchmark contra un despliegue real, antes de la explicación, y Devpost exige la URL. Requiere que el equipo cree la cuenta — no se puede hacer desde aquí |
| Reentrenar con `latency_pairing=signed_v2` y promoverlo | Ya se probó (sección 7.6): no supera el umbral de mejora fijado — no es un pendiente técnico, es una decisión ya tomada con evidencia |
| `422` por audio sin habla podría contar como fallo en el conjunto oculto | Decisión de producto pendiente del equipo (sección 7.8) |
| Gemini con cuenta real | Pendiente por decisión expresa del equipo, no de código |
| PostgreSQL/Tiger Data con base real | Sin credenciales disponibles; probado solo con mocks |
| Docker/Linux en un entorno real | No hay Docker instalado en ningún entorno de desarrollo usado; el build real ocurre del lado de la plataforma de despliegue |
| Arranque en frío del plan gratuito de hosting | Puede exceder el límite de 30 s en la primera llamada si el servicio estaba dormido — mitigar con una solicitud de calentamiento antes del benchmark (sección 14) |

## Guion de demo (15 minutos con el juez)

1. **Problema y enfoque (originalidad):** "Nuestro modelo no analiza cómo suena la voz — analiza cómo se comporta la conversación: turnos, silencios, latencias de respuesta con signo, solapamientos."
2. **Disciplina técnica (profundidad):** "Probamos agregar más señales dos veces y las rechazamos ambas veces con un umbral fijado antes de medir, porque no mejoraban lo suficiente."
3. **Demo en vivo:** subir un WAV en `/demo`, mostrar el veredicto y los datos clave; con "Ver más datos", mostrar la línea de tiempo de solapamientos/silencios.
4. **Latencia:** "100-200 ms por llamada, medido con su propio script de verificación, no el nuestro."
5. **Viabilidad:** "Corre en CPU, sin modelo pesado — así lo correría un banco de verdad."
6. **Honestidad (refuerza credibilidad):** "Esto es val, no su conjunto oculto — no sabemos cómo nos va a ir ahí, y no vamos a prometer un número que no hemos medido."

Contingencias: si falla ElevenLabs/Gemini/Postgres en vivo, mostrar el estado real vía `/voice/status` / `/conversation/status` y explicar la razón exacta — nunca presentar un reporte guardado como si fuera ejecución en vivo. Si el servidor no responde, reiniciarlo con el comando de arranque rápido de arriba.
