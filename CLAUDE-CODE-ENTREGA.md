# Altur_Hackmty — contexto y encargo para Claude Code

## Encargo del equipo

Trabaja sobre este proyecto Python existente para revisarlo, depurarlo, optimizarlo y dejarlo reproducible para el reto Altur de HackMTY. Implementa mejoras con evidencia; no te limites a proponerlas. Conserva el trabajo del equipo y documenta cambios, pruebas, resultados y pendientes.

**La conexión real con Gemini queda fuera de la tarea por ahora.** El usuario ha pedido dejar la API key a un lado. Se pueden mejorar las pruebas locales y el manejo de errores del adaptador, pero no buscar credenciales, configurar cuentas ni hacer llamadas a proveedores para esta fase.

No se busca prometer perfección ni una precisión inventada. Prioriza que el endpoint respete el contrato oficial, que la evaluación sea honesta y que la ejecución sea fiable y rápida.

## Ubicación y material

- Carpeta actual en la Mac: `/Users/josemanuelramirezviveros/Documents/Codex/2026-09-11/ho/outputs/Altur_Hackmty`.
- Repositorio de referencia del organizador: https://github.com/alturio/hackmty26 . Verifica sus instrucciones vigentes antes de afirmar cumplimiento; las notas de este documento no sustituyen el contrato oficial.
- Dataset usado localmente: `/Users/josemanuelramirezviveros/Documents/Codex/2026-09-11/ho/work/altur-data`, con `manifest.csv`, `turns/` y `audio/`. Verifica que siga accesible. No se incluye dentro del programa para compartir.
- Hay copias anteriores en `outputs/altur-dev4` y otras carpetas. Trabaja en **Altur_Hackmty**, no en esas copias.
- El programa, los modelos, tests y reportes están en la carpeta. Este documento es el contexto para trabajar con ellos; no contiene una copia de todo el código.

## Estado verificado y límites

La entrega anterior pasó 81 pruebas automáticas en macOS ARM64 con Python 3.9.6. Este es un resultado histórico: vuelve a ejecutar la suite en la carpeta actual antes de modificar nada. No presentes el resultado histórico como una ejecución nueva.

La verificación HTTP de Dev 4, guardada en `verification-dev4-http.json`, usó 71 WAV de validación y produjo:

| Métrica | Resultado registrado |
| --- | --- |
| Aciertos | 67 / 71, 94.366 % |
| Humanos correctos | 34 / 37 |
| Sintéticos correctos | 33 / 34 |
| Falsos positivos | 3 humanos marcados sintéticos |
| Falsos negativos | 1 sintético marcado humano |
| Errores HTTP en esas 71 llamadas | 0 |
| Latencia HTTP media local | 47.2 ms |
| Percentil 95 local | 59.1 ms |

Se enviaron además 8 solicitudes concurrentes y 4 entradas inválidas. La auditoría histórica persistió 83 eventos: 79 respuestas 200 y 4 errores 422, sin descartes ni errores de escritura. Esas solicitudes adicionales no aumentan el número de ejemplos independientes del conjunto de evaluación.

No hay prueba de benchmark oculto, despliegue público, uptime en nube ni Docker/Linux completada. Las métricas de latencia son de esta Mac. No usar los reportes de llamadas completas para afirmar resultados sobre fragmentos de audio sin probarlos.

## Arquitectura actual

### Dev 1: API y entrada de audio

- `app/main.py`: FastAPI, ciclo de vida, endpoints, correlación y auditoría.
- `app/audio.py`: base64 y validación/decodificación del audio.
- `app/limits.py`: límite del cuerpo antes del procesamiento JSON.
- Entrada: WAV estéreo PCM16, 8 kHz. Canal 0 = cliente; canal 1 = agente.
- `POST /detect` acepta `audio` o `audio_base64`. Si llegan ambos con contenido distinto, se rechaza la petición.
- Límite de WAV decodificado: 12 MiB; el límite del cuerpo contempla la expansión base64. Revisa las constantes exactas del código.
- Respuesta actual de clasificación: `{"is_synthetic": true}` o `false`. Existe campo opcional `confidence` en el esquema, pero el baseline no lo devuelve.
- No añadir campos al contrato del benchmark sin verificar lo que permite Altur.
- Encabezados adicionales: `X-Request-ID`, `X-Process-Time-Ms`, `X-Model-Variant`.
- Rutas de diagnóstico: `/health`, `/ready`, `/model`, `/docs`.

### Dev 2: clasificador

- `app/model.py`: carga, identidad e inferencia del modelo.
- `ml/features.py`, `ml/ensemble.py`: extracción de características y ensemble. Inspecciona los estimadores concretos en el código.
- Baseline: 87 características temporales derivadas de actividad de voz/interacción; no describirlo como un detector espectral profundo entrenado si el código no lo es.
- Se entrenó con 282 llamadas de train. Baseline es la opción predeterminada.
- `MODEL_VARIANT=alfa` selecciona el modelo alternativo entregado por el compañero.
- `predict_details` expone internamente `is_synthetic` y `p_synthetic`; `predict` conserva la interfaz anterior.
- `p_synthetic` no es confianza calibrada ni probabilidad de fraude.

Identidades registradas:

| Modelo | Archivo | SHA-256 |
| --- | --- | --- |
| Baseline | `models/model.pkl` | `9aef6ea7a5efc0a4ca3488054f914c2c69acaaa1cba0f7535fdd0342e7472f2f` |
| Alfa | `models/dev2Alfa.pkl` | `4a187789ecf0b93a16416810d8b8fc86f1351cdd7861114380b2fbb82dae09c1` |

Consultar `models/registry.json`, `models/model.json`, `ALFA-REVISION.md` y `audit-alfa.json`. La revisión previa de Alfa encontró evidencia de ajuste del scaler con estadísticas de las 353 llamadas, incluida validación; además obtuvo 64/71 aciertos en WAV. No asumir que por ser una entrega más reciente es el mejor modelo ni que conserva un holdout independiente.

### Dev 3: conversación

- `dev3/temporal.py`: turnos, pausas, solapamientos, latencias con signo y posibles interrupciones; 26 características descriptivas adicionales.
- `POST /conversation/analyze`: mismo audio base64; análisis temporal. Semántica opcional mediante `include_semantics=true`.
- `POST /conversation/semantic`: turnos transcritos y escenarios ficticios opcionales (`traps`).
- `GET /conversation/status`: estado del módulo y presencia de configuración.
- `dev3/semantics.py`: adaptador Gemini, validación estructural y comprobación de citas/índices contra la transcripción.
- La semántica no debe clasificar por sí sola una voz como sintética ni confirmar fraude. Una premisa inventada necesita un escenario proporcionado; no inferir que un concepto no existe por desconocerlo.
- Se compararon 87 vs. 113 características usando CV en train. Mejora media AUC aproximada: 0.001496, inferior al umbral predefinido de 0.005. Por eso las 26 características nuevas no modifican `/detect`.
- Reportes: `dev3-training-evaluation.json`, `verification-dev3-http.json`; detalles en `DEV3-ENTREGA.md`.

Estado real de Gemini: hubo configuración local, pero ninguna respuesta semántica real exitosa confirmada. Las pruebas con `gemini-2.5-flash` devolvieron 404; con `gemini-3.8-flash`, 503 y 400 `INVALID_ARGUMENT`, con mensaje genérico. No se aisló la causa. `configured` solo verifica variables, y `live_verified` sigue siendo false. No repetir la afirmación de que Gemini ya funciona.

`check_gemini.py` contiene un diagnóstico por etapas con un diálogo ficticio. Requiere las variables de la Terminal del usuario y puede consumir solicitudes externas: déjalo pendiente en esta fase. Los nombres de modelos mencionados son historial de pruebas, no una recomendación vigente ni prueba de disponibilidad.

### Dev 4: auditoría, demo y pitch

- `dev4/audit.py`: SQLite WAL, cola acotada de 1024 eventos, escritura en segundo plano.
- Base predeterminada `data/audit.sqlite3`, configurable con `AUDIT_DB_PATH`.
- Registra identificador, fecha, código HTTP, veredicto, score interno, latencia, duración e identidad del modelo. No guarda audio, transcripciones, cuerpos ni claves.
- Contadores de pendientes, descartes y errores visibles. Una caída abrupta puede perder eventos en memoria. La cola no garantiza entrega sin pérdida.
- Si falla abrir SQLite, el clasificador sigue disponible y las rutas de auditoría devuelven 503.
- `/audit/stats` y `/audit/calls`: lectura para operador.
- `ADMIN_TOKEN`: Bearer requerido cuando está configurado; en su ausencia solo se admiten clientes loopback. Revisa este límite antes de cualquier despliegue con proxy.
- `dev4/demo.html`, ruta `/demo`: subir WAV, veredicto, registros y controles de voz.
- `dev4/voice.py`, `/voice/status`, `/voice/alert`: ElevenLabs TTS con texto fijo de verificación adicional. Implementado y probado con mocks; sin prueba real. Generación manual, no llamadas telefónicas salientes.
- `dev4/postgres.py`: exportación explícita a PostgreSQL/Tiger Data, con transacción y deduplicación por `request_id`. Probada con mocks, no con base externa real. No corre dentro de `/detect`.
- `Pitch-Altur.pptx`, `GUION-DEMO.md`, `DEV4-ENTREGA.md`: presentación, guion y entrega técnica. Actualizar afirmaciones solo después de nuevas pruebas.

## Ejecutar y establecer la línea base

Desde la carpeta del proyecto:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-lock.txt
python3 -m pytest -q
python3 -m uvicorn app.main:app --host 127.0.0.1 --port 8025
```

Si `.venv` ya existe y funciona, reutilízalo. Se sugiere 8025 para evitar los puertos usados previamente, pero verifica disponibilidad. No detengas procesos ajenos por liberar un puerto. Los comandos siguientes deben apuntar al servidor de esta carpeta, no a una copia antigua:

```bash
python3 verify_real_http.py --data-root /ruta/al/altur-data --url http://127.0.0.1:8025 --report verification-new-http.json
python3 verify_dev3_http.py --data-root /ruta/al/altur-data --url http://127.0.0.1:8025
```

Inspecciona primero las opciones de los scripts para guardar resultados nuevos sin sobrescribir la evidencia histórica. Usa una ruta SQLite temporal para pruebas que no deban mezclarse con registros del usuario. No borrar `data/` sin autorización.

`requirements-lock.txt` captura el entorno local anterior; `requirements.txt` usa algunos rangos más amplios. Revisa coherencia y compatibilidad antes de cambios de versión. Docker usa Python 3.12 y requiere verificación propia. `.env.example` es documentación: el programa no carga `.env` automáticamente.

## Prioridades de revisión e implementación

### 1. Corrección y contrato

Revisar README oficial, esquema real de `/detect`, formatos permitidos, frecuencia/canales, duración, límites, errores y necesidades del benchmark. Auditar WAV malformado, base64 inválido, silencio, señal casi silenciosa, canal sin voz y valores no finitos. No asumir que el endpoint cumple escenarios no probados.

Revisar alineación temporal y orden de características, clases del ensemble, persistencia/carga de modelos y coherencia entre entrenamiento e inferencia. Buscar fuga de datos, errores de segmentación, índices y unidades. Corregir defectos reproducibles con tests que fallen antes y pasen después.

### 2. Latencia y recursos

Medir con los mismos WAV, hardware y condiciones: calentamiento, p50/p95/p99, máximo, concurrencia y errores. Separar procesamiento interno de latencia HTTP. Perfilar antes de optimizar. Revisar cómputos repetidos, copias, carga de modelos, número de hilos y contención entre solicitudes.

Evitar bloquear el event loop, crecimiento ilimitado de memoria y contención innecesaria de SQLite. Mantener audio y secretos fuera de logs. Las dependencias opcionales externas no deben añadir latencia al camino de `/detect`.

### 3. Auditoría, errores y operación

Revisar cierre de conexiones SQLite en todos los módulos, exportación cuando no existe la base, contabilidad de escrituras, cola llena, disco bloqueado y apagado con eventos pendientes. Revisar fallos de lectura del panel y acceso de operador detrás de proxies. Son puntos de revisión, no defectos confirmados todos ellos.

Comprobar la separación entre `ready`, configuración y disponibilidad real de proveedores. No convertir respuestas fallidas en resultados positivos. Revisar el panel mientras una alerta está en curso y el refresco periódico actualiza los controles.

### 4. Calidad del clasificador

Si se modifica el modelo, usar CV dentro de train para seleccionar características, hiperparámetros y umbrales. Conservar el baseline reproducible como referencia. No optimizar sobre las 71 llamadas de val: ya se han consultado repetidamente y no son un test nuevo. Etiquetar claramente comparaciones exploratorias y reservar datos realmente nuevos si se necesitan conclusiones independientes.

Reportar matriz de confusión, precision/recall por clase, F1, ROC-AUC cuando proceda, latencia y errores. Solo hablar de confianza calibrada si se evalúa calibración. No agregar heurísticas semánticas al score sin medirlas con datos adecuados. No perseguir 100 % en el conjunto público a costa de sobreajuste.

### 5. Reproducibilidad y entrega

Revisar Docker, volúmenes, usuario sin privilegios y arranque limpio. Preparar instrucciones consistentes: hay documentación heredada que menciona `altur-dev3`, `altur-dev4`, puertos 8021/8023 y reportes anteriores. Actualizar la guía principal para `Altur_Hackmty` sin alterar evidencia histórica.

Probar los cambios en el entorno disponible; señalar explícitamente las pruebas externas que no se pudieron ejecutar. No desplegar públicamente, contratar servicios ni publicar el repositorio como parte implícita de esta revisión.

## Reglas para conservar el trabajo

- Inspecciona las instrucciones locales y el estado de Git si existe. No hagas reset ni descartes cambios ajenos.
- No cambiar pesos silenciosamente; registrar variante, hash, dataset, particiones, configuración y razón del cambio.
- No incluir `.venv`, caches, claves, `.env` con secretos, bases de auditoría activas ni audios del dataset en un paquete compartido.
- Una clave se compartió anteriormente en el chat: no se reproduce aquí ni debe recuperarse del historial. La gestión de credenciales se retomará por separado.
- No confundir TTS con telefonía outbound, metadatos con auditoría sin pérdida, ni una integración simulada con un servicio real verificado.
- Mantener una degradación explícita cuando falten proveedores; Dev 1/2 y análisis temporal local deben poder operar sin ellos.

## Resultado esperado de Claude Code

Entrega los cambios implementados junto con:

1. Hallazgos confirmados, severidad, impacto y causa.
2. Resumen de correcciones y justificación de decisiones.
3. Tests ejecutados y resultados nuevos, diferenciados de reportes históricos.
4. Comparación antes/después de rendimiento y clasificación cuando corresponda; condiciones de medición.
5. README de arranque y demo actualizado, con pendientes reales.
6. Lista breve de riesgos restantes y siguientes pasos prioritarios.

Comienza inspeccionando el repositorio y ejecutando la línea base. Después resuelve primero los defectos confirmados de mayor impacto. Gemini queda pendiente por decisión del usuario.
