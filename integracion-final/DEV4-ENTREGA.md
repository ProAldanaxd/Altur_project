# Dev 4 — entrega para Juan Carlos

Esta carpeta reúne Dev 1–4. Auditoría SQLite y demo web funcionan localmente. Los adaptadores ElevenLabs y PostgreSQL/Tiger Data están implementados y probados con simulaciones; falta conectarlos con cuentas reales. Gemini también sigue pendiente de credenciales. No hay despliegue público todavía.

## Arranque

Desde esta carpeta, con Python 3.9 o posterior:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8023
```

Demo: http://127.0.0.1:8023/demo . API: http://127.0.0.1:8023/docs . `/ready` confirma el modelo. Probado en macOS ARM64 con Python 3.9.6; el Dockerfile está preparado, pero todavía no se ha probado en Linux/Docker.

`.env.example` documenta variables; **la aplicación no carga `.env` automáticamente**. Exportarlas en la terminal antes de arrancar. No enviar claves por chat ni incluirlas en el ZIP.

## Qué incluye

| Componente | Implementación | Estado |
| --- | --- | --- |
| Auditoría | `dev4/audit.py`, SQLite con WAL y cola de escritura | Probado con HTTP y reapertura de la base |
| Panel | `dev4/demo.html`, ruta `/demo` | Subida de WAV, resultados, registros y reproducción de alerta |
| Voz | `dev4/voice.py`, ElevenLabs TTS | Contrato y errores probados con proveedor simulado |
| PostgreSQL | `dev4/postgres.py`, exportación idempotente | Transacción y reintentos probados con conexión simulada |
| Presentación | `Pitch-Altur.pptx`, 5 diapositivas editables | Incluye notas del presentador; ver `GUION-DEMO.md` |

## Auditoría y contrato

`POST /detect` sigue devolviendo `is_synthetic`. No agrega la probabilidad interna a ese JSON ni llama a Gemini, ElevenLabs o PostgreSQL. Cada respuesta lleva `X-Request-ID`, `X-Process-Time-Ms` y `X-Model-Variant`.

SQLite guarda identificador, fecha UTC, código HTTP, veredicto, probabilidad de voz sintética, latencia interna, duración y variante/hash del modelo. Los errores no reciben un veredicto ni una probabilidad ficticios. No se almacenan audio, transcripciones, cuerpos de solicitudes ni secretos.

`p_synthetic` es la probabilidad interna del clasificador; **no equivale a fraude demostrado ni a confianza calibrada**. La latencia registrada mide procesamiento del servidor; no incluye la red del cliente ni la escritura posterior en SQLite.

Rutas de operador: `GET /audit/stats`, `GET /audit/calls?limit=20`, `GET /voice/status` y `POST /voice/alert`. Si existe `ADMIN_TOKEN`, exigen `Authorization: Bearer <token>`. Sin token solo aceptan clientes loopback. Antes de desplegar detrás de un proxy, configurar un token y HTTPS; no basar el acceso remoto en la dirección que vea el proxy. `/detect` mantiene su acceso para el benchmark.

La cola tiene capacidad de 1024 eventos. Si se llena, el evento se descarta y aumenta `dropped`; si falla SQLite, aumenta `write_errors`. Estos fallos se muestran en el panel. El contador `written` es del proceso; `persisted.total` cuenta las filas de la base. Un cierre abrupto puede perder eventos todavía en memoria. No es un sistema de auditoría regulatoria ni almacenamiento sin pérdida. Si falla la apertura de SQLite, `/detect` sigue funcionando y las rutas de auditoría devuelven 503.

La base predeterminada es `data/audit.sqlite3`, relativa a la carpeta de arranque. Con Docker se necesita un volumen persistente en `/app/data` escribible por UID 10001; sin volumen los registros pueden perderse al reemplazar el contenedor.

## ElevenLabs

Configurar `ELEVENLABS_API_KEY` y `ELEVENLABS_VOICE_ID` de una voz habilitada en la cuenta. `ELEVENLABS_MODEL_ID` es opcional; valor predeterminado `eleven_multilingual_v2`. Reiniciar el servidor y comprobar `/voice/status`. `configured` solo confirma la presencia de configuración; una generación real verifica permisos y disponibilidad.

El botón del panel genera manualmente un MP3 y permite reproducirlo. Cada generación puede consumir crédito del proveedor. El texto es fijo y no contiene datos de la llamada. La integración usa una solicitud simultánea como máximo y timeout HTTP de 20 segundos, sin reintentos automáticos. La ausencia de configuración devuelve 503 sin llamar al proveedor. No realiza llamadas telefónicas salientes.

Texto completo: “Atención. Se detectaron señales de voz sintética. Solicita una verificación adicional de identidad antes de continuar. Esta alerta no confirma fraude.”

Referencia del proveedor: https://elevenlabs.io/docs/api-reference/text-to-speech/convert

## PostgreSQL / Tiger Data

Con `DATABASE_URL` exportada para la base del equipo:

```bash
python -m pip install -r requirements-postgres.txt
python -m dev4.postgres --db-path data/audit.sqlite3
```

Este comando exporta hasta 500 registros pendientes por ejecución a `altur_calls`. Ejecutarlo de nuevo para continuar o reintentar. Primero confirma la transacción remota y después marca las filas locales como exportadas. La clave primaria `request_id` evita duplicados cuando se reintenta. No borra los registros locales. No hay sincronización remota automática ni prueba real contra Tiger Data todavía.

## Evidencia de esta entrega

- 81 pruebas automáticas aprobadas.
- 71 WAV del conjunto público de validación por HTTP: 67 aciertos, 3 falsos positivos y 1 falso negativo. Modelo entrenado solo con 282 llamadas de train.
- Latencia local de esas 71 solicitudes: media 47.2 ms, percentil 95 de 59.1 ms. Es una medición de esta Mac, no una promesa para nube.
- 8 solicitudes concurrentes adicionales correctas a nivel HTTP y 4 entradas inválidas rechazadas con 422.
- 83 eventos persistidos: 79 respuestas 200 y 4 errores; cero descartes y cero errores de escritura. Las 8 llamadas repetidas no aumentan el tamaño del conjunto de evaluación.

Reportes: `verification-dev4-http.json` y `verification-dev4-audit.json`. La base usada para la prueba y los WAV del dataset no se incluyen en el ZIP. Una nueva instalación inicia sin registros. `DEV3-ENTREGA.md` explica el análisis temporal y las limitaciones de Gemini; `ALFA-REVISION.md` documenta por qué baseline sigue siendo el modelo predeterminado.

## Próximos pasos de Juan Carlos

1. Conectar ElevenLabs y comprobar generación/reproducción real del texto fijo.
2. Crear la base del equipo, exportar un lote y verificar filas por `request_id`.
3. Ensayar los 3 minutos con `GUION-DEMO.md`, mostrando también los errores del modelo.
4. Coordinar con Dev 1 el despliegue, volumen persistente, token de operador, HTTPS y prueba externa del endpoint.
