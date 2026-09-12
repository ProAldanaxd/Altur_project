# Entrega Dev 1 — Backend y API Altur

## Estado real

Se entrega una base local de FastAPI que recibe audio, valida formato y separa canales. El clasificador NO está integrado. Un WAV válido devuelve 503, no una predicción. No hay despliegue público, dominio, credenciales de nube ni pesos del modelo en esta entrega. El Dockerfile está preparado, pero no se ha construido ni probado en Docker.

Se ejecutaron 10 pruebas automatizadas y 19 comprobaciones HTTP contra Uvicorn local, todas con el resultado esperado. La evidencia HTTP está en verification-http.json. Se usaron WAV generados para probar el formato, incluidos 273 segundos; no llamadas oficiales. Las pruebas con respuestas de clasificación usan un doble de prueba, nunca el servidor normal. No se ha medido precisión ni rendimiento con inferencia real.

## Responsabilidad de Dev 1

Mantener POST /detect compatible con los jueces, conectar el predictor de Dev 2, entregar canales a Dev 3 cuando corresponda y desplegar el servicio con disponibilidad y latencia verificadas. Vultr y un dominio .Tech son objetivos del equipo; todavía no están configurados.

## Archivos

| Archivo | Función |
| --- | --- |
| app/main.py | FastAPI, contratos Pydantic, rutas, carga del detector al iniciar y cabecera de tiempo |
| app/audio.py | Base64, WAV PCM16 de 8 kHz, validaciones y separación de canales |
| app/model.py | Interfaz pendiente de implementación por Dev 2 |
| tests/test_api.py | 10 pruebas automatizadas |
| tests/smoke_http.py | 19 comprobaciones contra un servidor HTTP real sin modelo |
| verification-http.json | Resultados guardados de la comprobación local |
| requirements.txt | Dependencias de ejecución con rangos de versiones |
| requirements-dev.txt | Dependencias de pruebas |
| Dockerfile | Imagen Python 3.12, libsndfile, usuario sin privilegios y Uvicorn |
| README.md | Instrucciones de uso, ejemplo de curl y fuentes |

## Arranque en la computadora del compañero

Descomprimir el ZIP, abrir una terminal en la carpeta altur-api y ejecutar:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Abrir http://127.0.0.1:8000/docs . La terminal debe permanecer abierta. En otra terminal, desde la misma carpeta y con el entorno activado:

```bash
python tests/smoke_http.py http://127.0.0.1:8000
```

El smoke test presupone que NO hay modelo y espera 503 en llamadas válidas. Actualizarlo cuando se integre el predictor. El servidor de la sesión original usaba 8017; ese puerto no es obligatorio. Una dirección 127.0.0.1 solo funciona en la computadora donde corre el servidor.

## Contrato actual

Entrada implementada, con nombre de campo PROVISIONAL:

```json
{"audio_base64": "BASE64_DEL_ARCHIVO_WAV_COMPLETO"}
```

- WAV PCM de 16 bits, estéreo, 8000 Hz.
- Canal 0: cliente; canal 1: agente. Se conserva la alineación temporal, sin recortar silencios ni remuestrear.
- Dev 2 recibe dos arrays NumPy mono float32 y sample_rate=8000.
- Límite local de archivo decodificado: 12 MiB; no es un límite publicado por Altur. No hay un corte a los 180 segundos.
- No acepta PCM crudo, prefijos data URI, audio mono ni otras frecuencias.

Respuesta mínima cuando exista el modelo:

```json
{"is_synthetic": true}
```

Respuesta opcional con score:

```json
{"is_synthetic": true, "confidence": 0.87}
```

is_synthetic es booleano obligatorio. La implementación admite confidence finito entre 0 y 1; al omitirlo o devolver None no aparece en el JSON. Altur no define de manera precisa su semántica en el README consultado: confirmarla antes de integrarlo.

| Ruta / caso | Resultado actual |
| --- | --- |
| GET /health | 200, API activa |
| GET /ready | 503 hasta que detector.ready sea True |
| GET /docs | 200, Swagger UI |
| POST /detect, entrada inválida | 422 |
| POST /detect, WAV válido sin modelo | 503, detalle: Falta integrar el modelo de Dev 2 |
| POST /detect, modelo listo | Valida y devuelve la predicción del predictor |

La cabecera X-Process-Time-Ms mide tiempo interno hasta obtener la respuesta del manejador; no incluye todo el transporte de red ni sustituye el benchmark.

## Integración con Dev 2 y Dev 3

Dev 2 debe entregar los pesos, dependencias, preprocesamiento exacto, umbral y función de inferencia. Implementar en app/model.py:

```python
class Detector:
    def __init__(self):
        # Cargar aquí los pesos y recursos una sola vez.
        # Marcar ready=True únicamente después de una carga exitosa.
        ...

    def predict(self, client, agent, sample_rate):
        # Ejecutar inferencia real.
        # Retornar un bool nativo de Python y, si procede, un float.
        ...
```

No cambiar ready a True sin implementar inferencia. Evitar un resultado constante como sustituto del modelo. Como las solicitudes pueden ejecutarse en paralelo, comprobar que el modelo soporte concurrencia o agregar sincronización. Dev 3 puede aportar características del canal agente dentro de este predictor. No depender de archivos turns ni etiquetas que no lleguen en la solicitud del benchmark.

## Pendientes en orden

1. Pedir a Altur un curl real: nombre del campo JSON, timeout, concurrencia, límites y definición de confidence. El README confirma base64 y formato del audio, pero no estos detalles.
2. Conectar el predictor real y agregar pruebas de integración con sus pesos.
3. Descargar las llamadas oficiales desde Releases y probar el conjunto val sin usar sus etiquetas para entrenar. Confirmar el formato con archivos reales.
4. Aplicar límite del cuerpo HTTP antes de cargar JSON, en proxy o middleware ASGI. Actualmente Pydantic limita el campo después de recibir el cuerpo. Alinear el límite con lo autorizado por Altur.
5. Medir latencia extremo a extremo p50/p95, memoria, errores y concurrencia con el modelo. La configuración Docker limita concurrencia a 16; ajustar según mediciones y benchmark.
6. Fijar versiones para el entorno de despliegue, construir y probar Docker con los pesos incluidos o montados. La prueba local se ejecutó con Python 3.9.6; el Dockerfile usa 3.12 y aún no está validado.
7. Desplegar en Vultr, configurar HTTPS, dominio/DNS, reinicio y monitoreo. Entregar a jueces solo después de comprobar /ready y /detect desde fuera del servidor.

## Fuentes y dataset

- Repositorio oficial: https://github.com/alturio/hackmty26
- Audios: https://github.com/alturio/hackmty26/releases/tag/v1.0
- Requisitos: https://github.com/alturio/hackmty26#evaluation

El repositorio oficial contiene manifest.csv y segmentos turns. Los WAV vienen en un ZIP aparte. Los splits separan hablantes; la evaluación oculta usa personas y voces nuevas. El paquete de esta entrega contiene únicamente nuestra API y pruebas; no contiene el dataset. Altur limita el uso de sus datos a HackMTY 2026 y prohíbe redistribuirlos.
