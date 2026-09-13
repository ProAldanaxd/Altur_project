# Altur — integración Dev 1 + Dev 2, con Alfa

Backend FastAPI en Python con dos modelos auditados. Baseline queda activo por defecto; Alfa está integrado y se selecciona con MODEL_VARIANT=alfa. No hay despliegue público todavía. Dev 3 semántico no está integrado.

## Arrancar

Desde esta carpeta:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pytest -q
python -m uvicorn app.main:app --host 127.0.0.1 --port 8020
```

Abrir http://127.0.0.1:8020/docs . Modelo predeterminado: baseline, entrenado con WAV de train y 87 features temporales. Los pesos están incluidos.

Para correr Alfa en otra terminal con el entorno activado:

```bash
MODEL_VARIANT=alfa python -m uvicorn app.main:app --host 127.0.0.1 --port 8019
```

Abrir http://127.0.0.1:8019/docs . No cambiar la variable mientras el servidor está abierto: reiniciar para cambiar modelo. Cambiar el nombre de una variante por otro no reentrena ni altera los pesos.

## API

- GET /health: proceso vivo.
- GET /ready: 200 solo después de cargar, verificar identidad, versión, esquema de features, clases y prueba numérica. Incluye variante y SHA256.
- GET /model: variante, huella de pesos, origen de entrenamiento y si se conservó holdout.
- POST /detect: acepta audio o audio_base64; si se proporcionan ambos deben coincidir.
- GET /docs: documentación interactiva.

Entrada:

```json
{"audio": "BASE64_DEL_WAV_COMPLETO"}
```

Respuesta:

```json
{"is_synthetic": true}
```

WAV estéreo PCM16 de 8 kHz: canal 0 cliente y canal 1 agente. No remuestrea ni recorta silencios. Confidence se omite hasta aclarar su semántica con Altur; el umbral interno es 0.5. Confirmar con Altur el campo de entrada definitivo, timeout y concurrencia: no se afirman esos puntos como contrato oficial.

Errores: 422 para entrada/formato inválido o sin habla detectada del cliente; 413 si el cuerpo excede el límite; 503 si falta el modelo. Los fallos inesperados no se convierten en predicciones ficticias.

Límites locales: 12 MiB de WAV decodificado; cuerpo HTTP de hasta MAX_BASE64_CHARS + 65536 bytes (aproximadamente 16 MiB). Se aplica antes de parsear JSON y también cuando no existe Content-Length. Si se duplican ambos campos grandes, el límite total del cuerpo sigue aplicando. Un proxy debe aplicar límites y timeouts acordes al benchmark. X-Model-Variant identifica el modelo; X-Process-Time-Ms mide tiempo interno, no latencia total de red.

## Comparación y elección

| Modelo, vía WAV y HTTP | Aciertos en las 71 llamadas | Errores HTTP |
| --- | --- | --- |
| Baseline, entrenado en 282 WAV de train | 67/71 (94.37%) | 0 |
| Alfa recibido | 64/71 (90.14%) | 0 |

Alfa tiene un scaler ajustado con 353 muestras. Su vector de medias coincide con el de todos los turnos oficiales y no con el VAD de WAV. Es evidencia fuerte de preparación con train+val; por ello su resultado sobre val es diagnóstico, no una evaluación independiente. En turnos oficiales marca 71/71, pero ese número no demuestra generalización.

Baseline conserva val fuera del entrenamiento y usa el mismo VAD que la API. Permanece predeterminado: Alfa está integrado para comparación y futuros ajustes, pero no representa una mejora comprobada. Esta comparación no sustituye al conjunto oculto. Ver ALFA-REVISION.md y los JSON de evidencia.

## Pruebas reproducibles

```bash
python -m pytest -q
python verify_real_http.py --data-root /ruta/al/dataset --url http://127.0.0.1:8020 --report verification-baseline-http.json
python verify_real_http.py --data-root /ruta/al/dataset --url http://127.0.0.1:8019 --report verification-alfa-http.json
python audit_alfa.py --data-root /ruta/al/dataset
```

La carpeta de datos requiere manifest.csv, turns/ y audio/. Dataset oficial: https://github.com/alturio/hackmty26 ; audios: https://github.com/alturio/hackmty26/releases/tag/v1.0 . Los WAV no se incluyen ni se redistribuyen.

La auditoría compara features y normalizador del artefacto. Las pruebas HTTP usan todas las llamadas val, verifican 8 solicitudes adicionales con cuatro hilos concurrentes y cuatro entradas inválidas. La última ejecución de pytest aprobó 19 pruebas. Ver informes para tiempos y errores, medidos localmente en macOS ARM64/Python 3.9.6.

## Pesos y entrenamiento

models/registry.json contiene las identidades verificadas de baseline y alfa. El archivo recibido dev2Alfa.pkl se conserva byte por byte; solo se cambia n_jobs a 1 en memoria. Cargar únicamente artefactos de confianza: pickle puede ejecutar código. Se exige scikit-learn 1.6.1 y se rechazan versiones de entrenamiento incompatibles.

MODEL_PATH permite apuntar a otra copia del mismo artefacto, pero su SHA256 debe coincidir con la variante elegida. No editar el registro para aceptar pesos nuevos sin auditarlos.

train_validate.py reconstruye el baseline y guarda métricas. Al generar pesos nuevos, auditar y actualizar la entrada baseline en registry.json antes de activarlos. No usar los resultados antiguos como validación del archivo nuevo. Reiniciar el servidor tras cambiar pesos.

## Docker y pendientes

```bash
docker build -t altur-dev1-dev2 .
docker run --name altur-api --restart unless-stopped -p 8000:8000 -e MODEL_VARIANT=baseline -d altur-dev1-dev2
```

Cambiar a -e MODEL_VARIANT=alfa para esa variante. Docker usa puerto 8000 y Python 3.12; la imagen sigue sin probar. requirements-lock.txt registra el entorno local; las dependencias ML están fijadas también en requirements.txt. Falta probar Linux/Docker, configurar HTTPS/Vultr/dominio y verificar desde una red externa. Los tiempos locales no garantizan latencia en nube.

La parte semántica de Dev 3 queda para el siguiente paso. Las features temporales actuales mantienen sus limitaciones conocidas; consultar INTEGRACION.md, que describe el trabajo previo del baseline.
