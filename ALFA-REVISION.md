# Revisión del artefacto dev2Alfa.pkl

## Resultado de integración

Alfa se carga correctamente en FastAPI usando las mismas 87 features y scikit-learn 1.6.1. El ensemble contiene LogisticRegression, RandomForest y HistGradientBoosting. No se modificaron los bytes del archivo recibido.

SHA256 original: `4a187789ecf0b93a16416810d8b8fc86f1351cdd7861114380b2fbb82dae09c1`.

Se conservan dos variantes: baseline predeterminada y alfa seleccionable mediante MODEL_VARIANT. Ambas reciben WAV real, separan canales y extraen turnos con el VAD de Dev 2. La API no entrega turnos oficiales al modelo durante inferencia.

## Evidencia de entrenamiento

El scaler de Alfa registra 353 muestras. Su vector de medias coincide, con tolerancia 1e-10, con las features de todos los turnos oficiales. No coincide con las features extraídas del audio. El archivo no contiene metadata de procedencia, pero esta coincidencia es evidencia fuerte de preparación con el conjunto completo; no consideramos val independiente para Alfa.

No se puede reconstruir toda la historia de entrenamiento solo a partir de los pesos. Pedir a Dev 2 el commit de entrenamiento, el split utilizado y su configuración de VAD para documentar la procedencia definitiva.

## Resultados con los mismos 71 WAV por HTTP local

| Variante | Aciertos | Humanos marcados sintéticos | Sintéticos marcados humanos | Media | p95 |
| --- | --- | --- | --- | --- | --- |
| baseline | 67/71 | 3 | 1 | 47.1 ms | 60.0 ms |
| alfa | 64/71 | 6 | 1 | 47.0 ms | 59.2 ms |

Ambas tuvieron cero errores HTTP. Cada servidor recibió también 8 solicitudes con 4 hilos concurrentes, todas con estado 200, y 4 solicitudes inválidas, todas con 422. Pytest: 19 aprobadas; quedó un aviso de joblib sobre consulta de núcleos físicos. Las pruebas incluyen ambos modelos reales, hashes alterados, variante desconocida y exceso de cuerpo con/sin Content-Length antes de parsear JSON.

El AUC interno de Alfa sobre WAV es 0.9515; el del baseline con holdout conservado es 0.9801. El 1.0 de Alfa usando turnos oficiales no es evidencia de generalización porque su preparación parece incluir esas muestras. No ajustar el umbral para mejorar artificialmente esta comparación.

## Decisión

Baseline sigue activa por defecto porque mantiene validación separada, alinea preprocesamiento de entrenamiento con producción y obtuvo mejor resultado observado con WAV. Alfa queda integrada, ejecutable y preservada para que Dev 2 pueda continuar. Esto no afirma que baseline ganará el conjunto oculto.

No se añadió todavía Gemini ni análisis semántico de Dev 3. El siguiente trabajo puede arrancar desde esta integración sin cambiar silenciosamente los pesos o las features. Mantener la comparación reproducible y evaluar mejoras con un protocolo independiente.

## Archivos para el compañero

- README.md: ejecución, contrato, variantes, pruebas y despliegue pendiente.
- app/model.py: carga, verificación de identidad y adaptación de ambos modelos.
- app/limits.py: límite del cuerpo HTTP antes de JSON.
- models/registry.json: variante, hash, versión y procedencia.
- models/dev2Alfa.pkl: archivo original recibido.
- models/model.pkl: baseline anterior conservado.
- audit_alfa.py y audit-alfa.json: auditoría de features y procedencia.
- verify_real_http.py y verification-*-http.json: comprobaciones HTTP.
- INTEGRACION.md: historial de construcción del baseline.

Docker/Vultr/HTTPS siguen pendientes. Las mediciones corresponden a macOS ARM64 con Python 3.9.6 en localhost. No se incluyen WAV, credenciales ni datos originales en el ZIP.
