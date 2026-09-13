> Documento histórico del baseline. Para el estado actual con Alfa, consultar README.md y ALFA-REVISION.md.

# Entrega integrada y revisión del ZIP de Dev 2

## Qué se recibió y qué cambió

El archivo dev2.zip contenía otro ZIP, código, notebook y documentación, pero ningún model.pkl. Se preservaron los originales en el área local de trabajo. No se ejecutó el notebook ni se asumieron sus métricas como comprobadas.

La implementación de Dev 2 usa 87 features temporales de ambos canales, no un clasificador acústico de voz. Para mantener compatibilidad se extrajeron sin cambios las funciones VAD/features a ml/features.py. El ensemble mantiene sus hiperparámetros; el RandomForest usa n_jobs=1 para controlar la multiplicación de hilos al servir solicitudes.

La API se integró en una carpeta nueva; el paquete anterior de Dev 1 sigue separado para no interferir con el trabajo del compañero.

## Hallazgo principal de Dev 3

Entrenar con los turnos oficiales y predecir usando turnos extraídos por otro VAD crea una diferencia de distribución. Se comprobó con datos reales:

| Entrenamiento / validación | AUC val | Aciertos |
| --- | --- | --- |
| Turnos oficiales / turnos oficiales | 0.9968 | 69/71 |
| Turnos oficiales / VAD de WAV | 0.9428 | 65/71 |
| VAD de WAV / VAD de WAV | 0.9801 | 67/71 |

Se entrega el último modelo. La elección usa el mismo preprocesamiento en entrenamiento e inferencia. Se usaron 282 llamadas de train; val nunca se incluyó en fit. El train.py original reentrenaba al final con todos los datos: esa parte se sustituyó para preservar la evaluación. No se barrió un conjunto de hiperparámetros sobre val.

Los parámetros de VAD siguen siendo los originales de Dev 2, no parámetros calibrados contra Altur. Se resolvió la diferencia entrenando con ese mismo VAD. Esto no convierte el VAD en una transcripción exacta.

## Cambios de integración

- app/model.py carga al iniciar el modelo reconstruido y verifica versión sklearn, columnas y clases.
- Se elimina la ruta que devolvía sintético/confidence=0.5 ante cualquier excepción. Los errores ya no cuentan como inferencias válidas.
- Se aceptan audio y audio_base64, rechazando valores contradictorios.
- confidence se omite porque su interpretación sigue sin confirmación oficial; is_synthetic conserva el umbral 0.5.
- El acceso al modelo se serializa con un lock y los hilos numéricos se limitan a uno. VAD y decodificación pueden concurrir; hay margen para optimizar según carga real.
- El VAD sin habla en el canal cliente devuelve 422.
- models/model.pkl contiene el modelo nuevo y models/model.json contiene métricas y procedencia del entrenamiento.
- verify_real_http.py evalúa todo val y cuenta errores como fallos. También comprueba concurrencia y entradas malformadas.

## Validación y limitaciones

Se verificaron 13 pruebas automatizadas, 71 audios de val por HTTP, 8 solicitudes adicionales concurrentes y 4 entradas inválidas. Ver verification-real-http.json.

Durante el entrenamiento aparecieron avisos numéricos de NumPy/BLAS en macOS. Las probabilidades y métricas resultaron finitas, se reprodujeron las métricas del baseline, y una prueba independiente comparó la regresión logística con una suma explícita. Las 13 pruebas finales pasaron; quedó un aviso de joblib al consultar núcleos físicos del entorno. No se ha demostrado reproducibilidad numérica en Linux: verificar al desplegar.

Las features heredadas tienen limitaciones conocidas que no se cambiaron sin reentrenar: latency_frac_negative siempre vale cero porque su fórmula solo considera turnos del agente que ya terminaron; el emparejamiento de respuestas puede asociar varias intervenciones del cliente al mismo turno del agente; solapamiento no demuestra por sí solo una interrupción. Evitar presentar esas features como medidas exactas de conducta o como pruebas universales de IA.

El desempeño medido usa conversaciones completas. No se ha validado detección temprana sobre fragmentos cortos, generalización a otros bancos, voces nuevas del conjunto oculto ni resistencia a imitaciones deliberadas de timing.

## Cómo seguimos con Dev 3

1. Mantener esta versión como referencia reproducible.
2. Revisar segmentación y casos difíciles usando solo train para ajustes; mantener una evaluación separada al experimentar.
3. Implementar emparejamiento explícito de turnos y latencias con signo en una versión nueva de features, reentrenando al cambiar el esquema.
4. La trampa semántica sigue pendiente. Requiere transcripciones o un servicio de audio y evaluación de respuestas a las preguntas del agente. No hay Gemini, transcripción ni API keys integradas; las señales semánticas no alteran el score actual.
5. Medir el aporte de cada cambio con un protocolo que evite ajustar repetidamente sobre val.

## Para el compañero de Dev 1

Iniciar con README.md. Prioridades: confirmar contrato de entrada/timeout/concurrencia con Altur, aplicar límites HTTP antes de parsear, probar Docker con pesos, verificar Linux, desplegar en Vultr y probar desde una red externa. No prometer 50 ms de nube a partir de la medida local. El ZIP incluye pesos y código, sin audios ni credenciales.

Fuentes: repositorio oficial https://github.com/alturio/hackmty26 ; código de la entrega dev2.zip del equipo. Las métricas de esta entrega provienen de ejecuciones locales, no solo del README recibido.
