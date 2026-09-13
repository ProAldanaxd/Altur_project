# Dev 3 — entrega y criterios de calidad

## Qué funciona

- dev3/temporal.py normaliza intervalos, une duplicados/solapamientos del mismo canal y calcula tiempos sin doble conteo.
- Latencias cliente-agente con signo: un inicio cliente anterior al fin del agente produce valor negativo.
- Detecta inicios de habla durante actividad del otro canal, simultaneidad, pausas y reinicios del cliente después de un turno con solapamiento.
- Devuelve línea temporal, eventos con segundos, estadísticas y 26 features versionadas temporal-v1.
- Integración API con /conversation/analyze y /conversation/status.
- dev3/semantics.py conecta con Gemini para transcribir canales por separado o revisar transcripciones existentes, con salidas estructuradas y citas comprobadas.
- /conversation/semantic recibe escenarios ficticios explícitos para revisar rechazo de premisa, aclaración, aceptación o elaboración. Sin escenario la premisa queda unverified: no se declara inexistente una entidad solo por intuición del modelo.

## Definiciones exactas

Intervalos semiabiertos [start,end). Dos segmentos que solo se tocan no se solapan. Se fusionan segmentos adyacentes del mismo canal para evitar duplicación. Las latencias emparejan el primer inicio del cliente estrictamente posterior al inicio del agente y anterior al siguiente inicio de agente. No reutilizan un cliente para varios agentes. Inicios simultáneos se reportan aparte. Un emparejamiento temporal no demuestra que una frase responda semánticamente a otra.

El tiempo activo es la unión de ambos canales; overlap es su intersección y dead_air es duración menos unión. Se comprueba client_speech + agent_speech - overlap = active_speech. Las estadísticas ausentes quedan null en el informe y solo se convierten a cero en el vector numérico, con indicadores explícitos de ausencia.

Un inicio durante el habla ajena es una interrupción potencial, no evidencia de intención. Un reinicio después de solapamiento es observable, no necesariamente recuperación. La voz humana/sintética no se decide con reglas de tipo pausa mayor a X. Las features no agregan un score inventado.

## Preprocesamiento

Conserva el mismo VAD energético de Dev 2 para compatibilidad: frames de 20 ms, mínimo de habla 200 ms y unión de huecos menores de 200 ms. El análisis temporal es correcto respecto a esos intervalos; ruido, eco, música y habla tenue pueden alterar la segmentación. No se ha medido exactitud contra anotaciones humanas independientes. Los turnos entregados por Altur también son automáticos.

## Evidencia

- 71 pruebas automatizadas aprobadas. Incluyen unión contra un oráculo discreto independiente, latencias negativas, emparejamiento, silencios, entradas inválidas, regresión de /detect y proveedor simulado.
- 71 WAV oficiales de val por /conversation/analyze: cero errores y todas las invariantes comprobadas. Se observaron 159 pares con latencia negativa según nuestro VAD.
- Ocho solicitudes conversacionales adicionales con cuatro hilos: todas 200.
- Las mismas 71 llamadas por /detect mantienen 67 aciertos, cero errores HTTP y los mismos pesos.
- Las dos suites HTTP se ejecutaron con actividad concurrente en la máquina; sus tiempos son mediciones locales de esa ejecución, no una comparación aislada de regresión de rendimiento.
- Los mocks Gemini prueban formato, errores, timeouts, asignación aislada de canales y rechazo de citas inventadas. No miden calidad semántica ni demuestran conectividad real con Google.

## Evaluación del aporte al clasificador

Se compararon las 87 features de Dev 2 frente a 113 (87+26) con el mismo ensemble, mismos hiperparámetros y cinco folds estratificados de train, semilla42. Cada transformación aprendida se ajustó solo dentro del fold de entrenamiento. Las 71 llamadas val no se usaron para esta comparación.

AUC medio por fold: 0.987642 con Dev2, 0.989138 con Dev2+Dev3; mejora 0.001496. Accuracy out-of-fold: 0.93617 a 0.94681. La mejora es pequeña y no consistente en todos los folds. No alcanzó el criterio fijado antes de la ejecución: aumento AUC >=0.005 sin reducción de accuracy. Por ello no se cambiaron pesos ni confidence de producción.

La CV es a nivel llamada; faltan IDs de hablante para separar personas dentro de train. No equivale al conjunto oculto ni prueba calibración de confidence. Para una mejora futura usar agrupación por hablante cuando esté disponible y evitar ajustar repetidamente sobre val.

## Estado de Gemini

Falta una API key y un modelo habilitado en la cuenta. Sin ello el estado real es unavailable/missing_api_key. No se han enviado audios ni transcripciones a Google en esta entrega. El adaptador requiere verificación real antes de anunciar una integración de sponsor funcionando.

Las transcripciones suministradas, audios, respuestas del proveedor y descripciones de escenarios se tratan como datos no confiables, nunca instrucciones. Se comprueban índices, canales, secuencia temporal y que las citas existan textualmente. Esto no garantiza que la interpretación de Gemini sea correcta; requiere revisión y un conjunto de ejemplos semánticos etiquetados por el equipo.

No hay sistema offline que afirme detectar alucinaciones sin transcribir. Las referencias a premisas ficticias necesitan el guion de pruebas del equipo. Aceptar una premisa no prueba que alguien sea una IA.

## Para continuar

1. Configurar key/modelo Gemini y correr el ejemplo ficticio incluido; revisar respuesta y costos en la cuenta.
2. Validar transcripciones y reacciones con ejemplos etiquetados, incluyendo humanos que se confunden y agentes que preguntan por servicios reales.
3. Medir calidad y latencia de semántica por separado. El benchmark /detect no depende del servicio externo.
4. Si se desea fusionar scores, entrenar y calibrar con datos separados y comparar antes de activar; hoy affects_detect=false.
5. Dev 1 puede continuar Docker/Vultr/HTTPS con el proyecto completo. No hace falta modificar sus pesos para usar el análisis explicativo.

Fuentes de implementación Gemini: https://ai.google.dev/api/generate-content ; https://ai.google.dev/gemini-api/docs/generate-content/audio ; https://ai.google.dev/gemini-api/docs/generate-content/structured-output . Dataset y formato de Altur: https://github.com/alturio/hackmty26 .
