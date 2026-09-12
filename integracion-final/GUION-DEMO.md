# Pitch de 3 minutos y demo

Presentación editable: `Pitch-Altur.pptx`. Las notas incluyen este recorrido. Ensayar con cronómetro; ajustar pausas a la velocidad del presentador.

## 0:00–0:25 · Problema

En una llamada bancaria, una voz sintética puede conversar con el agente. Nuestro sistema recibe la llamada y estima si la voz del cliente es sintética. Queremos ayudar al agente a decidir cuándo pedir una verificación adicional, conservando evidencia de cada decisión. El resultado es una señal de apoyo: no sustituye la verificación de identidad.

## 0:25–1:00 · Cómo funciona

Recibimos un WAV estéreo de ocho kilohertz: cliente en canal cero y agente en canal uno. Detectamos actividad de voz y extraemos patrones temporales de la interacción. El clasificador combina tres modelos y usa 87 variables. Además, mostramos pausas, solapamientos y tiempos de respuesta. Esa explicación adicional todavía no modifica el veredicto: su mejora en validación cruzada fue pequeña y conservamos el modelo probado.

## 1:00–1:45 · Resultados

Probamos el endpoint con las 71 llamadas del conjunto público de validación. Acertó 67: un 94.4 por ciento. Clasificó correctamente 34 de 37 humanos y 33 de 34 sintéticos. Hubo tres falsas alertas sobre humanos y un sintético no detectado. Medimos solicitudes completas por HTTP; los resultados corresponden a esta Mac y a la validación pública. El conjunto oculto de los jueces y el rendimiento en nube todavía están pendientes.

## 1:45–2:30 · Demo

Seleccionamos un WAV y mostramos el veredicto. La solicitud recibe un identificador. En el registro aparecen ese identificador, la probabilidad interna, la latencia y el modelo utilizado. No guardamos el audio ni la transcripción. La escritura ocurre en segundo plano y sus fallos son visibles. La alerta propone pedir una verificación adicional de identidad. La integración de ElevenLabs está preparada; hasta probarla con credenciales reales, presentamos el texto y su estado pendiente.

## 2:30–3:00 · Cierre

Hoy funcionan juntos la clasificación local, el análisis temporal y la auditoría persistente. El modelo conserva una validación separada. Sigue conectar las cuentas de Gemini, ElevenLabs y PostgreSQL, medir sus resultados y desplegar con HTTPS para probar desde fuera de la red local. La propuesta es una señal que el agente puede revisar, acompañada de un historial de cómo se tomó la decisión.

## Recorrido técnico dentro de los 15 minutos

| Minutos | Responsable | Acción |
| --- | --- | --- |
| 0–3 | Juan Carlos | Pitch con las 5 diapositivas |
| 3–6 | Dev 1 | `/ready`, WAV humano y WAV sintético por `/demo`; enseñar respuesta e identificador |
| 6–8 | Dev 2 | Explicar train/val, matriz de errores y decisión de mantener baseline |
| 8–10 | Dev 3 | Mostrar pausas, solapamientos y latencias en `/conversation/analyze` |
| 10–12 | Juan Carlos | Registro persistido; reproducir alerta solo si la conexión real fue verificada |
| 12–15 | Equipo | Preguntas y limitaciones |

## Preparación y contingencias

Antes del ensayo, iniciar el servidor y comprobar `/ready`. Elegir dos WAV autorizados del dataset y anotar sus etiquetas reales. La selección de ejemplos ilustra funcionamiento; no reemplaza la evaluación completa. Tener abiertos el panel, `/docs` y los reportes de validación.

Si falta ElevenLabs, mostrar el texto y el estado pendiente. Si Gemini no está disponible, mostrar el análisis temporal local. Si falla PostgreSQL, mostrar SQLite y aclarar que la exportación remota no está verificada. Si el servidor no responde, reiniciarlo con el comando del README; no presentar un reporte guardado como si fuera ejecución en vivo.

Cuando existan pruebas reales de los proveedores o del despliegue, actualizar las diapositivas y este guion para describir el estado verificado.
