# Revisión final contra el reto oficial de Altur

Basado en `hackmty26-altur-challenge.pdf` (Tecnologías Altur S.A.P.I. de C.V., actualizado agosto 2026) y el README vigente de https://github.com/alturio/hackmty26 (verificado sin cambios el 2026-09-12). Este documento evalúa el proyecto contra los **criterios de jueces publicados**, no contra una lista inventada por nosotros.

## 1. Cumplimiento del contrato técnico — verificado con el script real del juez

No es una suposición: se corrió `scripts/check_endpoint.py` (bajado directo de `alturio/hackmty26`, el mismo cliente HTTP que usa el juez) contra nuestro servidor real, con las 71 llamadas de val del dataset oficial.

| Requisito del PDF/README oficial | Estado |
| --- | --- |
| `POST /detect`, WAV estéreo 8kHz base64, canal 0=cliente/canal 1=agente | Cumple — `app/audio.py` lo valida estrictamente |
| Respuesta `{"is_synthetic": bool, "confidence": 0.0-1.0}` | Cumple — `confidence` activado y con la semántica correcta (ver `CAMBIOS-Y-VALIDACION.md` hallazgo #7) |
| 30 s máximo por llamada | Cumple con margen amplio: latencia media 100-150 ms, máxima observada 228 ms |
| Llamadas de 1-4 min, cuerpo JSON hasta ~5 MB | Cumple — límite propio es más permisivo (12 MiB de WAV, ~16 MiB de cuerpo) |
| Balanced accuracy como métrica principal | Medido honestamente en val: **0.945** (no es el conjunto oculto, ver limitaciones abajo) |
| AUC / calibración si se manda `confidence` | Medido: **AUC 0.980, Brier 0.046** (coincide con las métricas internas ya documentadas) |
| Un status ≠ 200 cuenta como respuesta incorrecta | **Riesgo confirmado sin resolver**: ver sección 4 |

Evidencia completa en `work/altur_official/check_endpoint_val_full_fixed.json` y `CAMBIOS-Y-VALIDACION.md`.

## 2. Contra los criterios de jueces del PDF

### Robustness (desempeño con hablantes, motores y condiciones no vistos)

Lo único medible hoy es val (71 llamadas, hablantes fuera de train, pero del mismo dataset/distribución). El desempeño contra el conjunto oculto real (voces y personas nuevas, posiblemente motores TTS distintos) es desconocido — nadie puede afirmar eso sin correrlo. Lo honesto: `balanced_accuracy 0.945` en val es la única evidencia que tenemos, y se debe presentar así ante el juez, no como garantía.

### Originality (uso de señales más allá de un clasificador acústico convencional)

Punto fuerte, pero **no está dicho explícitamente en ningún documento hasta ahora** (ya se agregó al README en esta revisión). El modelo desplegado usa 87 características derivadas **enteramente de la dinámica de turnos** (duración de intervenciones, latencias con signo, solapamientos, silencios, entropía de esos patrones) — es el segundo de los tres enfoques que sugiere el material del reto (comportamiento conversacional), no un clasificador espectral/MFCC/prosódico convencional. Esto hay que decirlo en voz alta durante los 15 minutos con el juez.

### Technical depth (ejecución y comprensión de por qué funciona la solución)

Punto fuerte con evidencia concreta:
- Se probaron 26 features adicionales y se **rechazaron con un umbral fijado antes de medir** (mejora de AUC 0.0015 < 0.005), documentado en `INTEGRACION.md`.
- Se encontró y corrigió un defecto real en `latency_frac_negative` (estructuralmente siempre 0), se implementó la corrección de forma reversible (`latency_pairing`), se probó contra el dataset oficial completo, y **tampoco se promovió** por no superar el mismo umbral — misma disciplina, dos veces.
- Se encontró y corrigió una inversión real de semántica en `confidence` (AUC 0.442 → 0.980) usando el script oficial del juez antes de darlo por bueno, no una prueba propia.
- Separación estricta train/val respetada en todo momento; nunca se ajustó nada contra val de forma repetida.

### Feasibility (viabilidad de despliegue real en un banco)

Punto fuerte: CPU únicamente (sin GPU), ensemble de scikit-learn liviano, ~100-200 ms por llamada, sin dependencias pesadas. El WAV de entrada es exactamente el formato telefónico descrito (8kHz, estéreo). Limitación honesta: no se ha probado con ruido de línea telefónica real, códecs de compresión (G.711/G.729), ni llamadas con más de dos canales o metadatos distintos a los del dataset.

### Latency (velocidad de decisión con confianza razonable)

Punto fuerte y ya medido con el script oficial: media 100-150 ms, máximo 228 ms sobre 71 llamadas reales, muy por debajo del límite de 30 s. Vale la pena decir este número explícitamente en la demo.

## 3. Lo que el PDF aclara que NO habíamos visto

- **Formato de evaluación real:** el juez visita la mesa del equipo 15 minutos, corre su benchmark en vivo contra el endpoint, y el equipo explica su solución en ese mismo tiempo. **El endpoint debe estar accesible durante todo ese lapso** — esto es un requisito operativo, no solo técnico, y no estaba resuelto en ninguna parte del proyecto hasta ahora.
- **El material del reto valora profundidad sobre cantidad de señales**: confirma que la decisión de no integrar las 26 features de Dev 3 fue la correcta bajo los propios criterios del reto, no una limitación nuestra.
- **El canal del agente es señal, no ruido**: el material del reto aclara que el audio del agente sirve para entender a qué estaba reaccionando quien llama, no es solo contexto. Ya lo usamos (turnos del agente entran a las 87 features y a los 26 de Dev 3), vale la pena decirlo explícitamente como decisión de diseño, no como dato incidental.
- **Semántico es una de las tres direcciones sugeridas explícitamente** (no una idea nuestra aislada) — el trabajo de Gemini en `dev3/semantics.py` está más alineado con el reto de lo que el proyecto comunicaba antes.

## 4. Riesgo operativo sin resolver — accesibilidad del endpoint durante el juzgamiento

Ahora mismo el servidor solo se ha probado en `127.0.0.1` (localhost). El PDF exige que el juez pueda llamarlo desde su propia laptop durante la visita de 15 minutos. Hay dos caminos, ninguno configurado todavía:

**Opción A — Red local del venue (más simple, depende de la red del lugar):**
1. Correr el servidor con `--host 0.0.0.0` en vez de `127.0.0.1` para que escuche en todas las interfaces:
   ```powershell
   python -m uvicorn app.main:app --host 0.0.0.0 --port 8025
   ```
2. Encontrar la IP local de la laptop en la red del venue: `ipconfig` (buscar "Dirección IPv4").
3. Verificar que el firewall de Windows no bloquee el puerto (puede pedir permitir la app la primera vez).
4. Dar al juez `http://<tu-ip-local>:8025/detect` — **funciona solo si la laptop del juez está en la misma red/WiFi** que la del equipo. Riesgo: redes de eventos a veces aíslan dispositivos entre sí (client isolation) por seguridad, y esto no funcionaría.

**Opción B — Túnel público (más confiable, funciona sin importar la red):**
Usar un servicio de túnel (ngrok, Cloudflare Tunnel) que expone `localhost:8025` con una URL pública `https://algo.ngrok-free.app` accesible desde cualquier red. Ninguno está instalado en esta máquina — recomendación:
```powershell
# instalar ngrok una vez (https://ngrok.com/download), luego:
ngrok http 8025
```
Esto da una URL pública temporal que se le entrega al juez. Más confiable que depender de la red del venue, pero requiere internet real (no solo LAN) y una cuenta gratuita de ngrok.

**Recomendación:** preparar y ensayar la Opción B con anticipación (no el día del evento) — es la que menos depende de factores fuera de tu control. Dejar la Opción A como respaldo si el venue confirma que no hay aislamiento de red.

**Esto no lo puedo resolver yo por ti**: requiere decidir qué laptop corre el servidor el día del evento, y probarlo en las condiciones reales de red antes de la evaluación. Es la única brecha genuinamente operativa que queda abierta en todo el proyecto.

## 5. Riesgo técnico sin resolver (recordatorio)

`/detect` devuelve `422` si el VAD no detecta habla en el canal del cliente. Según el PDF, cualquier status ≠ 200 cuenta como respuesta incorrecta. No ocurrió en las 71 llamadas de val, pero el conjunto oculto tiene voces nuevas — ver hallazgo #8 en `REVISION-TECNICA.md`. Sigue siendo una decisión de producto pendiente del equipo (fallar limpio vs. arriesgar un valor por defecto), no resuelta unilateralmente.

## 6. Checklist para correr justo antes de que llegue el juez

```powershell
cd "ruta\a\Altur_project"
.\.venv\Scripts\Activate.ps1
python -m pytest -q                                    # confirmar 88/88 verde
python -m uvicorn app.main:app --host 0.0.0.0 --port 8025   # o el túnel de la Opción B
```
En otra terminal, ya con el túnel/IP decidido:
```powershell
python work/altur_official/check_endpoint.py --url http://<tu-url-o-ip>:8025/detect --split val --n 10
```
Si esto corre limpio (0 errores) contra la URL que le vas a dar al juez — no contra `127.0.0.1` — significa que la ruta de red real funciona, no solo el código.

## 7. Qué decir en los 15 minutos (guion corto, complementa `GUION-DEMO.md`)

1. "Nuestro modelo no analiza cómo suena la voz — analiza cómo se comporta la conversación: turnos, silencios, latencias de respuesta con signo, solapamientos." (originalidad)
2. "Probamos agregar más señales dos veces y las rechazamos ambas veces con un umbral fijado antes de medir, porque no mejoraban lo suficiente." (profundidad técnica)
3. "100-200 ms por llamada, medido con su propio script de verificación, no el nuestro." (latencia)
4. "Corre en CPU, sin modelo pesado — así lo correría un banco de verdad." (viabilidad)
5. "Esto es val, no su conjunto oculto — no sabemos cómo nos va a ir ahí, y no vamos a prometer un número que no hemos medido." (honestidad, refuerza credibilidad)
