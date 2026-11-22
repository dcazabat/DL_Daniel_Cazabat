# Videos y recursos de la Clase 3

Metadatos obtenidos el **16/11/2026** desde YouTube (API oEmbed + `yt-dlp`). Los títulos, canales,
duraciones y fechas son los reales del video, no los del apunte. Las transcripciones completas
(automáticas, sin revisar) están en `transcripciones/`.

> Los contadores de visitas son de la fecha de consulta y cambian. Las duraciones están redondeadas
> al segundo.

---

## 1. Videos obligatorios (según `tarea.md`)

| # | Video | Canal | Duración | Publicado | URL |
|---|---|---|---:|---|---|
| 1 | MNIST Dataset: Ultimate Guide to Handwritten Digit Dataset and its applications | Coding With Aman Dhillon | 2 min 36 s | 2023-10-16 | <https://youtube.com/watch?v=sv5-z1fIIYE> |
| 2 | Descubre Ultralytics YOLO, modelos, soluciones y más para Visión por Computador | OMES | 4 min 36 s | 2025-04-02 | <https://youtu.be/pKm3oMcWpuk> |
| 3 | ¿Qué es un LLM? Cómo funcionan los Modelos de Lenguaje de IA | CKA Coaching and Consulting | 4 min 42 s | 2026-08-24 | <https://youtu.be/JV-KVcFWkmk> |

### 1.1 MNIST Dataset: Ultimate Guide (2:36, en inglés)

Video **introductorio al dataset**, no un tutorial de entrenamiento. Recorre, en menos de tres
minutos: qué es MNIST, por qué se lo usa como "playground" de ML y visión por computador, cuáles son
los subconjuntos (train/test), la diversidad de escrituras, que las imágenes son escala de grises y
cómo se accede al dataset. Cierra con "el entrenador silencioso detrás de escena" (la comunidad que
lo usa como benchmark).

- **Para qué sirve en la clase:** da contexto histórico y de formato (28×28, 0-255, 70.000 imágenes).
- **Para qué no alcanza:** no muestra Keras ni el entrenamiento. Eso está en el notebook.
- **Transcripción:** no disponible (YouTube devolvió HTTP 429); ver `transcripciones/00-nota-mnist-dataset.md`.

### 1.2 Ultralytics YOLO (4:36, español)

Recorrido por la suite de Ultralytics antes de programar:

| Capítulo | Min | Idea |
|---|---|---|
| Introducción | 0:00 | Qué es Ultralytics y por qué YOLO |
| ¿Qué es Ultralytics? | 0:26 | Organización de I+D; YOLO = *You Only Look Once*, equilibrio velocidad/precisión para tiempo real |
| Tasks | 1:28 | detección, segmentación, clasificación, pose, OBB |
| Modes | 1:53 | train/fine-tune, val, predict, export, track (todo el ciclo de vida) |
| Models | 2:18 | de YOLOv3 a v12 + SAM |
| Datasets | 2:34 | datasets oficiales por tarea |
| Soluciones | 2:49 | conteo de objetos, recorte, desenfoque, monitoreo de postura, estimación de velocidad |
| Instalación | 3:35 | `pip install ultralytics`, Python ≥ 3.8 |
| Licenciamientos | 3:56 | AGPL-3.0 (estudiantes/open source) vs Enterprise (comercial) |

- **Ojo:** el video menciona "YOLO v11" como última y "v12 en desarrollo". Hoy la línea vigente es
  **YOLO26** (Ultralytics la recomienda para proyectos nuevos) y la librería va por la 8.4.x. El
  concepto (tasks/modes/models) sigue siendo exactamente el mismo.
- Complementa perfecto con el notebook de YOLOv8: el video explica **qué** hay; el notebook, **cómo**.

### 1.3 ¿Qué es un LLM? (4:42, español)

Explicación conceptual, sin código:

- Un LLM es un modelo entrenado con enormes cantidades de texto que **aprende a predecir el
  siguiente token**; de ahí salen responder, redactar, resumir, traducir y programar.
- Se llama "grande" por tres cosas: **tamaño del dataset**, **cantidad de parámetros** y **capacidad
  de cómputo** (GPU/TPU).
- Analogía útil para clase: *"un analista junior extremadamente rápido que leyó millones de
  documentos, pero necesita supervisión, reglas y controles"*.
- Usos: lenguaje natural, código y ciberseguridad (alertas, reportes SOC, *threat hunting*).
- Modelos: GPT-4 (OpenAI), Claude (Anthropic), Llama (Meta).
- Herramientas locales: **Ollama** (CLI, API local compatible con la de OpenAI) y **LM Studio**
  (GUI, *local-first*, más orientado a principiantes).

> La transcripción automática confunde nombres ("LM Cloud" por Claude, "Antro-" por Anthropic,
> "IPUs" por GPUs). Es un buen ejemplo para mostrar en clase que un LLM no es una fuente de verdad.

---

## 2. Videos de apoyo (los `.url` de la carpeta)

| # | Video | Canal | Duración | Publicado | URL |
|---|---|---|---:|---|---|
| 4 | El PERCEPTRÓN ¡explicado! (demo interactivo) | Codificando Bits | 23 min 12 s | 2025-07-17 | <https://www.youtube.com/watch?v=pCy4yPzcCqs> |
| 5 | Conceptos básicos del DEEP LEARNING | Codificando Bits | 10 min 54 s | 2018-06-25 | <https://www.youtube.com/watch?v=Hz5LfWah6VE> |
| 6 | Machine Learning Explicado: Cómo Aprenden Realmente Las Máquinas | Objetivo IA | 10 min 24 s | 2025-11-19 | <https://www.youtube.com/watch?v=6c3BxGzE-Fo> |
| 7 | Aprende a Crear Prompts: Metodología de 4 Pasos | Luis J Carlos | 8 min 36 s | 2026-06-29 | <https://www.youtube.com/watch?v=fXfR_VwoVXQ> |

### 2.1 El Perceptrón (23:12) — el más importante para entender el Módulo 2

Demo interactiva, sin fórmulas abstractas. La lógica que construye:

1. **Problema:** clasificar puntos de 2 características (`x1`, `x2`) en dos categorías.
2. **Modelo:** `z = w1·x1 + w2·x2 + b`. Si `z > 0` → categoría A; si `z < 0` → categoría B.
3. **Transformación:** el dato pasa de 2 dimensiones a **un solo número** `z`.
4. **Clasificación:** comparar `z` con el umbral 0.
5. **Parámetros:** `w1` y `w2` cambian la **orientación** de la frontera; `b` la **desplaza**.
   Con sliders se ve que hay **infinitas** combinaciones que separan bien (no hay solución única).
6. **Lo mágico:** en la práctica esos parámetros **no se eligen a mano**: los calcula el
   entrenamiento. Ese es el puente entre "perceptrón de 1958" y "red neuronal de hoy".
7. **Límite:** la frontera es siempre una **recta** (hiperplano en más dimensiones). Si los datos
   **no son linealmente separables**, un perceptrón no alcanza → hay que **combinar varios** → red
   neuronal.

Capítulos: 0:00 intro · 1:17 para qué sirve · 4:53 cómo funciona · 9:20 parámetros · 13:36
clasificación · 16:06 funcionamiento completo · 19:46 datos más complejos (el enlace `.url` abre
directo en 4:53).

### 2.2 Conceptos básicos del Deep Learning (10:54)

Vocabulario que el deck da por sabido:

- **Datos estructurados** (tablas, bases de datos) vs **no estructurados** (audio, texto, imágenes, video).
- **Tareas:** clasificación, regresión, transcripción (voz→texto), traducción, *image captioning*,
  detección de anomalías (fraude), texto-a-voz.
- **Representación:** transformar los datos a un espacio donde la tarea sea fácil; el ejemplo de los
  dos grupos que se separan con una recta es exactamente una capa densa + activación.
- **Aprendizaje:** modificación iterativa de los coeficientes del modelo.
- **Entrenamiento:** el proceso completo de iteraciones que ajusta esos coeficientes.
- **Supervisado vs no supervisado:** con o sin la respuesta correcta durante el entrenamiento.

### 2.3 Machine Learning explicado (10:24)

Historia y panorama:

- **Arthur Samuel (1956-1962):** un programa de damas que juega contra sí mismo, guarda posiciones y
  aprende de victorias/derrotas. En 1962 vence a un maestro humano. Nace la idea de aprender de la
  experiencia.
- **Programación tradicional vs ML:** reglas explícitas vs ejemplos. "Receta vs aprender a cocinar
  mirando 100 veces."
- **Tres tipos:** supervisado (datos etiquetados), no supervisado (buscar patrones), por refuerzo
  (recompensa/castigo).
- **Sesgo de datos:** el caso lobo/husky — el modelo aprendió a detectar **nieve**, no lobos. La
  moraleja más importante del video para cualquier proyecto real.
- **Aplicaciones:** recomendación (Amazon/Spotify/Netflix), biometría, spam, diagnóstico por
  imágenes, fraude bancario, tráfico, conducción autónoma.
- **Deep Learning:** capas que van de lo simple a lo complejo (líneas → formas → partes → objeto) y
  "volumen" de las conexiones = pesos. GPT predice la siguiente palabra.

> Detalle de fechas: el video alterna **1956** (la idea) y **1959** (cuando Samuel presenta el
> programa de damas que aprende) y cierra diciendo "1959". Si se toma como referencia histórica,
> conviene fijar una sola: el trabajo de damas es de **1959**.

### 2.4 Prompts efectivos (8:36)

Los 8 errores típicos: prompt vago, sin contexto, sin formato pedido, sin nivel de profundidad, sin
objetivo claro, demasiadas consignas juntas, no refinar el resultado, y creer que la IA adivina
intenciones.

**Fórmula de 4 pasos:** `Objetivo + Contexto + Detalles + Formato final`.

Es la que se usó, según el deck, para escribir el prompt con el que Gemini generó el prompt de
Lovable. Dato meta: la sección de la fórmula está narrada por una **voz sintética** (ElevenLabs) —
otro ejemplo de IA en el flujo de producción.

---

## 3. Recursos que complementan la clase

| Recurso | Para qué |
|---|---|
| <https://keras.io/api/datasets/mnist/> | Documentación oficial del dataset que usa el notebook |
| <https://huggingface.co/models> | Miles de modelos para `pipeline` (el "siguiente paso" del notebook de LLM) |
| <https://huggingface.co/pysentimiento/robertuito-sentiment-analysis> | Ficha del modelo de sentimiento en español que usa el notebook |
| <https://ollama.com> | Instalar y correr LLMs locales (demo 03) |
| <https://docs.ultralytics.com/models/yolov8/> | Parámetros y métricas reales de YOLOv8 (3,2 M para `n`) |
| <https://docs.ultralytics.com/models/yolov5/> | Repo en modo mantenimiento: útil para entender el linaje |
| <https://universe.roboflow.com> | Datasets listos para el notebook de clasificación con dataset propio |
| <https://colab.research.google.com> | Ejecutar los notebooks (runtime con GPU: Entorno de ejecución → Cambiar tipo) |

---

## 4. Cómo se obtuvieron estos datos

1. **Metadatos:** `yt-dlp -J` para título, canal, duración, fecha, capítulos y miniaturas; contraste
   con `https://www.youtube.com/oembed` (HTTP 200 en los 7 videos).
2. **Transcripciones:** subtítulos automáticos en español (`--write-auto-subs`). Seis de siete se
   pudieron bajar; el de MNIST devolvió **HTTP 429 (Too Many Requests)** y quedó sin transcripción.
3. **Limpieza:** se eliminaron marcas de tiempo, etiquetas de formato y las repeticiones que genera
   el sistema de subtítulos "enrollables" de YouTube. **No se corrigió** el texto: es ASR.
4. **Resúmenes de este documento:** escritos a partir de las transcripciones y los capítulos, no
   copiados del texto automático.
