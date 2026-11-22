# Errores y problemas encontrados — Clase 3 (Módulo 2: Deep Learning)

Revisión del material de la **Clase 3** de la Diplomatura en IA (UTN, cohorte 2026):
los 4 notebooks de `Diplomatura_IA_DL/Clase_3/Colab_clase_3/`, el deck `Clase_3.pdf` y `tarea.md`.

- **Fecha de revisión:** 16/11/2026 · **Actualizado:** 18/11/2026 (MN-14 y V8-09: fixes de ejecución en Colab)
- **Material original:** `Diplomatura_IA_DL/Clase_3/` (no se modificó: hay copia intacta en `originales/`)
- **Copia corregida:** `colab_corregido/` — cada cambio está marcado en el código con `# [CORREGIDO]`
- **Cómo se verificó cada cosa:** ver `VERIFICACION.md`

---

## Resumen

| Notebook | Hallazgos | Altos | Medios | Bajos |
|---|---:|---:|---:|---:|
| `notebook_clase3_mnist_canvas_2` | 14 | 3 | 4 | 7 |
| `notebook_clase3_LLM` | 6 | 0 | 3 | 3 |
| `notebook_clase3_YOLOv5_Clasificacion` | 9 | 1 | 6 | 2 |
| `notebook_clase3_train_yolov8_...custom_dataset_` | 10 | 1 | 4 | 5 |
| Material de clase (PDF) y `tarea.md` | 7 | 0 | 3 | 4 |
| **Total** | **46** | **5** | **21** | **20** |

**Los 5 hallazgos de severidad alta:**

1. **MN-01** — el experimento de rotación predice sobre píxeles 0-255 en vez de 0-1: los resultados que se muestran son inválidos.
2. **MN-02** — el notebook pide "guardar el modelo como JSON" y esa celda no existe: el alumno queda bloqueado antes del canvas.
3. **MN-14** — el panel del canvas pasa un `IPython.display.HTML` a un `widgets.VBox` (TraitError en Colab) y, aunque no fallara, el `<script>` embebido en `widgets.HTML` no se ejecuta.
4. **V5-01** — `--half` en un runtime de Colab sin GPU: la validación falla.
5. **V8-01** — `ultralytics==8.2.103` fijado contra un Colab actual (numpy 2.x / torch nuevo): instalación con conflictos.

**Leyenda de severidad**

| Sev. | Significado |
|---|---|
| **Alta** | Rompe la ejecución o invalida el resultado que ve el alumno. |
| **Media** | Funciona, pero puede fallar según el entorno, confunde el concepto o desperdicia tiempo. |
| **Baja** | Robustez, estilo, reproducibilidad o prolijidad. |

---

## 1. `notebook_clase3_mnist_canvas_2.ipynb`

| ID | Celda | Tipo | Sev. | Hallazgo | Corrección aplicada |
|---|---|---:|---|---|---|
| MN-01 | 18 | **Bug funcional** | **Alta** | El experimento de rotación hace `img_orig = X_test[idx_prueba]` (uint8, 0-255) y predice con eso. El modelo se entrenó con valores en [0,1], así que las predicciones que se muestran son inválidas: la red "falla" por la escala, no por la rotación. La celda siguiente (19) sí usa `X_test_n`, lo que confirma la inconsistencia. | `img_orig = X_test[idx_prueba].astype('float32') / 255.0` + `np.clip(..., 0, 1)` después de rotar + aclaración en el `print`. |
| MN-02 | 14 y 15 | Doc. / flujo | **Alta** | La Parte 4 dice *"Primero ejecutá la celda de abajo para guardar el modelo como un archivo JSON"*, pero esa celda ya no existe (quedó del enfoque viejo, cuando se embebían los pesos en el HTML). El texto previo de la celda 15 también hablaba de "preparar el modelo para el canvas". | Nota reescrita + la celda 15 ahora valida `assert 'modelo' in globals()` y explica que el canvas usa el modelo en memoria. |
| MN-03 | 6 | Doc. | Media | El diagrama de la arquitectura muestra **un** Dropout; el código tiene `Dropout(0.2)` y `Dropout(0.1)`. | Diagrama con los dos Dropout y total de parámetros (109.386) para que cierre con el deck. |
| MN-04 | 16 | **Bug de robustez** | Media | Preprocesamiento con **doble remuestreo**: redimensiona 248→28 con LANCZOS, aplica el umbral 0.1 sobre esa imagen ya promediada, recorta y vuelve a redimensionar. Un trazo fino puede quedar por debajo del umbral y desaparecer (reproducido: una línea de 1 px se pierde por completo). | Se recorta el trazo a resolución completa, se centra con padding proporcional (~20%) y se reescala **una sola vez** a 28×28, en una función `preparar_imagen()`. Verificado: el dígito queda en un bbox 20×20 centrado, igual que el MNIST canónico. |
| MN-05 | 16 (JS) | Fragilidad | Media | `document.querySelector('input[type=text]')` toma **el primer input de texto de la página**. En Colab conviven otros widgets/entradas: el puente JS→Python puede escribir en el campo equivocado y la predicción no llega. | `img_input.add_class('canvas-bridge')` y el JS busca `.canvas-bridge input` (con fallback al selector anterior). |
| MN-06 | 16 (JS) | Fragilidad | Media | Si se reejecuta la celda del canvas, `img_input.observe(...)` acumula handlers: cada trazo dispara 2+ inferencias y el panel se redibuja varias veces. | Guard `_estado_canvas['ultima']` (misma imagen no se reprocesa) + el panel anterior se cierra al reejecutar la celda. |
| MN-07 | 16 (JS) | Bug menor | Baja | `getPos()` ignora el escalado CSS del canvas: con zoom del navegador o DPI distinto, el trazo aparece desplazado respecto del puntero. | Factor `canvas.width / rect.width` (y lo mismo en Y). |
| MN-08 | 16 (JS) | UX | Baja | Un clic sin arrastre (o un tap en tablet) no dibuja nada: sólo hay `moveTo` y nunca un `stroke`. | Se dibuja un punto con `ctx.arc(...)` en `mousedown` y `touchstart`. |
| MN-09 | 16 | API deprecada | Baja | `Image.LANCZOS` es un alias deprecado desde Pillow 9.1 (`Image.Resampling.LANCZOS` es lo recomendado). | `LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS` (compatible con versiones viejas y nuevas). |
| MN-10 | 10 | Gráfico | Baja | `ax2.set_ylim(85, 100)` fijo: si el modelo entrena peor (menos épocas, CPU lenta, dataset distinto), la curva se corta y parece plana. | Límite inferior calculado desde `historia.history`. |
| MN-11 | 2 | Limpieza | Baja | `import matplotlib.gridspec` sin uso; `warnings.filterwarnings('ignore')` global (oculta warnings reales, incluidos los de deprecación del propio TF); f-string sin placeholder. | Imports limpios, versiones de TensorFlow/Keras impresas, sin filtro global. |
| MN-12 | 19 | Reproducibilidad | Baja | El ruido gaussiano se genera sin semilla: dos alumnos obtienen curvas distintas y no se puede comparar en grupo. | `np.random.seed(42)`. |
| MN-13 | 3 | Verificación | Baja | No se valida nada del dataset; si Colab devolviera otra versión o un cache corrupto, se entrena igual y nadie se enteró. | `assert` de formas (60000/10000 × 28 × 28) y de las 10 clases. |
| MN-14 | 16 | **Bug de ejecución** | **Alta** | `display(widgets.VBox([img_input, HTML(canvas_html), out]))` le pasa un `IPython.display.HTML` a un contenedor de ipywidgets: los `VBox` **sólo aceptan widgets** → `TraitError` y la celda no muestra nada. Y aunque se cambiara por `widgets.HTML`, ese widget inserta su contenido con `innerHTML`: el `<script>` del canvas **no se ejecuta** y el canvas queda inerte (sin listeners, sin botones). *Detectado al ejecutar el notebook en Colab (17/11/2026).* | Panel armado con `widgets.HTML(value=canvas_html)`; el JS se inyecta por el canal de JavaScript (`google.colab.output.eval_js` en Colab, `IPython.display.Javascript` en Jupyter) y espera activamente a que `#cv` esté en el DOM. Botones enlazados con `addEventListener` (sin `onclick` inline). Al reejecutar la celda se cierra el panel anterior: no quedan ids duplicados ni observers viejos. |

---

## 2. `notebook_clase3_LLM.ipynb`

| ID | Celda | Tipo | Sev. | Hallazgo | Corrección aplicada |
|---|---|---:|---|---|---|
| LL-01 | 0 | Docencia | Media | El título y el texto hablan de "Modelos de Lenguaje **Grande**" pero el modelo es `gpt2` (124 M de parámetros): tres órdenes de magnitud menos que GPT-4 o Llama 3. Sin aclararlo, el alumno se lleva una idea equivocada de qué es "grande". | Nota en la cabecera: por qué se usa GPT-2 (corre en CPU, muestra el mecanismo) y qué significa "grande" hoy. |
| LL-02 | 6 | API | Media | `max_length=50` cuenta **también los tokens del prompt**, así que la longitud de la continuación depende del prompt y no es la esperada; además el muestreo por defecto puede no ser reproducible. | `max_new_tokens=30`, `do_sample=False` y se imprime también sólo la continuación. |
| LL-03 | 2 | Entorno | Media | `!pip install -q transformers`: `!pip` instala en el intérprete del *shell*, que en algunos entornos no es el del kernel; y sin piso de versión, el resultado cambia con el tiempo (hoy `transformers` va por la 5.x). | `%pip install -q "transformers>=4.44,<5"` (serie 4.x, la que documenta el cuaderno). |
| LL-04 | 4 y 8 | Rendimiento | Baja | El notebook recomienda GPU en el texto, pero `pipeline(...)` nunca recibe `device`: siempre corre en CPU. | `DEVICE = 0 if torch.cuda.is_available() else -1` y `device=DEVICE` en ambos pipelines, con aviso por pantalla. |
| LL-05 | (nueva) | Contenido | Baja | Se usa `max_new_tokens` sin explicar qué es un **token**, que es justo el concepto que aparece en el video de la tarea. | Celda nueva con `AutoTokenizer`: muestra cómo se parte la frase y por qué "intelligence" son dos tokens. |
| LL-06 | 11 | Cierre | Baja | Los "siguientes pasos" no conectan con la **demo 03 del deck** (LLM local con Ollama), que no tiene material descargable. | Se agrega Ollama + `localhost:11434` y la sugerencia de `gpt2-medium` / modelo en español. |

---

## 3. `notebook_clase3_YOLOv5_Clasificacion.ipynb`

| ID | Celda | Tipo | Sev. | Hallazgo | Corrección aplicada |
|---|---|---:|---|---|---|
| V5-01 | 8 | **Bug de entorno** | **Alta** | `--half` se pasa siempre, pero `--half` es sólo para GPU (en `train.py` de YOLOv5 es un `assert` explícito; en CPU la inferencia en FP16 no está soportada). En un Colab sin GPU la validación falla. | `HALF = '--half' if torch.cuda.is_available() else ''` y se interpola en el comando. |
| V5-02 | 2 | Reproducibilidad | Media | `git clone` de `master` sin tag: cualquier cambio del repo puede romper el notebook (y el repo está en modo mantenimiento). | `--depth 1 --branch v7.0`. |
| V5-03 | 10 | Bloqueo | Media | El logger por defecto es `Comet`, y `comet_ml.init()` pide cuenta/API key: al ejecutar todo, el flujo se detiene en un login. | Default `TensorBoard`, con comentario sobre qué pasa si se elige Comet/ClearML. |
| V5-04 | 7 y 8 | Costo / tiempo | Media | Descarga **6,3 GB** (ImageNet val, 50.000 imágenes) sin advertencia ni alternativa; en clase puede consumir todo el tiempo. | Advertencia + `DATA_VAL` configurable (`'imagenette160'` como opción liviana). |
| V5-05 | 2 | Entorno | Media | `%pip install -qr requirements.txt` puede reinstalar `torch`/`torchvision` y romper el runtime de Colab; el notebook no avisa que hay que reiniciar. | Comentario explícito antes del `%pip`. |
| V5-06 | 2 | Bug menor | Baja | `display = utils.notebook_init()` pisa `IPython.display.display` (y el notebook traduce ese patrón del original en inglés). | `utils.notebook_init()` sin asignación. |
| V5-07 | 2 | Trazabilidad | Baja | No informa versiones ni si hay GPU, justo lo que hace falta para interpretar el `--half` y los tiempos. | `print(torch.__version__, torch.cuda.is_available())`. |
| V5-08 | 2 y 4 | **Compatibilidad** | Media | PyTorch 2.6+ cambia el valor por defecto de `torch.load(..., weights_only=True)` y rechaza las clases internas del checkpoint de YOLOv5 v7.0. | Se establece `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` para cargar únicamente los checkpoints oficiales/confiables usados por el notebook. |
| V5-09 | 19 | **Compatibilidad** | Media | Torch Hub cargaba `master`, cuyo `hubconf.py` actual importa `ultralytics` y dejaba fallar la demo si ese paquete no estaba instalado. | La inferencia carga `yolov5s` desde el clon local v7.0 con `source="local"`, que evita el repositorio `master` y no requiere `ultralytics`. |

> Nota: YOLOv5 está en **modo mantenimiento** y la documentación oficial de Ultralytics recomienda
> YOLO26 para proyectos nuevos. El notebook sirve como material histórico (es el que usa el repo
> `ultralytics/yolov5`), pero conviene decir en clase que la línea vigente es otra.

---

## 4. `notebook_clase3_train_yolov8_clasificacion_custom_dataset_.ipynb`

| ID | Celda | Tipo | Sev. | Hallazgo | Corrección aplicada |
|---|---|---:|---|---|---|
| V8-01 | 5 | **Compatibilidad** | **Alta** | `!pip install ultralytics==8.2.103` es un pin de octubre de 2024. Contra un Colab actual (numpy 2.x, torch nuevo, Python 3.12/3.13) arrastra conflictos de dependencias: downgrade de numpy, errores de ABI con torch, fallos de importación. La última versión hoy es 8.4.215. | `%pip install -q -U "ultralytics>=8.3"` (soporta numpy 2.x) con comentario del motivo. |
| V8-02 | 0 | Doc. | Media | El "descargo de responsabilidad" del 18/01/2023 ("la clasificación está subdesarrollada") se presenta como estado actual; hoy `classify` es estable. Confunde sobre si la práctica sirve o no. | Se convierte en **nota histórica** con el estado actual y las dos advertencias que sí siguen vigentes. |
| V8-03 | 13, 16, 28 | API | Media | `conf=0.25` en `task=classify`: `conf` es un umbral de **detección**; en clasificación no filtra nada. Da la falsa idea de que se está descartando incertidumbre. | Se quita `conf` y se explicita `imgsz` (224 para el modelo COCO, 128 para el modelo propio). |
| V8-04 | 26 y 28 | Métrica | Media | Se entrena con `imgsz=128`, pero se valida y se predice con el default (224): la métrica y las predicciones no corresponden a la resolución de entrenamiento. | `imgsz=128` en `mode=val` y `mode=predict`. |
| V8-05 | 18 | Doc. | Baja | El paso 1 de Roboflow dice "elegir el tipo de proyecto correcto. En nuestro caso, **Detección de Objetos**" — texto copiado de la guía de detección en un tutorial de **clasificación**. | Corregido a "**Clasificación** (no Detección de Objetos)". |
| V8-06 | 20 | Idempotencia | Baja | `!mv {dataset.location}/valid {dataset.location}/val` falla si se reejecuta (el destino ya existe) y corta el flujo. | Movida a Python idempotente con `shutil.move` y chequeo de existencia. |
| V8-07 | 16 | SDK | Baja | `model.predict(...)` devuelve un tensor de probabilidades y el notebook no muestra cómo pasar de índice a **nombre de clase**, que es la limitación que el propio descargo mencionaba. | `model.names[probs.top1]` + confianza con `top1conf`. |
| V8-08 | 2 | Entorno | Baja | `!nvidia-smi` falla en un runtime de CPU y parece un error del notebook. | Comentario: si falla, no hay GPU (cómo cambiarla). |
| V8-09 | 13, 14, 16 + nuevas | **Robustez de ejecución** | Media | Las celdas de inferencia usan `YOLO` (importado en otra celda) y `HOME` (definido en otra). Al ejecutar una celda suelta —o después de que Colab reinicia el runtime— aparecen en cadena `NameError: name 'YOLO' is not defined`, `ModuleNotFoundError: No module named 'ultralytics'` y `NameError: name 'HOME' is not defined`. *Reportado al ejecutar en Colab (17-18/11/2026).* | Celda nueva **🔧 Entorno** (instala `ultralytics` si falta, importa `YOLO`/`Image`, fija `HOME`); CLI/SDK/vista autocontenidas (`HOME` local, `YOLO('yolov8n-cls.pt')` que descarga solo, `Image` importada en la propia celda, chequeo de `yolo` en el PATH) y errores accionables en vez de `NameError`. |
| V8-10 | 14 | **Ruta fija** | Media | La vista de la imagen usa una ruta literal `runs/classify/predict/dog.jpeg`. Falla por dos motivos: cada reejecución del CLI crea `predict2`, `predict3`… y Ultralytics guarda la anotada como **`.jpg`** aunque la fuente sea `.jpeg`. Resultado: `FileNotFoundError` justo en la demo. *Reportado en Colab (18/11/2026).* | `glob` sobre `runs/classify/predict*/*.{jpg,jpeg,png}`, ordenado por fecha de modificación: muestra la salida más reciente, sin importar el nombre de la carpeta ni la extensión, y avisa si todavía no se corrió la inferencia. |

---

## 5. Material de clase: `Clase_3.pdf` y `tarea.md`

| ID | Ubicación | Tipo | Sev. | Hallazgo | Sugerencia |
|---|---|---:|---|---|---|
| CL-01 | Deck, slide MNIST | Inconsistencia | Baja | El diagrama del flujo dice "Tu trazo (**256×256** px)"; el canvas del notebook es de **248×248**. No rompe nada (se reescala), pero un alumno detallista lo va a notar. | Unificar: 248 en el deck o 256 en el canvas. |
| CL-02 | Deck, slide YOLO | Dato inexacto | Media | Dice "una CNN de **9 millones de parámetros**" y luego "modelo pre-entrenado (6 MB)". La doc oficial de Ultralytics da **3,2 M** de parámetros para `yolov8n` (9 M no corresponde a ninguna variante n/s: s son 11,2 M). | Corregir a ≈3,2 M (o cambiar de modelo si se quiere ejemplificar con uno más grande). |
| CL-03 | Deck, slide YOLO | Claim a verificar | Baja | "La versión nano corre a **30+ FPS en una CPU normal**". En la práctica, `yolov8n` a 640 px en CPU suele dar ~10-15 FPS; 30+ se logra con GPU o formatos exportados (ONNX/TensorRT) y resoluciones menores. | Decir "tiempo real en GPU; en CPU, varios FPS" o medirlo en vivo. |
| CL-04 | Deck, demo 03 | Cobertura | Media | La demo de **Ollama + Lovable** no tiene notebook ni guía en la carpeta: el alumno no puede reproducirla (y es la que más interés genera). | Agregar un `README` mínimo: `ollama run llama3`, endpoint `localhost:11434`, y el prompt usado con Lovable/Gemini. |
| CL-05 | Deck, demo 02 | Cobertura | Media | El deck promete "ver una app de detección en tiempo real" (OpenCV + cámara, 4 líneas de código), pero los notebooks entregados hacen **clasificación** por CLI y detección sobre imagen: no hay app con cámara. | Incluir el script de ~30 líneas (`cv2.VideoCapture` + `model(frame)` + `results[0].plot()`) o aclarar que es demo en vivo no distribuida. |
| CL-06 | Deck / tarea | Expectativa | Baja | La tarea asigna "mnist = <video>" y ese video (2 min 36 s, en inglés, de 2023) **solo presenta el dataset**: no enseña a entrenar. Quien lo mire esperando el "paso a paso" se frustra. | Aclarar que es material de contexto y que el paso a paso está en el notebook. |
| TA-01 | `tarea.md` | Forma | Baja | Errores de tipeo: "gupos", "clase_3.pfd", "mnits", "Deberan" sin tilde; y "incluirá como revisor" sin decir **cómo** (Settings → Collaborators del repo). | Corregir y agregar el paso exacto para dar acceso a `Kalima83`. |

---

## 6. Qué se cambió, archivo por archivo

| Original | Copia corregida | Cambios |
|---|---|---|
| `notebook_clase3_mnist_canvas_2.ipynb` | `..._mnist_canvas_2_CORREGIDO.ipynb` | MN-01 … MN-14 (14 puntos) |
| `notebook_clase3_LLM.ipynb` | `..._LLM_CORREGIDO.ipynb` | LL-01 … LL-06 (6 puntos, +2 celdas nuevas) |
| `notebook_clase3_YOLOv5_Clasificacion.ipynb` | `..._YOLOv5_Clasificacion_CORREGIDO.ipynb` | V5-01 … V5-07 (7 puntos) |
| `notebook_clase3_train_yolov8_..._custom_dataset_.ipynb` | `..._custom_dataset_CORREGIDO.ipynb` | V8-01 … V8-09 (9 puntos, +2 celdas nuevas) |

Convenciones de la copia corregida:

- Todos los cambios están comentados en el código con `# [CORREGIDO]` y, cuando aplica, el motivo.
- Al inicio de cada notebook hay un bloque con la fecha de corrección y un espacio para **grupo e integrantes** (lo que pide `tarea.md`).
- Se mantuvo el estilo, el idioma y la estructura de celdas del original: no se reescribieron secciones sanas.
- Las salidas de ejecución están vacías, como corresponde a un material para correr en clase.
- El contenido se publicó en el repositorio privado `DiploIA-grupo2/DL-clase-3`, rama `main`.

---

## 7. Lo que no se pudo verificar (y conviene probar en clase)

| Punto | Por qué | Cómo cerrarlo |
|---|---|---|
| Ejecución real del notebook de MNIST | Este entorno tiene Python 3.15 y **TensorFlow no publica wheels para 3.15** (última: 2.21.0, hasta cp313). Tampoco hay GPU. | Correr 1 época en Colab y comparar: accuracy test ≈ 97-98 %. |
| Canvas interactivo (JS → ipywidgets) | Requiere navegador con kernel de Colab. | Probar dibujando un 3 y viendo la miniatura "lo que ve la red". |
| `ultralytics>=8.3` con el dataset de bananas | Requiere login de Roboflow y GPU. | Correr el notebook completo en Colab con GPU. |
| YOLOv5 con ImageNet val | 6,3 GB de descarga. | Usar la alternativa `imagenette160` para una prueba rápida. |
| Demo de Ollama | No hay Ollama instalado en este equipo. | `ollama run llama3` y probar el endpoint local. |

Detalle de las pruebas que sí se hicieron (sintaxis, `nbformat`, preprocesamiento con imágenes sintéticas,
metadatos de los videos, aritmética de parámetros): `VERIFICACION.md`.
