# Verificación — qué se probó y qué no

Fecha: **16/11/2026**. Objetivo: que la copia corregida no introduzca errores nuevos y que cada
hallazgo de `ERRORES_ENCONTRADOS.md` tenga evidencia, no sólo lectura.

---

## 1. Entorno de verificación

| Ítem | Valor |
|---|---|
| SO | Linux (Fedora), kernel x86_64 |
| Python | 3.15 (sistema) |
| numpy / Pillow | 2.5.4 / 12.3.0 |
| nbformat | 5.11.1 (venv aislado en `/tmp/opencode/venv`) |
| TensorFlow | **no instalado** (no hay wheel para Python 3.15) |
| GPU | **no disponible** (`nvidia-smi` ausente) |
| Colab / Roboflow / Ollama | **no disponibles** en este equipo |

Consecuencia: se validó todo lo que no requiere ejecutar entrenamientos ni navegador. Lo que sí
requiere Colab está listado en la sección 6.

---

## 2. Estructura y sintaxis de los notebooks

Se validaron **los 8 archivos** (4 originales + 4 corregidos) con:

1. `nbformat.validate()` — esquema del notebook.
2. `ast.parse()` de cada celda de código, reemplazando las líneas `!`/`%` (magics de IPython, no son
   Python) por `pass` conservando la indentación.

Resultado: **8/8 válidos, 0 errores de sintaxis** en las celdas de código.

### Estadística de cambios (original → corregido)

| Notebook | Celdas | Celdas tocadas | Líneas + / − | `nbformat` | `ast` |
|---|---|---|---|---|---|
| MNIST canvas | 21 → 21 | 11 | +179 / −70 | OK | OK |
| LLM | 12 → 14 | 7 + 2 nuevas | +61 / −8 | OK | OK |
| YOLOv5 | 20 → 20 | 6 | +30 / −8 | OK | OK |
| YOLOv8 clasificación | 29 → 31 | 10 + 2 nuevas | +133 / −19 | OK | OK |

Las celdas no listadas quedaron **byte a byte idénticas** al original.

---

## 3. Prueba del preprocesamiento del canvas (MN-04)

Es el cambio con más riesgo de romper algo, así que se probó **ejecutando** la función
`preparar_imagen()` del notebook corregido contra imágenes sintéticas (fondo negro, trazo blanco,
formato PNG igual al que manda el navegador), y se comparó con la implementación original.

| Caso | Original: bbox / centro | Corregido: bbox / centro |
|---|---|---|
| "3" centrado | 19×20 / (12.9, 13.9) | **20×20 / (13.1, 14.1)** |
| "7" centrado | 19×16 / (11.2, 14.5) | **20×17 / (10.4, 14.6)** |
| Trazo chico en una esquina | 20×23 / (14.0, 12.7) | **18×20 / (13.5, 13.5)** |
| Línea de 1 px | **se pierde** (0 píxeles > 0.1) | **se conserva** |
| Imagen vacía | array en blanco (predice igual) | **`None`** (no predice) |
| Contrato de salida | `(28,28)` float32, rango [0,1] | `(28,28)` float32, rango [0,1] ✓ |

Lo esperable en MNIST canónico es que el dígito ocupe **20×20 píxeles centrado en el campo de 28×28**.
La versión corregida cae sistemáticamente ahí; la original depende del grosor del trazo y del tamaño
del recorte.

Comando reproducible: ver `herramientas/` (los notebooks se regeneran desde los originales con
`nbedit.py` + `fix_*.py`, usando anclas que deben coincidir **exactamente una vez**; si un ancla no
matchea, el script falla en lugar de escribir algo a medias).

---

## 4. Verificación de datos y afirmaciones

| Afirmación | Cómo se verificó | Resultado |
|---|---|---|
| 109.386 parámetros en la red del notebook | Aritmética de las capas: 784·128+128 = 100.480; 128·64+64 = 8.256; 64·10+10 = 650 | ✓ coincide con el deck |
| MNIST: 70.000 imágenes (60.000 + 10.000) | Documentación de Keras + `assert` agregado al notebook | ✓ |
| `yolov8n` = 3,2 M de parámetros | Tabla oficial de Ultralytics (docs/models/yolov8) | ✓ → el "9 millones" del deck no corresponde a v8n |
| Última versión de `ultralytics` | PyPI JSON: **8.4.215** | ✓ (el pin 8.2.103 del notebook es de oct-2024) |
| Última versión de `transformers` | PyPI JSON: **5.23.0**, 48 releases 5.x | ✓ → se fija la serie 4.x a propósito |
| `tensorflow-cpu` para Python 3.15 | PyPI JSON: 2.21.0, wheels hasta cp313 | ✗ no hay wheel → no se pudo entrenar localmente |
| Títulos/duraciones/canales de los 7 videos | YouTube oEmbed (HTTP 200) + `yt-dlp -J` | ✓ datos reales en `VIDEOS_Y_RECURSOS.md` |
| YOLOv5 clonado en `v7.0` | `git ls-remote --tags` (tag → commit `915bbf2`) + GitHub API: en `v7.0` existen `classify/predict.py`, `train.py`, `val.py` y `tutorial.ipynb` | ✓ los comandos del notebook apuntan a archivos que existen en esa versión |

---

## 5. Búsqueda de errores específicos (evidencia del bug MN-01)

El error de la celda 18 del notebook de MNIST se comprobó leyendo tipos, no "a ojo":

- `X_test` viene de `keras.datasets.mnist.load_data()` → `dtype = uint8`, valores 0-255.
- El entrenamiento usa `X_train_flat` construido con `astype('float32') / 255.0` → rango [0,1].
- La celda 18 hacía `modelo.predict(img_rot.flatten().reshape(1,-1))` con la imagen **sin dividir por
  255**.
- La celda siguiente (19) sí usa `X_test_n` (normalizada), lo que confirma que la 18 era la excepción.

Efecto práctico: con entradas 255× más grandes, las preactivaciones se saturan; la red responde con
alta confianza a clases arbitrarias. El experimento "las redes densas fallan con la rotación" mostraba
en realidad "el código olvidó normalizar".

---

## 6. Qué **no** se pudo verificar (checklist para clase)

| # | Pendiente | Cómo verificarlo en 5 minutos |
|---|---|---|
| 1 | Entrenamiento MNIST real | Colab → runtime CPU → *Run all* → debe dar 97-98 % de test |
| 2 | Canvas JS → ipywidgets | Dibujar un 3 grande; deben funcionar el trazo, los botones ⚡ Predecir / ✕ Borrar, el slider y la miniatura "lo que ve la red" |
| 3 | Rotación corregida | Celda 18: a 0° y 15° debe acertar con confianza alta; a 90° puede fallar (eso sí es el límite real) |
| 4 | YOLOv5 con `--half` | Con GPU: `CUDA disponible: True`; sin GPU: la línea `HALF` queda vacía y el comando corre en FP32 |
| 5 | YOLOv8 con dataset de bananas | Requiere `roboflow.login()` (API key) y GPU; 25 épocas a 128 px |
| 6 | `imgsz=128` en val/predict | Comparar `top1_acc` en `results.csv` (128) contra el default 224: debería mejorar |
| 7 | Notebook de LLM | Colab con GPU: `transformers` 4.x y `torch` precargado; `pipeline` baja GPT-2 (~500 MB) |
| 8 | Demo de Ollama | `ollama run llama3` + `curl localhost:11434/api/tags` |

> Si algo falla en clase, el primer sospechoso es **la versión de las librerías**, no el código: los
> notebooks corregidos fijan versiones justamente para reducir ese ruido.

---

## 7. Cómo se generó esta copia (auditoría)

```
Clase_3_revision/
├── originales/         copia intacta del material recibido (no editar)
├── colab_corregido/    4 notebooks con los cambios marcados [CORREGIDO]
├── herramientas/       scripts que regeneran colab_corregido/ desde originales/
└── *.md                documentación
```

Para regenerar:

```bash
cd herramientas
python3 fix_mnist.py    # escribe colab_corregido/..._CORREGIDO.ipynb
python3 fix_llm.py
python3 fix_yolo.py     # YOLOv5 + YOLOv8
```

Cada script parte de `originales/`, aplica reemplazos con `assert` de unicidad y falla ruidosamente si
el material cambió. No dependen de este equipo salvo por las rutas absolutas, que se pueden ajustar en
las primeras líneas.

---

## 8. Corrección posterior a una ejecución real en Colab (MN-14)

El **17/11/2026** se ejecutó el notebook corregido en Colab y falló al construir el panel del canvas
con un `TraitError` (y, al corregirlo, con un canvas que se veía pero no respondía).

**Causa 1 — `TraitError`:** `widgets.VBox` es un contenedor de **widgets** y se le estaba pasando
`IPython.display.HTML`, que es un objeto de salida de IPython, no un widget. Está en el notebook
original y no se detectó en la primera revisión porque sólo aparece al ejecutar.

**Causa 2 — `<script>` que no corre:** `widgets.HTML` inserta su contenido con `innerHTML`, y el HTML
insertado así **no ejecuta** etiquetas `<script>`. Resultado: canvas visible pero sin listeners.

Cambios aplicados en la celda 16:

| # | Cambio |
|---|---|
| 1 | `widgets.HTML(value=canvas_html)` en lugar de `HTML(canvas_html)` dentro del `VBox`. |
| 2 | El JS se separa en `canvas_js` y se inyecta por el canal de JavaScript: `google.colab.output.eval_js` en Colab, `IPython.display.Javascript` en Jupyter/Lab. |
| 3 | `arrancarCanvas()` reintenta cada 100 ms hasta que `#cv` exista en el DOM, antes de enganchar listeners. |
| 4 | Botones con `addEventListener`; se eliminaron los `onclick` inline del HTML. |
| 5 | Al reejecutar la celda se cierra el panel anterior (sin ids duplicados ni observers viejos). |

**Verificado acá:** `nbformat.validate()` OK, `ast.parse()` OK, y revisión del JS completo (listeners
de mouse y touch, slider, preview, botones y puente al widget oculto).

**No verificable acá:** la ejecución en un navegador con kernel (no hay Colab ni GPU en este equipo).
La corrección se apoya en comportamiento documentado de ipywidgets (`VBox` sólo acepta widgets;
`HTML` renderiza con `innerHTML`) y en el uso estándar de `eval_js`/`Javascript`.

---

## 9. Celdas autocontenidas en YOLOv8-cls (V8-09, 18/11/2026)

Reportado al ejecutar en Colab: al correr una celda suelta, los errores se encadenaron:

```
NameError: name 'YOLO' is not defined
ModuleNotFoundError: No module named 'ultralytics'
NameError: name 'HOME' is not defined
```

**Causa:** el notebook asume ejecución secuencial desde el principio. `HOME` se define en la celda 3,
`YOLO`/`Image` se importan en la 7, y las celdas de inferencia los usan sin volver a declararlos. En
Colab es normal saltear celdas, y el runtime se reinicia solo tras un tiempo de inactividad.

**Cambios aplicados:**

| # | Celda | Cambio |
|---|---|---|
| 1 | nueva (después de `HOME = os.getcwd()`) | **🔧 Entorno**: fija `HOME`, instala `ultralytics` si falta (subprocess con `sys.executable`, equivalente a `%pip`), importa `YOLO`, `display` e `Image`, e informa versiones. |
| 2 | CLI `mode=predict` | `HOME` local, chequeo de `shutil.which('yolo')` y mensaje accionable si falta el comando. |
| 3 | vista de la imagen anotada | importa `Image` en la propia celda y avisa (en vez de `FileNotFoundError`) si falta correr la celda de CLI. |
| 4 | SDK `model.predict` | `YOLO('yolov8n-cls.pt')` sin `HOME` (Ultralytics descarga los pesos si no están) y `ModuleNotFoundError` con la instrucción exacta. |
| 5 | YOLOv5, validación | se agregó `import torch` en la celda (mismo patrón, por si se ejecuta suelta). |

**Verificado acá:** `nbformat.validate()` OK, `ast.parse()` OK en las 31 celdas, y revisión manual de
las cuatro celdas modificadas. La API usada (`Probs.top1`, `top1conf`, `top5`, `top5conf`,
`model.names`) es la documentada por Ultralytics para clasificación.

**No verificable acá:** la ejecución real (requiere instalar `ultralytics` + torch y bajar el modelo y
la imagen). Para probarlo en Colab: runtime nuevo → ejecutar **sólo** la celda 🔧 Entorno → ejecutar
**sólo** la celda del SDK; debe imprimir la clase y el top-5 sin `NameError`.

---

## 10. Vista de la imagen anotada con búsqueda dinámica (V8-10, 18/11/2026)

Segundo error reportado en Colab: `FileNotFoundError` al mostrar la imagen que genera la inferencia.

Dos causas encadenadas:

1. La ruta era literal (`runs/classify/predict/dog.jpeg`), pero Ultralytics crea **`predict2`,
   `predict3`…** en cada reejecución del comando.
2. La imagen se guarda como **`.jpg`** aunque la fuente sea `dog.jpeg`.

La celda ahora busca con `glob` en `runs/classify/predict*/*.{jpg,jpeg,png}`, ordena por fecha de
modificación y muestra la más reciente; si no hay ninguna, avisa que falta correr la inferencia.

**Probado ejecutando el código real de la celda** (sin el magic `%cd` ni la llamada a `Image`) contra
estructuras de carpetas simuladas:

| Escenario | Resultado |
|---|---|
| `predict/dog.jpeg` | elige `predict/dog.jpeg` |
| `predict/dog.jpeg` + `predict3/dog.jpg` (más nuevo) | elige `predict3/dog.jpg` |
| `predict2/banana.png` (otro nombre y extensión) | elige `predict2/banana.png` |
| sin `runs/` | `FileNotFoundError` con el mensaje accionable |

Salida real de los cuatro casos: coincidieron con lo esperado.
