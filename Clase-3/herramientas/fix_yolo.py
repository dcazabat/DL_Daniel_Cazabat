import sys
sys.path.insert(0, "/tmp/opencode")
from nbedit import NB

BASE = "/home/dcazabat/Documentos/Cursos/DiploIA/Clase_3_revision"
HEADER = """
---

> 📘 **Versión corregida — 16/11/2026.** Copia de trabajo con las correcciones documentadas en
> `ERRORES_ENCONTRADOS.md` (proyecto `Clase_3_revision`). En el código, cada cambio está marcado
> con `# [CORREGIDO]`. El archivo original no se modificó.

**Grupo:** `____` · **Integrantes:** `____, ____, ____, ____`
"""


def add_group_header(nb, cell=0):
    src = nb.src(cell)
    nb.set(cell, src.rstrip("\n") + "\n" + HEADER, "cabecera: nota de versión + datos de grupo")


# ══════════════════════════════════════════════════════════════════════
# YOLOv5
# ══════════════════════════════════════════════════════════════════════
nb = NB(f"{BASE}/originales/notebook_clase3_YOLOv5_Clasificacion.ipynb", "yolov5")
add_group_header(nb)

nb.rep(2, """!git clone https://github.com/ultralytics/yolov5  # clonar
%cd yolov5
%pip install -qr requirements.txt  # instalar

import torch

import utils

display = utils.notebook_init()  # comprobar""",
       """# [CORREGIDO] se clona una versión etiquetada (v7.0) en lugar de `master`:
# el notebook queda reproducible aunque el repo cambie.
import os

if os.path.basename(os.getcwd()) != 'yolov5':
    if not os.path.isdir('yolov5'):
        !git clone --depth 1 --branch v7.0 https://github.com/ultralytics/yolov5
    %cd yolov5
# Ojo: esto puede reinstalar torch/torchvision. Si Colab pide reiniciar el
# entorno después de instalar, hacelo y volvé a ejecutar esta celda.
%pip install -qr requirements.txt

import torch
import utils

# [CORREGIDO] PyTorch 2.6+ usa weights_only=True por defecto. YOLOv5 v7.0
# necesita compatibilidad para cargar sus checkpoints oficiales/confiables.
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

# [CORREGIDO] sin asignar a `display`: antes se pisaba IPython.display.display
utils.notebook_init()
print('PyTorch', torch.__version__, '| CUDA disponible:', torch.cuda.is_available())""",
       "E30: clon sin versión fija + se pisaba IPython.display + no se informaba GPU")

nb.rep(4, """!python classify/predict.py --weights yolov5s-cls.pt --img 224 --source data/images
# display.Image(filename='runs/predict-cls/exp/zidane.jpg', width=600)""",
       """import os

# [CORREGIDO] compatibilidad con PyTorch 2.6+ para el checkpoint oficial.
os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD'] = '1'

!python classify/predict.py --weights yolov5s-cls.pt --img 224 --source data/images
# display.Image(filename='runs/predict-cls/exp/zidane.jpg', width=600)""",
       "V5-08: PyTorch 2.6+ rechazaba el checkpoint de clasificación de YOLOv5")

nb.rep(7, """# Descargar Imagenet val (6.3G, 50000 imágenes)
!bash data/scripts/get_imagenet.sh --val""",
       """# ⚠️ Descarga 6,3 GB (50.000 imágenes): varios minutos y mucho disco.
# Si no tenés espacio/tiempo, saltá esta celda y validá sobre Imagenette
# en la celda siguiente (DATA_VAL = 'imagenette160', ~1.400 imágenes).
!bash data/scripts/get_imagenet.sh --val""",
       "E31: descarga de 6,3 GB sin advertencia ni alternativa liviana")

nb.rep(8, """# Validar YOLOv5s en Imagenet val
!python classify/val.py --weights yolov5s-cls.pt --data ../datasets/imagenet --img 224 --half""",
       """import torch   # [CORREGIDO] autocontenida: no depende de la celda de instalación

# Validar YOLOv5s en ImageNet val (o en Imagenette si no bajaste ImageNet)
DATA_VAL = '../datasets/imagenet'   # alternativa liviana: 'imagenette160'

# [CORREGIDO] --half exige GPU; en un runtime de CPU el comando fallaba
HALF = '--half' if torch.cuda.is_available() else ''
!python classify/val.py --weights yolov5s-cls.pt --data {DATA_VAL} --img 224 {HALF}""",
       "E32: --half en runtime de CPU rompía la validación")

nb.rep(10, 'logger = "Comet"  # @param [\'Comet\', \'ClearML\', \'TensorBoard\']',
       """# [CORREGIDO] TensorBoard por defecto: Comet y ClearML piden cuenta y API key
# (si elegís Comet, `comet_ml.init()` abre un login interactivo).
logger = "TensorBoard"  # @param ['Comet', 'ClearML', 'TensorBoard']""",
       "E33: logger por defecto requería API key de Comet")

nb.rep(19, """model = torch.hub.load(
    "ultralytics/yolov5", "yolov5s", force_reload=True, trust_repo=True
)  # o yolov5n - yolov5x6 o personalizado""",
"""# [CORREGIDO] carga v7.0 desde el clon local. `master` ya importa `ultralytics`
# y puede fallar aunque el paquete no esté instalado.
model = torch.hub.load(
    os.getcwd(), "yolov5s", source="local", trust_repo=True
)  # o yolov5n - yolov5x6 o personalizado""",
"E34: torch.hub master reemplazado por el clon local v7.0")

nb.save(f"{BASE}/colab_corregido/notebook_clase3_YOLOv5_Clasificacion_CORREGIDO.ipynb")
print("OK yolov5", nb.log)

# ══════════════════════════════════════════════════════════════════════
# YOLOv8 clasificación (Roboflow)
# ══════════════════════════════════════════════════════════════════════
nb = NB(f"{BASE}/originales/notebook_clase3_train_yolov8_clasificacion_custom_dataset_.ipynb", "yolov8")
add_group_header(nb)

nb.rep(0, "**Grupo:** `____` · **Integrantes:** `____, ____, ____, ____`",
       """**Grupo:** `____` · **Integrantes:** `____, ____, ____, ____`

> 🔧 **Actualización 18/11/2026 — celdas autocontenidas y vista de la imagen (V8-09, V8-10).**
> Ya no aparecen `NameError: YOLO`, `NameError: HOME` ni `ModuleNotFoundError: ultralytics` al
> ejecutar una celda suelta: la celda **🔧 Entorno** (después de `HOME = os.getcwd()`) instala e
> importa lo necesario. La vista de la imagen anotada busca el archivo más reciente en
> `runs/classify/predict*` (sirve con `predict2`, `predict3`… y con `.jpg`, `.jpeg` o `.png`).
> Las celdas de entrenamiento/validación siguen dependiendo del dataset y del modelo entrenado: ésas
> van en orden.""",
       "V8-09: nota de celdas autocontenidas")

nb.rep(0, """## ⚠️ Descargo de responsabilidad

A partir del 18.01.2023, la Clasificación YOLOv8 parece un poco subdesarrollada. Es posible entrenar modelos, pero su usabilidad es cuestionable. Los problemas conocidos incluyen:
- El modelo preentrenado en el conjunto de datos Imagenet opera sobre el ID de las clases y no sobre sus nombres. Solo después de un posprocesamiento personalizado se puede saber cómo se clasificó la imagen.
- No hay datos de entrenamiento detallados disponibles. En este punto, es casi un estándar guardar información como la matriz de confusión o gráficos de métricas clave después de completar una sesión de entrenamiento. YOLOv8 ofrece esta función pero por el momento solo para Detección de Objetos y Segmentación de Instancias.
- En el caso de la CLI, no se guardan las predicciones en forma de texto. Están anotadas en la imagen, lo que en la práctica imposibilita la construcción de cualquier aplicación que las utilice.
- Con SDK, se devuelve una matriz difícil de interpretar en lugar de un vector de probabilidades. Solo después de un posprocesamiento personalizado se puede saber cómo se clasificó la imagen.""",
       """## ⚠️ Nota histórica sobre la clasificación en YOLOv8

> **Descargo original (18/01/2023):** en esa fecha el modo `classify` estaba poco maduro —el modelo
> preentrenado devolvía *ids* de clase en lugar de nombres, faltaban métricas de entrenamiento y las
> salidas por CLI/SDK eran poco utilizables.

**Estado actual:** ese descargo ya no aplica tal cual. La versión que instala este notebook
(Ultralytics ≥ 8.3) tiene `classify` estable y guarda métricas y predicciones en `runs/classify/`.
Se conserva la nota —actualizada— porque explica los dos hábitos que sí siguen vigentes:

1. En clasificación, la salida es un **vector de probabilidades**: hay que mapearlo a nombres con
   `model.names` (ver la celda del SDK más abajo).
2. Conviene revisar `results.csv` y las imágenes anotadas para interpretar el resultado; el CLI no
   imprime un informe de texto.""",
       "E40: descargo desactualizado (ene-2023) presentado como estado actual")

nb.rep(2, "!nvidia-smi",
       """# Si este comando falla, no hay GPU: Editá Configuración del cuaderno > Acelerador > GPU
!nvidia-smi""",
       "E41: fallo esperado de nvidia-smi sin aclaración")

nb.rep(5, """# Método de instalación Pip (recomendado)

!pip install ultralytics==8.2.103 -q""",
       """# Método de instalación Pip (recomendado)

# [CORREGIDO] no se fija 8.2.103: es anterior a numpy 2.x y en un Colab actual
# puede fallar (AttributeError: module 'numpy' has no attribute 'float_',
# o conflictos de torch/torchvision). Se instala la última versión estable.
%pip install -q -U "ultralytics>=8.3" """.rstrip(),
       "E42: pin 8.2.103 incompatible con numpy 2.x de Colab")

nb.rep(13, """%cd {HOME}
!yolo task=classify mode=predict model=yolov8n-cls.pt conf=0.25 source='https://media.roboflow.com/notebooks/examples/dog.jpeg'""",
       """import os, shutil

# [CORREGIDO] autocontenida: no depende de la celda de HOME ni del orden de ejecución
HOME = globals().get('HOME') or os.getcwd()
if shutil.which('yolo') is None:
    raise RuntimeError('El comando `yolo` no está disponible: ejecutá la celda 🔧 Entorno primero.')
%cd {HOME}

# [CORREGIDO] `conf` es un umbral de detección: no aplica a clasificación
!yolo task=classify mode=predict model=yolov8n-cls.pt imgsz=224 source='https://media.roboflow.com/notebooks/examples/dog.jpeg'""",
       "E43+V8-09: sin conf, con HOME propio y chequeo del CLI")

nb.rep(14, """%cd {HOME}
Image(filename='runs/classify/predict/dog.jpeg', height=600)""",
       """import glob
import os
from IPython.display import Image   # [CORREGIDO] autocontenida

HOME = globals().get('HOME') or os.getcwd()
%cd {HOME}

# [CORREGIDO] la ruta no es fija: cada reejecución del CLI crea predict2, predict3...
# y Ultralytics puede guardar la anotada como .jpg aunque la fuente sea .jpeg.
_candidatas = sorted(
    glob.glob('runs/classify/predict*/*.jpg')
    + glob.glob('runs/classify/predict*/*.jpeg')
    + glob.glob('runs/classify/predict*/*.png'),
    key=os.path.getmtime,
    reverse=True,
)
if not _candidatas:
    raise FileNotFoundError(
        'No hay imágenes anotadas en runs/classify/predict*/. '
        'Ejecutá antes la celda de CLI (task=classify mode=predict).'
    )

_anotada = _candidatas[0]
print('Mostrando:', _anotada)
Image(filename=_anotada, height=600)""",
       "V8-09/V8-10: vista de la imagen con búsqueda dinámica (.jpg/.jpeg y predict2, predict3...)")

nb.rep(16, """model = YOLO(f'{HOME}/yolov8n-cls.pt')
results = model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', conf=0.25)""",
       """import os

try:
    from ultralytics import YOLO   # [CORREGIDO] autocontenida
except ModuleNotFoundError as _e:
    raise ModuleNotFoundError(
        'Falta ultralytics: ejecutá la celda 🔧 Entorno, o '
        '%pip install -U "ultralytics>=8.3" y reiniciá el runtime.'
    ) from _e

# [CORREGIDO] sin HOME ni ruta absoluta: si el .pt no está, Ultralytics lo descarga solo
model = YOLO('yolov8n-cls.pt')
results = model.predict(source='https://media.roboflow.com/notebooks/examples/dog.jpeg', imgsz=224)

# [CORREGIDO] el SDK devuelve probabilidades: para obtener el nombre de la clase
# hay que mapear el índice con model.names (era la limitación del descargo de 2023)
probs = results[0].probs
print('Clase predicha:', model.names[probs.top1], f'({float(probs.top1conf):.1%})')
print('Top-5:', [(model.names[i], round(float(c), 3))
                 for i, c in zip(probs.top5, probs.top5conf)])""",
       "E44+V8-09: SDK autocontenido y con top-5 mapeado a nombres")

nb.rep(18, "Tenga en cuenta elegir el tipo de proyecto correcto. En nuestro caso, Detección de Objetos.",
       "Tenga en cuenta elegir el tipo de proyecto correcto. En nuestro caso, **Clasificación** (no Detección de Objetos).",
       "E45: instrucción copiada de la guía de detección")

nb.rep(20, "!mv {dataset.location}/valid {dataset.location}/val",
       """# [CORREGIDO] idempotente: al reejecutar ya no falla si val/ existe
import os, shutil

_origen, _destino = f"{dataset.location}/valid", f"{dataset.location}/val"
if os.path.isdir(_origen) and not os.path.isdir(_destino):
    shutil.move(_origen, _destino)
print('val/ listo:', os.path.isdir(_destino))""",
       "E46: mv fallaba al reejecutar la celda")

nb.rep(26, """!yolo task=classify mode=val model={HOME}/runs/classify/train/weights/best.pt data={dataset.location}""",
       """# [CORREGIDO] imgsz=128 igual que en el entrenamiento: validar a 224 penaliza la métrica
!yolo task=classify mode=val model={HOME}/runs/classify/train/weights/best.pt data={dataset.location} imgsz=128""",
       "E47: validación a resolución distinta de la de entrenamiento")

nb.rep(28, """!yolo task=classify mode=predict model={HOME}/runs/classify/train/weights/best.pt conf=0.25 source={dataset.location}/test/overripe""",
       """# [CORREGIDO] sin `conf` (no aplica) y con el imgsz del entrenamiento
!yolo task=classify mode=predict model={HOME}/runs/classify/train/weights/best.pt imgsz=128 source={dataset.location}/test/overripe""",
       "E47: predicción a 224 px tras entrenar a 128 px")

nb.new_cell("markdown", """### 🔧 Celda de entorno (ejecutala si reiniciaste el runtime)

Con esta celda ya no hace falta correr el notebook de principio a fin para probar una parte:
deja instalado `ultralytics`, importado `YOLO` y definido `HOME`. Las celdas de entrenamiento,
validación y predicción con el modelo propio sí necesitan el dataset y el modelo entrenado.""", at=4)

nb.new_cell("code", """# ─── ENTORNO: primero esta celda (y cada vez que reinicies el runtime) ───
# [CORREGIDO] V8-09: instala ultralytics si falta, importa YOLO y fija HOME.
# Evita los errores clásicos al ejecutar celdas sueltas:
#   NameError: name 'YOLO' is not defined
#   NameError: name 'HOME' is not defined
#   ModuleNotFoundError: No module named 'ultralytics'

import os, sys

HOME = globals().get('HOME') or os.getcwd()
print('HOME:', HOME)

try:
    import ultralytics
except ModuleNotFoundError:
    print('ultralytics no está instalado: instalando (equivale a %pip install)...')
    import subprocess
    subprocess.check_call(
        [sys.executable, '-m', 'pip', 'install', '-q', '-U', 'ultralytics>=8.3']
    )
    import ultralytics

from ultralytics import YOLO
from IPython.display import display, Image

print(f'ultralytics {ultralytics.__version__} · YOLO importado · HOME listo')""", at=5)

nb.save(f"{BASE}/colab_corregido/notebook_clase3_train_yolov8_clasificacion_custom_dataset_CORREGIDO.ipynb")
print("OK yolov8", nb.log)
