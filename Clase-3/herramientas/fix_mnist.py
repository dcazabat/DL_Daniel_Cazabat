import sys
sys.path.insert(0, "/tmp/opencode")
from nbedit import NB

SRC = "/home/dcazabat/Documentos/Cursos/DiploIA/Clase_3_revision/originales/notebook_clase3_mnist_canvas_2.ipynb"
OUT = "/home/dcazabat/Documentos/Cursos/DiploIA/Clase_3_revision/colab_corregido/notebook_clase3_mnist_canvas_2_CORREGIDO.ipynb"

nb = NB(SRC, "mnist")

HEADER = """
> 📘 **Versión corregida — 16/11/2026.** Copia de trabajo con las correcciones documentadas en
> `ERRORES_ENCONTRADOS.md` (proyecto `Clase_3_revision`). En el código, cada cambio está marcado
> con `# [CORREGIDO]`. El archivo original no se modificó.
>
> 🔧 **Actualización 17/11/2026:** corregido el `TraitError` de Colab al armar el panel del canvas
> (MN-14) y la ejecución del JavaScript del canvas.

**Grupo:** `____` · **Integrantes:** `____, ____, ____, ____`
"""
nb.rep(0, "> ⚡ **Tiempo estimado:** ~15 minutos para entrenar y ~10 minutos para explorar",
       "> ⚡ **Tiempo estimado:** ~15 minutos para entrenar y ~10 minutos para explorar\n" + HEADER,
       "cabecera: nota de versión + grupo (pedido en tarea.md)")

# --- Celda 2: imports ---
nb.resub(2, r"import numpy as np.*?print\(f'   TensorFlow/Keras listo'\)",
         """import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print('✅ Librerías cargadas')
print(f'   TensorFlow {tf.__version__} | Keras {keras.__version__}')""",
         "E21: quita gridspec sin uso y el filtro global de warnings; imprime versiones")

# --- Celda 3: integridad del dataset ---
nb.rep(3, "print(f'  Valor máximo de píxel: {X_train.max()} (blanco)')",
       """print(f'  Valor máximo de píxel: {X_train.max()} (blanco)')

# [CORREGIDO] chequeo explícito de integridad del dataset
assert X_train.shape == (60000, 28, 28) and X_test.shape == (10000, 28, 28), \\
    f'Formas inesperadas: {X_train.shape} / {X_test.shape}'
assert set(np.unique(y_train)) == set(range(10)), 'Faltan clases en y_train'
print('\\n✅ Integridad verificada: 60.000 train + 10.000 test, clases 0-9')""",
       "E8: verificación de formas y clases")

# --- Celda 6: diagrama con los dos Dropout ---
nb.rep(6, """Capa densa — 64 neuronas — activación ReLU
    ↓
Salida — 10 neuronas — activación Softmax (una por dígito)
```""",
       """Capa densa — 64 neuronas — activación ReLU
    ↓
Dropout (10%) — segunda capa de regularización
    ↓
Salida — 10 neuronas — activación Softmax (una por dígito)
```

Total: **109.386 parámetros entrenables** (100.480 + 8.256 + 650), igual que en la diapositiva.""",
       "E2: el diagrama no mostraba el segundo Dropout")

# --- Celda 10: eje Y dinámico ---
nb.rep(10, "ax2.set_ylim(85, 100)",
       """_y_min = min(historia.history['accuracy'] + historia.history['val_accuracy']) * 100
ax2.set_ylim(max(0, _y_min - 1), 100)  # [CORREGIDO] eje dinámico: no recorta la curva""",
       "E10: límite fijo 85-100 podía recortar las curvas")

# --- Celda 14: instrucción obsoleta del JSON ---
nb.rep(14, """> 💡 Primero ejecutá la celda de abajo para guardar el modelo como un archivo JSON.
> Luego ejecutá la celda del canvas.""",
       """> 💡 El canvas **no** guarda el modelo en disco: usa el objeto `modelo` que ya está entrenado
> en memoria. Por eso tenés que haber ejecutado las **Partes 1 y 2** antes de abrir el canvas.
> Si ves `NameError: modelo`, volvé a correr la celda de entrenamiento.""",
       "E1: la nota pedía guardar un modelo en JSON que no existe")

# --- Celda 15: reemplazo completo ---
nb.resub(15, r"# ─── PREPARAR EL MODELO PARA EL CANVAS.*\Z",
         """# ─── CHEQUEO PREVIO AL CANVAS ───────────────────────────────────────
# [CORREGIDO] No se guarda nada en disco: el canvas usa `modelo` en memoria.

assert 'modelo' in globals(), 'Primero ejecutá la Parte 2 (construir y entrenar la red).'

print("✅ Modelo en memoria listo para recibir imágenes del canvas")
print(f"   Accuracy del modelo: {acc*100:.2f}%")
print()
print("Ejecutá la siguiente celda para abrir el canvas interactivo.")""",
         "E1: la celda 'guardar JSON' no guardaba nada; ahora valida que el modelo exista")

# --- Celda 16: cabecera + helper de preprocesamiento ---
nb.resub(16, r"import base64, io, re\n.*?out = widgets\.Output\(\)",
         """import base64, io
import numpy as np
from PIL import Image
from IPython.display import HTML, display, clear_output
import ipywidgets as widgets

# [CORREGIDO] Image.LANCZOS quedó como alias deprecado en Pillow >= 9.1
LANCZOS = getattr(Image, 'Resampling', Image).LANCZOS

# [CORREGIDO] si la celda se vuelve a ejecutar, cierra el panel anterior: así no
# quedan dos canvas con el mismo id en el DOM ni observers duplicados.
for _w in globals().get('_canvas_previos', []):
    try:
        _w.close()
    except Exception:
        pass
_canvas_previos = []

# Widget oculto para recibir la imagen del canvas
img_input = widgets.Text(value='', layout=widgets.Layout(display='none'))
out = widgets.Output()

# [CORREGIDO] estado para evitar inferencias duplicadas si se reejecuta la celda
_estado_canvas = globals().setdefault('_estado_canvas', {'ultima': None})


def preparar_imagen(data_url, umbral=0.10):
    \"\"\"PNG del canvas → tensor 28×28 estilo MNIST, en rango [0, 1].

    Orden correcto: recortar el trazo a resolución completa, centrar con
    padding proporcional y reescalar UNA sola vez a 28×28. El código anterior
    reescalaba primero a 28×28 y recortaba después, con doble remuestreo y
    riesgo de perder trazos finos al aplicar el umbral.
    \"\"\"
    b64 = data_url.split(',', 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert('L')  # gris

    arr = np.asarray(img, dtype='float32') / 255.0
    ys, xs = np.where(arr > umbral)
    if len(ys) == 0:
        return None

    crop = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape

    # Padding ~20% por lado: el dígito ocupa ~70% del campo (MNIST: 20 px en 28)
    lado = max(h, w)
    margen = max(1, int(round(lado * 0.20)))
    lienzo = np.zeros((lado + 2 * margen, lado + 2 * margen), dtype='float32')
    y0 = (lienzo.shape[0] - h) // 2
    x0 = (lienzo.shape[1] - w) // 2
    lienzo[y0:y0 + h, x0:x0 + w] = crop

    img28 = Image.fromarray((lienzo * 255).astype('uint8')).resize((28, 28), LANCZOS)
    return np.asarray(img28, dtype='float32') / 255.0
""",
         "E12/E13: preprocesamiento con doble remuestreo + alias Pillow deprecado")

# --- Celda 16: cuerpo del handler ---
nb.resub(16, r"    data_url = change\['new'\]\n.*?    x = arr\.flatten\(\)\.reshape\(1, -1\)",
         """    data_url = change['new']
    if not data_url.startswith('data:image/png;base64,'):
        return
    if data_url == _estado_canvas['ultima']:
        return  # [CORREGIDO] misma imagen: evita doble inferencia
    _estado_canvas['ultima'] = data_url

    arr = preparar_imagen(data_url)
    if arr is None:
        return

    # Aplanar y predecir
    x = arr.reshape(1, -1)""",
         "E12/E14: usa el helper y evita inferencias duplicadas")

# --- Celda 16: miniatura de lo que ve la red ---
nb.rep(16, "    barras_json = [round(float(p)*100, 1) for p in probs]",
       """    # [CORREGIDO] miniatura de lo que realmente ve la red (28×28 → 56×56, sin suavizar)
    prev = Image.fromarray((arr * 255).astype('uint8')).resize((56, 56), Image.NEAREST)
    _buf = io.BytesIO()
    prev.save(_buf, format='PNG')
    prev_b64 = base64.b64encode(_buf.getvalue()).decode()

    barras_json = [round(float(p)*100, 1) for p in probs]""",
       "E15: muestra el tensor 28×28 procesado (verificación visual del preproceso)")

nb.rep(16, """          <div class="res-lbl">confianza del modelo</div>""",
       """          <div class="res-lbl">confianza del modelo</div>
          <div style="text-align:center;margin:4px 0 2px;">
            <img src="data:image/png;base64,{prev_b64}" width="56" height="56"
                 style="image-rendering:pixelated;border:1px solid #1e293b;border-radius:6px;">
            <div style="font-size:9px;color:#475569;letter-spacing:0.05em;">lo que ve la red (28×28)</div>
          </div>""",
       "E15: miniatura en el panel de resultado")

# --- Celda 16: registro del observer + clase para el JS ---
nb.rep(16, "img_input.observe(on_image_received, names='value')",
       """# [CORREGIDO] el JS busca el input por clase (ver sendImage): selector estable
img_input.add_class('canvas-bridge')
img_input.observe(on_image_received, names='value')""",
       "E16: selector estable para el puente JS → Python")

# --- Celda 16: getPos con escala CSS + punto al hacer clic ---
nb.rep(16, """function getPos(e) {
  const r = canvas.getBoundingClientRect();
  const s = e.touches ? e.touches[0] : e;
  return [s.clientX - r.left, s.clientY - r.top];
}""",
       """function getPos(e) {
  // [CORREGIDO] convierte coordenadas CSS a coordenadas internas del canvas
  // (si el navegador escala el canvas por CSS, sin esto el trazo se desplaza)
  const r = canvas.getBoundingClientRect();
  const s = e.touches ? e.touches[0] : e;
  return [(s.clientX - r.left) * (canvas.width / r.width),
          (s.clientY - r.top)  * (canvas.height / r.height)];
}""",
       "E17: escala CSS ignorada en getPos")

nb.rep(16, "canvas.addEventListener('touchstart', e => { e.preventDefault(); drawing=true; const[x,y]=getPos(e); ctx.beginPath(); ctx.moveTo(x,y); }, {passive:false});",
       """canvas.addEventListener('touchstart', e => {
  e.preventDefault(); drawing = true;
  const [x, y] = getPos(e);
  ctx.beginPath(); ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
  ctx.fillStyle = '#fff'; ctx.fill();
  ctx.beginPath(); ctx.moveTo(x, y);
}, {passive:false});""",
       "E18b: mismo arreglo del punto en pantallas táctiles")

nb.rep(16, "canvas.addEventListener('mousedown',  e => { drawing=true; const[x,y]=getPos(e); ctx.beginPath(); ctx.moveTo(x,y); });",
       """// [CORREGIDO] un clic sin arrastre también deja marca
canvas.addEventListener('mousedown',  e => {
  drawing = true;
  const [x, y] = getPos(e);
  ctx.beginPath(); ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
  ctx.fillStyle = '#fff'; ctx.fill();
  ctx.beginPath(); ctx.moveTo(x, y);
});""",
       "E18: clic sin arrastre no dibujaba nada")

nb.rep(16, "  const input = document.querySelector('input[type=text]');",
       """  // [CORREGIDO] selector por clase; el fallback mantiene compatibilidad
  const input = document.querySelector('.canvas-bridge input')
             || document.querySelector('input[type=text]');""",
       "E16: el selector tomaba cualquier input de texto de la página")

# --- Celda 18: BUG de normalización antes de rotar ---
nb.rep(18, "img_orig   = X_test[idx_prueba]",
       "img_orig   = X_test[idx_prueba].astype('float32') / 255.0  # [CORREGIDO] normalizar ANTES de rotar",
       "E3 (bug): se predecía sobre píxeles 0-255 en vez de 0-1")
nb.rep(18, "    img_rot = rotate(img_orig, angulo, reshape=False)",
       "    img_rot = np.clip(rotate(img_orig, angulo, reshape=False), 0, 1)  # [CORREGIDO] mantiene rango [0, 1]",
       "E3 (bug): interpolación podía salir de rango")
nb.rep(18, "print('💡 Las redes densas no son invariantes a la rotación.')",
       """print('💡 Las redes densas no son invariantes a la rotación.')
print('   Ojo: rotar sin normalizar (0-255) invalida la predicción, porque el modelo se entrenó con [0, 1].')""",
       "E3: aclaración del error de escala")

# --- Celda 19: reproducibilidad del ruido ---
nb.rep(19, "niveles_ruido = [0, 0.1, 0.2, 0.4, 0.7]",
       """niveles_ruido = [0, 0.1, 0.2, 0.4, 0.7]
np.random.seed(42)  # [CORREGIDO] resultados reproducibles""",
       "E19: ruido aleatorio sin semilla")

# --- Celda 20: tabla resumen ---
nb.rep(20, "| **Arquitectura densa** | Entrada(784) → Dense(128, ReLU) → Dropout → Dense(64, ReLU) → Dense(10, Softmax) |",
       """| **Arquitectura densa** | Entrada(784) → Dense(128, ReLU) → Dropout(0.2) → Dense(64, ReLU) → Dropout(0.1) → Dense(10, Softmax) — 109.386 parámetros |
| **Preprocesamiento de entrada** | El canvas replica el formato MNIST: recorte del trazo, centrado y escala [0, 1] |""",
       "E2: tabla alineada con el código (dos Dropout) + fila de preproceso")

# --- Celda 16: TraitError en Colab (VBox + IPython.HTML) y ejecución del JS ---
nb.rep(16, """<button class="bn bp" id="btn-pred" onclick="sendImage()">⚡ Predecir</button>
        <button class="bn bc" onclick="clearC()">✕ Borrar</button>""",
       """<button class="bn bp" id="btn-pred">⚡ Predecir</button>
        <button class="bn bc" id="btn-clear">✕ Borrar</button>""",
       "MN-14: se quitan los onclick inline (los listeners se enlazan por JS)")

nb.rep(16, "<script>\nconst canvas = document.getElementById('cv');",
       """\"\"\"

# [CORREGIDO] El JS del canvas va aparte: `widgets.HTML` NO ejecuta <script>
# (se inserta con innerHTML) y `VBox` sólo acepta widgets, no HTML de IPython.
canvas_js = r\"\"\"
function arrancarCanvas(intentos) {
  // el widget puede tardar un instante en quedar en el DOM
  if (!document.getElementById('cv')) {
    if ((intentos = (intentos || 0) + 1) > 50) {
      console.warn('MNIST: no se encontró #cv; el widget HTML no se renderizó');
      return;
    }
    setTimeout(function () { arrancarCanvas(intentos); }, 100);
    return;
  }

const canvas = document.getElementById('cv');""",
       "MN-14: el <script> embebido no se ejecutaba")

nb.rep(16, """</script>
\"\"\"

display(widgets.VBox([img_input, HTML(canvas_html), out]))""",
       """document.getElementById('btn-pred').addEventListener('click', sendImage);
document.getElementById('btn-clear').addEventListener('click', clearC);
}

arrancarCanvas();
\"\"\"

# [CORREGIDO] TraitError en Colab: widgets.VBox sólo acepta widgets; antes se le
# pasaba un IPython.display.HTML. Ahora se usa widgets.HTML(value=...).
_panel = widgets.VBox([
    img_input,                          # input oculto: puente JS -> Python
    widgets.HTML(value=canvas_html),    # markup + CSS del canvas
    out,                                # panel de resultados
])
_canvas_previos.extend([img_input, out, _panel])   # para cerrarlos si se reejecuta
display(_panel)

def _inyectar_js(js):
    \"\"\"Ejecuta JS en el documento: Colab usa eval_js, Jupyter el mime Javascript.\"\"\"
    codigo = '(function(){\\n' + js + '\\n})()'
    try:
        from google.colab import output as _colab_output
        _colab_output.eval_js(codigo)
        return
    except Exception:
        pass
    from IPython.display import Javascript
    display(Javascript(codigo))


_inyectar_js(canvas_js)""",
       "MN-14: VBox con HTML de IPython + JS inyectado por el canal correcto")

nb.save(OUT)
print("OK ->", OUT)
for i, tag in nb.log:
    print(f"  celda {i}: {tag}")
