# Resumen — Clase 3: "IA en acción: 3 demos en vivo"

**Diplomatura en IA · UTN 2026 · Módulo 2: Deep Learning**
Material base: `Clase_3.pdf` (10 diapositivas, PowerPoint de Daniel Bazán, 11/11/2026),
los 4 notebooks de `Colab_clase_3/`, `tarea.md` y los 7 videos de la clase.

---

## 1. La idea de la clase en una frase

Todo lo que se vio en las clases 1 y 2 (neurona, perceptrón, capas) **existe y corre**: la clase
baja la teoría a tres demos ejecutables —un clasificador de dígitos entrenado en vivo, un detector
de objetos preentrenado y un chat con un LLM local— para mostrar el ciclo completo: datos →
entrenamiento → inferencia → límites.

El subtítulo del deck lo dice: *"Del código a la realidad — sin instalaciones, solo resultados"*.

---

## 2. Agenda (según el deck)

1. **Bienvenida + contexto** — por qué ver demos.
2. **Demo 01 — MNIST Canvas** — entrenar una red densa en Colab y clasificar dígitos dibujados a mano.
3. **Demo 02 — YOLO** — detección de objetos en tiempo real y cómo está construida la app.
4. **Demo 03 — Chat con LLM local** — app web contra un modelo servido por Ollama.
5. **Cierre + reflexión** — qué tienen en común las tres y próximos pasos.

---

## 3. Demo 01 — MNIST Canvas

### 3.1 El dataset
- **70.000 imágenes** en escala de grises de dígitos escritos a mano: **60.000** para entrenar,
  **10.000** para test (el `keras.datasets.mnist` no trae validación: el notebook la saca con
  `validation_split=0.1`).
- Cada imagen es **28×28 = 784 píxeles**, cada píxel un entero **0-255** (0 = negro, 255 = blanco).
- Para la red, la imagen no es "un tres": es **una matriz de 784 números**. Aprender = encontrar
  qué combinación de números corresponde a cada dígito.

### 3.2 La arquitectura (3 capas, ~109.000 parámetros)

```
Entrada: 784 (28×28 aplanado)
   ↓
Densa 128 + ReLU        784×128 + 128  = 100.480 parámetros
   ↓
Dropout 20 %
   ↓
Densa 64 + ReLU         128×64  + 64   =   8.256 parámetros
   ↓
Dropout 10 %
   ↓
Salida 10 + Softmax      64×10  + 10   =     650 parámetros
                                       ─────────────────────
                                       109.386 parámetros
```

- **Softmax** convierte los 10 valores de salida en probabilidades que suman 1.
- **Dropout** apaga neuronas al azar durante el entrenamiento para que la red no memorice
  (regularización). El deck muestra sólo uno; el código tiene dos (20 % y 10 %).
- Entrenamiento del notebook: `adam` + `sparse_categorical_crossentropy`, 10 épocas, batch 128.
  Resultado esperado: **~97-98 % de accuracy en test**.

### 3.3 Qué pasa cuando dibujás un 7

1. El canvas captura el trazo (blanco sobre negro, igual que MNIST).
2. Se **escala a 28×28** y se **aplana a 784** números.
3. **Forward pass**: se multiplican por los pesos, se suman sesgos, se aplican ReLU.
4. **Softmax** → 10 probabilidades.
5. Salida: `pred = 7, 96 %`, con una barra de confianza por dígito.

El paso 2 es el que más se subestima: el canvas tiene que **replicar el preprocesamiento de MNIST**
(recorte, centrado, escala 0-1). Si no, el modelo ve algo distinto de lo que aprendió y falla aunque
el dibujo sea perfecto.

### 3.4 El límite que abre la próxima puerta
- Rotá el dígito 90° y el modelo se derrumba: **una red densa no es invariante a la rotación**,
  porque aprendió posiciones exactas de píxeles.
- Lo mismo con ruido extremo o con trazos fuera de centro.
- Conclusión de la clase: hace falta que la red aprenda **patrones locales** (bordes, curvas) que se
  puedan reconocer en cualquier posición → **redes convolucionales (CNNs)**, tema del Módulo 3.

---

## 4. Demo 02 — YOLO: detección de objetos

### 4.1 Qué es
**YOLO = You Only Look Once**: una familia de redes que miran la imagen **una sola vez** y en ese
único forward pass devuelven, para cada objeto: si hay algo, **dónde** (bounding box) y **qué es**
(clase + confianza). Se entrena sobre **COCO (80 clases)**: persona, auto, perro, teléfono, silla…

### 4.2 Cómo funciona (versión intuitiva)
- Divide la imagen en una grilla.
- Para cada celda predice presencia, coordenadas y clase — todo en paralelo, sin proponer regiones
  en dos etapas (esa es la diferencia con las arquitecturas tipo R-CNN).
- El modelo **nano** (`yolov8n`) es el más chico: **3,2 M de parámetros**, ~6 MB, pensado para
  tiempo real. (El deck dice "9 millones": ver `ERRORES_ENCONTRADOS.md`, CL-02.)

### 4.3 Las 4 líneas que importan

```python
from ultralytics import YOLO
model = YOLO('yolov8n.pt')          # 1. descarga y carga el modelo preentrenado

cap = cv2.VideoCapture(0)           # 2. abre la cámara
ret, frame = cap.read()             #    frame = imagen numpy (480, 640, 3)

results = model(frame)              # 3. forward pass sobre el frame completo
boxes = results[0].boxes            #    coordenadas + clase + confianza

annotated = results[0].plot()       # 4. dibuja cajas y etiquetas
cv2.imshow('YOLO', annotated)       #    y se repite ~30 veces por segundo
```

**No se entrena nada**: se usa *transfer learning* — un modelo que ya aprendió 80 clases sobre
millones de imágenes y se reutiliza tal cual (o se ajusta con `mode=train` sobre un dataset propio,
como hace el notebook de las bananas).

> Ojo con la diferencia que el deck y los notebooks mezclan: la demo del PDF es **detección**
> (`yolov8n.pt`, 80 clases, cajas), mientras que los notebooks entregados trabajan con
> **clasificación** (`yolov8n-cls.pt`, una etiqueta por imagen). Es la misma familia y la misma API,
> pero el problema y la salida son distintos.

---

## 5. Demo 03 — Chat con LLM local

### 5.1 Qué se ve
Una app tipo ChatGPT que corre **en la computadora del docente**, sin mandar datos a internet:
el modelo lo sirve **Ollama** y la interfaz la generó **Lovable.dev** a partir de un prompt escrito
con **Gemini**.

### 5.2 Las piezas
| Pieza | Qué aporta |
|---|---|
| **Ollama** | Open source; un comando descarga y corre el modelo; expone una **API local en `localhost:11434`**; soporta Llama 3, Mistral, Gemma, Phi… |
| **Lovable.dev** | Genera una app web completa (React + backend) a partir de un prompt. |
| **Gemini** | Se usó para **escribir el prompt** que después se le dio a Lovable. |
| **El LLM** | Llama 3 / Mistral, corriendo local; responde en streaming. |

### 5.3 El flujo
```
Usuario escribe → App (Lovable/React) → API local :11434 → LLM local → respuesta en streaming
```

### 5.4 Los 3 conceptos que deja la demo
1. **Un LLM local es posible**: no hace falta la nube ni pagar por token.
2. **El prompt importa tanto como el modelo**: la misma app con otro prompt es otra app.
3. **IA construida con IA**: el frontend lo escribió una IA, guiada por un prompt de otra IA. Cambia
   el rol del programador (orquestar, revisar, especificar), no lo elimina.

> Esta demo **no tiene notebook** en la carpeta de la clase. Para reproducirla: instalar Ollama,
> `ollama run llama3` y apuntar cualquier cliente a `http://localhost:11434`.

---

## 6. Qué tienen en común las tres demos

| Denominador | Detalle |
|---|---|
| **Los mismos bloques** | Capas densas, pesos, activaciones, función de pérdida: lo del Módulo 2 está en las tres. |
| **Modelos preentrenados** | Nadie entrenó desde cero: MNIST sí (es el ejercicio), YOLO usa COCO y el chat usa Llama 3. **Transfer learning** en acción. |
| **Herramientas abiertas** | Python, Keras, Ultralytics, Ollama: todo gratis y auditable. |
| **IA construida con IA** | El prompt de la app lo generó un modelo; la app la generó otro. |

**Frase para llevarse:** la IA de frontera ya es accesible; la diferencia entre "usarla" y "entenderla"
es saber qué pasa adentro del forward pass.

---

## 7. Números para recordar

| Dato | Valor |
|---|---|
| Imágenes MNIST | 70.000 (60.000 train + 10.000 test) |
| Píxeles por imagen | 784 (28×28), rango 0-255 → normalizado a 0-1 |
| Arquitectura MNIST | 784 → 128 → 64 → 10 |
| Parámetros MNIST | 109.386 |
| Accuracy esperada MNIST | ~97-98 % en test |
| Clases COCO (detección) | 80 |
| Parámetros `yolov8n` | 3,2 M (~6 MB) |
| Puerto de la API de Ollama | `localhost:11434` |
| Capas del ejemplo del video de DL | 1.ª líneas → 2.ª formas → 3.ª partes → 4.ª objeto |

---

## 8. Glosario mínimo

- **Tensor / matriz de entrada**: los datos tal como entran al modelo (para MNIST, 784 números).
- **Peso / sesgo (parámetro)**: números que el modelo ajusta al entrenar.
- **Forward pass**: recorrido de los datos por las capas hasta la salida.
- **ReLU**: activación que deja pasar lo positivo y anula lo negativo.
- **Softmax**: normaliza la salida en probabilidades que suman 1.
- **Loss (pérdida)**: cuánto se equivoca el modelo; es lo que se minimiza.
- **Época (epoch)**: una pasada completa por el set de entrenamiento.
- **Batch**: grupo de ejemplos procesados antes de actualizar los pesos.
- **Overfitting**: memorizar el train y perder generalización (se detecta comparando con validación).
- **Dropout**: apaga neuronas al azar para regularizar.
- **Transfer learning**: reutilizar un modelo ya entrenado y ajustarlo a otra tarea.
- **Token**: unidad de texto (palabra, parte de palabra o signo) que procesa un LLM.
- **Inferencia**: usar el modelo ya entrenado para predecir.

---

## 9. Mapa demo → material

| Demo del deck | Notebook en `Colab_clase_3/` | Video de la tarea |
|---|---|---|
| 01 MNIST Canvas | `notebook_clase3_mnist_canvas_2.ipynb` | MNIST dataset (2:36, contexto) |
| 02 YOLO | `notebook_clase3_YOLOv5_Clasificacion.ipynb` + `..._train_yolov8_clasificacion_custom_dataset_.ipynb` | Ultralytics YOLO (4:36) |
| 03 Chat LLM local | `notebook_clase3_LLM.ipynb` (pipeline de HF, no Ollama) | ¿Qué es un LLM? (4:42) |
| Base teórica | — | El Perceptrón (23:12), Conceptos de DL (10:54), ML explicado (10:24), Prompts (8:36) |

Los tres notebooks de la clase están corregidos y documentados en `colab_corregido/`.

---

## 10. Preguntas de repaso

1. ¿Por qué la entrada de la primera capa tiene 784 neuronas y no 28?
2. ¿Qué hace exactamente Softmax y por qué la capa de salida tiene 10 neuronas?
3. ¿Por qué se normaliza de 0-255 a 0-1 antes de entrenar?
4. ¿Cuántos parámetros tiene la primera capa y por qué? (784×128+128)
5. ¿Para qué sirve Dropout y cómo se detecta que falta?
6. ¿Qué diferencia hay entre `loss` de entrenamiento y de validación?
7. ¿Por qué rotar el dígito rompe la predicción, y qué tipo de red lo resuelve?
8. ¿Qué significa que un modelo "vea" 784 números en lugar de un dibujo?
9. ¿Qué es transfer learning y por qué las tres demos lo usan?
10. ¿Qué diferencia hay entre detección y clasificación en YOLO?
11. ¿Qué es un token y por qué `max_new_tokens` es más preciso que `max_length`?
12. ¿Qué ventaja tiene correr un LLM con Ollama en lugar de usar una API en la nube?

<details>
<summary>Respuestas breves</summary>

1. Porque la capa `Dense` recibe un **vector**; 28×28 se aplana a 784 valores.
2. Convierte 10 puntajes en probabilidades que suman 1; 10 = dígitos 0-9.
3. Porque valores chicos y en el mismo rango estabilizan y aceleran el descenso de gradiente (y la
   inicialización/hechos como Adam están pensados para esa escala).
4. 100.480 = un peso por cada conexión entrada-neurona (784×128) más un sesgo por neurona.
5. Apaga neuronas al azar para evitar que la red dependa de combinaciones específicas; sin él, la
   curva de validación se separa de la de entrenamiento.
6. El de entrenamiento baja siempre; si el de validación empieza a subir, hay overfitting.
7. La red densa aprendió posiciones de píxeles, no formas; una **CNN** (convolución) es invariante a
   traslaciones y aprende patrones locales.
8. Que la imagen es una matriz de números y "reconocer" es encontrar regiones en ese espacio.
9. Reutilizar pesos ya aprendidos en un dataset grande y adaptarlos a la tarea propia: ahorra datos,
   tiempo y cómputo.
10. Detección devuelve cajas + clase para varios objetos (COCO, 80 clases); clasificación devuelve
    una etiqueta por imagen (`-cls`, ImageNet/dataset propio).
11. Un token es la unidad que procesa el modelo; `max_length` cuenta también los del prompt, así que
    la longitud generada queda indeterminada.
12. Privacidad (los datos no salen), costo (sin pago por token) y control (versión, cuantización,
    sin límites de cuota).

</details>

---

## 11. Próxima clase (según el deck)

**Regularización: Dropout, Batch Normalization y Early Stopping.** Se conecta directo con dos cosas
que ya aparecieron acá: el Dropout del notebook de MNIST y las curvas de aprendizaje donde se ve el
overfitting.

---

## 12. Observaciones sobre el material

Las inconsistencias detectadas en el deck (256 vs 248 px, "9 millones" de parámetros, FPS en CPU) y
los errores de los notebooks están documentados, con su corrección, en `ERRORES_ENCONTRADOS.md`.
