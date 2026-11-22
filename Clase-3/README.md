# Clase 3 — revisión, corrección y documentación

Revisión del material de la **Clase 3** de la Diplomatura en IA (UTN 2026, Módulo 2: Deep Learning),
a partir de lo entregado en `Diplomatura_IA_DL/Clase_3/`:

- los 4 notebooks de `Colab_clase_3/`
- el deck `Clase_3.pdf` (10 diapositivas)
- la consigna `tarea.md`
- los 7 videos de la clase (3 obligatorios + 4 de apoyo)

**Fecha de la revisión:** 16/11/2026 · **Actualizado:** 18/11/2026 (fixes de ejecución en Colab)
**Estado:** 44 hallazgos documentados, 5 de severidad alta, todos corregidos en copias nuevas.

> **Repositorio:** <https://github.com/DiploIA-grupo2/DL-clase-3> (privado, rama `main`).
> El repo `Diplomatura_IA_DL` no fue modificado.

---

## Por dónde empezar

| Si querés… | Abrí |
|---|---|
| Ver **qué estaba mal** y cómo se arregló | [`ERRORES_ENCONTRADOS.md`](ERRORES_ENCONTRADOS.md) |
| **Entender la clase** sin leer el PDF | [`RESUMEN_CLASE_3.md`](RESUMEN_CLASE_3.md) |
| Saber **qué dice cada video** (con datos reales) | [`VIDEOS_Y_RECURSOS.md`](VIDEOS_Y_RECURSOS.md) |
| Ver **qué se probó** y qué falta probar | [`VERIFICACION.md`](VERIFICACION.md) |
| Correr los notebooks ya arreglados | [`colab_corregido/`](colab_corregido/) |

---

## Estructura

```
Clase_3_revision/
├── README.md                  este índice
├── ERRORES_ENCONTRADOS.md     41 hallazgos: celda, severidad, evidencia, corrección
├── RESUMEN_CLASE_3.md         resumen de la clase + glosario + repaso
├── VIDEOS_Y_RECURSOS.md       los 7 videos con metadatos reales y resumen
├── VERIFICACION.md            pruebas hechas, resultados y checklist pendiente
│
├── colab_corregido/           ⬅ para usar en clase
│   ├── notebook_clase3_mnist_canvas_2_CORREGIDO.ipynb
│   ├── notebook_clase3_LLM_CORREGIDO.ipynb
│   ├── notebook_clase3_YOLOv5_Clasificacion_CORREGIDO.ipynb
│   └── notebook_clase3_train_yolov8_clasificacion_custom_dataset_CORREGIDO.ipynb
│
├── originales/                copia intacta de lo recibido (referencia)
│   ├── *.ipynb · Clase_3.pdf · tarea.md
│
├── transcripciones/           texto de los videos (automático, sin revisar)
│   ├── 00-nota-mnist-dataset.md   (sin transcripción: YouTube respondió 429)
│   ├── 01-el-perceptron.md
│   ├── 02-como-aprenden-las-maquinas-ml.md
│   ├── 03-conceptos-basicos-deep-learning.md
│   ├── 04-ultralytics-yolo.md
│   ├── 05-que-es-un-llm.md
│   └── 06-prompts-efectivos.md
│
└── herramientas/              scripts que regeneran colab_corregido/ desde originales/
    ├── nbedit.py
    ├── fix_mnist.py
    ├── fix_llm.py
    └── fix_yolo.py
```

---

## Qué se encontró (resumen)

| Notebook | Hallazgos | Lo más grave |
|---|---:|---|
| MNIST canvas | 14 | El panel del canvas se armaba con un `HTML` de IPython dentro de un `VBox` (TraitError en Colab) y el `<script>` no se ejecutaba; además, predicciones sobre píxeles 0-255 en el experimento de rotación |
| LLM | 6 | `max_length` contaba el prompt; `transformers` sin versión fija; GPU nunca usada |
| YOLOv5 | 7 | `--half` en runtime sin GPU rompe la validación |
| YOLOv8 clasificación | 10 | Pin `ultralytics==8.2.103` (oct-2024) incompatible con Colab actual; celdas que dependían del orden de ejecución (`NameError: YOLO/HOME`) y ruta fija de la imagen anotada |
| Deck y tarea | 7 | "9 millones de parámetros" para `yolov8n` (son 3,2 M); demo de Ollama sin material reproducible |

Cada ítem tiene su tabla con **celda, tipo, severidad, evidencia y corrección** en
`ERRORES_ENCONTRADOS.md`.

---

## Cómo usar los notebooks corregidos

1. Subir el `.ipynb` a [Colab](https://colab.research.google.com) (Archivo → Subir notebook).
2. Completar el bloque de **Grupo / Integrantes** al inicio (lo pide `tarea.md`).
3. Entorno de ejecución → **Cambiar tipo de entorno** → GPU (recomendado para YOLO y LLM; MNIST
   funciona en CPU).
4. Ejecutar en orden. Los cambios están marcados con `# [CORREGIDO]`, así que se puede comparar
   contra `originales/` celda por celda.

Los notebooks están **sin salidas** a propósito: el valor está en verlas aparecer en clase.

---

## Cómo dar acceso al docente

`tarea.md` pide que un integrante agregue a **Kalima83** como revisor del repo. Con GitHub CLI:

```bash
gh api -X PUT repos/DiploIA-grupo2/DL-clase-3/collaborators/Kalima83 -f permission=push
```

O desde la web: **Settings → Collaborators → Add people → `Kalima83`** (permiso *Write*), y aceptar
la invitación que llega por mail.

---

## Origen y atribución

- **Deck y consigna:** UTN / Diplomatura en IA 2026 (material del docente).
- **Notebook YOLOv5:** tutorial oficial de Ultralytics (traducido), licencia del repo
  `ultralytics/yolov5` (AGPL-3.0).
- **Notebook YOLOv8-cls:** tutorial de Roboflow Notebooks (traducido).
- **Notebook MNIST y LLM:** material propio de la cátedra.
- **Metadatos y transcripciones de video:** YouTube, obtenidos con `yt-dlp`; las transcripciones son
  automáticas y no están revisadas (ver aviso en cada archivo).

Esta carpeta es **documentación de trabajo**: no reemplaza al material original, lo complementa.
