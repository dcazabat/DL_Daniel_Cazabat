import sys
sys.path.insert(0, "/tmp/opencode")
from nbedit import NB

SRC = "/home/dcazabat/Documentos/Cursos/DiploIA/Clase_3_revision/originales/notebook_clase3_LLM.ipynb"
OUT = "/home/dcazabat/Documentos/Cursos/DiploIA/Clase_3_revision/colab_corregido/notebook_clase3_LLM_CORREGIDO.ipynb"

nb = NB(SRC, "llm")

HEADER = """
---

> 📘 **Versión corregida — 16/11/2026.** Copia de trabajo con las correcciones documentadas en
> `ERRORES_ENCONTRADOS.md` (proyecto `Clase_3_revision`). En el código, cada cambio está marcado
> con `# [CORREGIDO]`. El archivo original no se modificó.
>
> **Aclaración conceptual:** `gpt2` (124 M de parámetros) **no** es un "modelo de lenguaje grande"
> en el sentido actual (GPT-4, Llama 3: cientos de miles de millones). Se usa porque corre en CPU
> en segundos y permite ver el mecanismo: predecir el siguiente token.

**Grupo:** `____` · **Integrantes:** `____, ____, ____, ____`
"""
nb.rep(0, "## ¿Qué aprenderemos?",
       HEADER.strip().replace("---\n\n", "") + "\n\n## ¿Qué aprenderemos?",
       "cabecera: nota de versión, aclaración GPT-2 y datos de grupo")

# --- instalación ---
nb.rep(2, """# Instalamos la biblioteca transformers
!pip install -q transformers""",
       """# Instalamos la biblioteca transformers
# [CORREGIDO] %pip instala en el kernel activo (no en el shell) y fija la serie
# 4.x, que es la que documenta este cuaderno. La 5.x ya existe: no fue probada
# con este material, así que se evita el salto de major en clase.
%pip install -q "transformers>=4.44,<5"
""",
       "E22: !pip -> %pip y versión mínima")

# --- pipeline de generación con device ---
nb.rep(4, """from transformers import pipeline

# Creamos el pipeline para la tarea de "text-generation" (generación de texto)
# Usaremos el modelo gpt2 básico
generador = pipeline('text-generation', model='gpt2')""",
       """import torch
from transformers import pipeline

# [CORREGIDO] usa la GPU si está disponible (validá con: Runtime > Cambiar tipo > T4 GPU)
DEVICE = 0 if torch.cuda.is_available() else -1

# Creamos el pipeline para la tarea de "text-generation" (generación de texto)
# Usaremos el modelo gpt2 básico
generador = pipeline('text-generation', model='gpt2', device=DEVICE)

print(f'Dispositivo: {"GPU" if DEVICE == 0 else "CPU"}')""",
       "E23: no se usaba la GPU; se informa el dispositivo")

# --- generación: max_new_tokens ---
nb.rep(6, """# Nuestro texto inicial (prompt)
texto_inicial = "Artificial intelligence is going to"

# Generamos el texto. Limitamos la longitud a 50 tokens (palabras/subpalabras)
resultado = generador(texto_inicial, max_length=50, num_return_sequences=1)

# Imprimimos el resultado
print(resultado[0]['generated_text'])""",
       """# Nuestro texto inicial (prompt)
texto_inicial = "Artificial intelligence is going to"

# [CORREGIDO] max_new_tokens = tokens NUEVOS (antes max_length=50 contaba
# también los del prompt, así que la longitud real de la salida variaba).
resultado = generador(
    texto_inicial,
    max_new_tokens=30,
    num_return_sequences=1,
    do_sample=False,        # greedy: la salida es reproducible
)

# Imprimimos el resultado (prompt + continuación)
print(resultado[0]['generated_text'])

# Y solo la parte generada:
continuacion = resultado[0]['generated_text'][len(texto_inicial):]
print('\\n--- solo la continuación ---')
print(continuacion.strip())""",
       "E24: max_length contaba el prompt; ahora max_new_tokens")

# --- sentiment: device ---
nb.rep(8, """analizador_sentimiento = pipeline(
    "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis"
)""",
       """analizador_sentimiento = pipeline(
    "sentiment-analysis",
    model="pysentimiento/robertuito-sentiment-analysis",
    device=DEVICE,   # [CORREGIDO] mismo criterio que el generador
)""",
       "E23: device también en el clasificador")

# --- frases: aclarar etiquetas ---
nb.rep(10, """for frase in frases:
    resultado = analizador_sentimiento(frase)
    etiqueta = resultado[0]['label']
    certeza = resultado[0]['score']""",
       """for frase in frases:
    resultado = analizador_sentimiento(frase)
    etiqueta = resultado[0]['label']   # POS / NEG / NEU
    certeza = resultado[0]['score']""",
       "E25: documenta las etiquetas del modelo")

# --- cierre: siguientes pasos alineados con la Clase 3 ---
nb.rep(11, """**Siguientes pasos sugeridos:**
* Cambia el `texto_inicial` en la sección de generación de texto para ver qué inventa el modelo.
* Prueba tus propias frases en el analizador de sentimientos.
* Explora el [Hugging Face Hub](https://huggingface.co/models) para descubrir otros modelos que puedas cargar en tu función `pipeline`.""",
       """**Siguientes pasos sugeridos:**
* Cambia el `texto_inicial` en la sección de generación de texto para ver qué inventa el modelo.
* Prueba tus propias frases en el analizador de sentimientos.
* Explorá el [Hugging Face Hub](https://huggingface.co/models) para descubrir otros modelos que puedas cargar en tu función `pipeline`.
* Probá un modelo más grande en la misma pipeline: `pipeline('text-generation', model='gpt2-medium')`, o uno en español como `PlanTL-GOB-ES/roberta-base-bne`.
* **Demo de la clase (Ollama):** correr un LLM local no necesita Colab. Instalá [Ollama](https://ollama.com), ejecutá `ollama run llama3` y consumí su API en `http://localhost:11434`. Es lo que se mostró en la demo 03 del PDF.""",
       "E26: cierre alineado con la demo de Ollama de la clase")

# --- nuevas celdas: tokens ---
AT = 7
nb.new_cell("markdown", """## 🔤 Paréntesis: ¿qué es un "token"?

Los modelos no ven letras ni palabras: ven **tokens**, que son fragmentos producidos por un
tokenizer (BPE). Un token puede ser una palabra, parte de una palabra o un signo. Esto explica
por qué `max_new_tokens` es la forma correcta de limitar la longitud: se cuenta en tokens, no en
caracteres.""", at=AT)

nb.new_cell("code", """from transformers import AutoTokenizer

# [CORREGIDO] se agrega el paso que conecta la demo con el concepto de "token"
tokenizer = AutoTokenizer.from_pretrained('gpt2')

frase = "Artificial intelligence is going to"
ids = tokenizer(frase)['input_ids']

print(f'Frase: {frase!r}')
print(f'Cantidad de tokens: {len(ids)} (para {len(frase.split())} palabras)')
for i, tid in enumerate(ids):
    print(f'  {i}: id={tid:<6} -> {tokenizer.decode([tid])!r}')

print('\\n💡 "intelligence" se parte en dos tokens: por eso los modelos "cuentan" tokens, no palabras.')""",
         at=AT + 1)

nb.save(OUT)
print("OK ->", OUT)
for i, tag in nb.log:
    print(f"  celda {i}: {tag}")
