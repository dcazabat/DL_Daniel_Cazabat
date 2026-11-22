# ¿Qué es un LLM? Cómo funcionan los Modelos de Lenguaje de IA

| Campo | Dato |
|---|---|
| Canal | CKA Coaching and Consulting |
| Duración | 4 min 42 s |
| Publicado | 2026-08-24 |
| URL | <https://www.youtube.com/watch?v=JV-KVcFWkmk> |
| Transcripción | automática (subtítulos de YouTube, idioma español) |

> ⚠️ **Transcripción automática sin revisar.** Proviene de los subtítulos autogenerados por
> YouTube: no distingue hablantes, no tiene puntuación confiable y comete errores propios del
> reconocimiento de voz (nombres propios, siglas, cifras). Sirve para buscar y citar ideas, no
> como texto literal. Ver `../VIDEOS_Y_RECURSOS.md` para el resumen curado de cada video.

## Capítulos (según el autor)

| Capítulo | Minuto |
|---|---|
| — | — |

## Transcripción

¿Qué son las LLM? Las LLM o como sus siglas indican Large Language Models son el corazón de la inteligencia artificial generativa basada en texto. Entenderlas bien es clave para comprender cómo funciona la inteligencia artificial generativa. Las LLM son modelos de inteligencia artificial entrenados con enormes cantidades de texto para comprender y generar lenguaje natural de forma coherente y contextual. Una LLM es un modelo que aprende a predecir la siguiente palabra o token, que hablaremos de eso más adelante, en un texto y a partir de esa capacidad puede responder preguntas, redactar documentos, explicar conceptos técnicos, escribir código, resumir información y mantener conversaciones.

Una LLM es una analista junior extremadamente rápido, que ha leído millones de documentos, pero que necesita supervisión, reglas claras y controles de seguridad para no cometer errores graves. Se llaman grandes por tres razones principales. Una, por el tamaño del dataset, o sea, billones de palabras y fragmentos de texto. Dos, cantidad de parámetros, millones o miles de millones de parámetros. Y tres, capacidad de cómputo, requieren GPUs o IPUs potentes. Entrenamiento costoso y complejo. Estos modelos hacen muchísimos cálculos matemáticos en paralelo. Además, entrenar o ejecutar una LLM implica billones de operaciones matemáticas, algo para lo que la CPU tradicional no es eficiente.

GPU, que significa Graphic Processing Unit, es el procesador diseñado para ejecutar miles de operaciones al mismo tiempo. Originalmente creada para gráficos y videojuegos, pero ideal para la inteligencia artificial. Entonces, ¿qué puede hacer una LLM? Uno, el lenguaje natural, ya puede redactar informes, resumir documentos, traducir textos. Dos, código, escribir scripts, dar errores. Tres, ciberseguridad, analizar alertas, explicar incidentes, redactar reportes SOC, ayudar en threat hunting, entre otros. Veamos algunos ejemplos de LLM. Tenemos el GPT-4 que viene desarrollado por la empresa OpenAI. Eh, sectores clave, seguridad, empresas, desarrollo y uso principal, análisis, generación y asistentes.

También tenemos el LM Cloud, eh, desarrollado por la empresa Antro- Anthropic, que se especifi- que en el sector legal, compliance y riesgos. Y por último tenemos uno muy popular llamado llama, desarrollado por la empresa Meta. Ese, los sectores clave son el seguridad, gobiernos y su uso principal en la inteligencia artificial privada y controlada. Antes de concluir, queremos hablar de dos herramientas útiles para los que buscan trabajar sobre modelos LLM open source. Como primero tenemos lo que es Ollama. Ollama es la página oficial de una plataforma que permite ejecutar modelos de lenguaje LLM de forma local.

En tu propio computador o servidor sin depender de la nube. En términos simples, Ollama es una herramienta para usar la inteligencia artificial generativa on premise. Desde su página puedes descargar el software, instalarlo en Windows, el Mac o Linux y ejecutar los modelos que pueden ser desde llama, mistral, eh, gema, phi, entre otros, directamente en tu equipo. Ollama expone una API local, lo cual permite simular a lo que es el OpenAI, lo que facilita integrarlo con scripts de Python, laboratorios de ciberseguridad, SOC simulados o entornos educativos.

En una próxima sección estaremos trabajando sobre un laboratorio utilizando la herramienta Ollama para ejecutar modelos como mistral, eh, que son útiles, ya que su consumo es moderado en cuanto a recursos. Por otro lado, tenemos lo que es el LLM Studio. LLM Studio es una aplicación local o local first que permite descargar, gestionar y ejecutar modelos de lenguaje grandes LLM directamente en tu computadora, sin depender de servicios en la nube. Tiene una, a diferencia de Ollama, tiene una interfaz GUI donde puedes explorar modelos, descargarlos, iniciar chat y ajustar configuraciones sin escribir comandos.

A diferencia de Ollama, tradicionalmente este que trabaja bajo una línea de comandos o CLI, aunque también ofrece opciones gráficas más recientes. Por otro lado, LLM Studio está más enfocado para principiantes y Ollama más para desarrolladores.
