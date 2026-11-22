# TRABAJO PRÁCTICO N° 1 (PARTE 2)

## Gestión de Repositorios y Control de Versiones con GitHub

- **Requisito previo:** Parte 1 terminada en Colab
- **Plataforma:** GitHub.com
- **Objetivo:** Publicar portafolio y README

---

## 1. CREACIÓN DE LA CUENTA Y REPOSITORIO EN GITHUB

### 📌 Paso 1: Crear el repositorio de entrega

1. Ingresa a [github.com](https://github.com) y crea una cuenta o inicia sesión.
2. En la esquina superior derecha, haz clic en el botón **"+"** y selecciona **"New repository"**.
3. En el campo _Repository name_, escribe exactamente:

```
DL_Nombre_Apellido
```

4. Asegúrate de seleccionar la opción **Public (Público)**.
5. Marca la casilla **"Add a README file"**.
6. Haz clic en el botón verde al final: **"Create repository"**.

---

## 2. GUARDAR EL TRABAJO DE COLAB EN GITHUB

### 📌 Paso 2: Vincular y exportar el cuaderno desde Google Colab

1. Vuelve a tu pestaña de Google Colab donde tienes abierto tu archivo `TP1_Nombre_Apellido.ipynb`.
2. En el menú superior, ve a **Archivo ➔ Guardar una copia en GitHub**.
3. Si te pide autorización, inicia sesión en GitHub y concede el acceso a Colab.
4. En la ventana emergente:
   - Selecciona tu repositorio: `DL_Nombre_Apellido`.
   - Verifica que el nombre del archivo sea: `TP1_Nombre_Apellido.ipynb`.
   - Escribe un mensaje de cambio (ejemplo: "Subiendo TP1 finalizado").
5. Haz clic en **Aceptar**. Colab abrirá automáticamente una nueva pestaña mostrando tu archivo subido en tu repositorio de GitHub.

> 💡 **¿Por qué hacemos esto?** GitHub actúa como tu portafolio profesional en la nube, permitiendo a los profesores y colegas revisar tu código, tus explicaciones y el historial de cambios de manera ordenada.

---

## 3. EDICIÓN OBLIGATORIA DEL ARCHIVO README.MD

El archivo `README.md` es la portada explicativa de tu proyecto en GitHub. Debes documentar qué contiene el repositorio y qué aprendiste.

### 📌 Paso 3: Abrir la edición del README en GitHub

1. Entra a tu repositorio en GitHub (`github.com/tu-usuario/DL_Nombre_Apellido`).
2. Haz clic sobre el archivo nombrado `README.md` en la lista de archivos.
3. En el extremo superior derecho del panel del archivo, haz clic en el ícono del Lápiz (✏️ **Edit this file**).
4. Borra el contenido por defecto y copia/pega la siguiente plantilla adaptada con tus datos:

**Plantilla en Markdown para Copiar y Pegar:**

```markdown
# Trabajo Práctico N° 1 - Aprendizaje Profundo

**Alumno:** [Tu Nombre y Apellido]
**Materia:** Inteligencia Artificial / Aprendizaje Profundo
**Fecha:** [Fecha de entrega]

---

## 📁 Contenido del Repositorio

Este repositorio contiene la entrega del primer trabajo práctico introductorio:

- `TP1_Nombre_Apellido.ipynb`: Cuaderno de Google Colab con ejercicios de presupuesto personal, condicionales `if/else`, cálculo de aumentos con bucles `for` y resolución del desafío práctico.

---

## 💡 Conceptos Aprendidos en esta Práctica

1. **Variables:** Guardar datos en nombres representativos (`ingreso_mensual`, `comida`).
2. **Operaciones:** Uso de operaciones matemáticas automáticas (`+`, `-`, `*`, `/`).
3. **Condicionales (`if`/`else`):** Lógica para la toma de decisiones según los valores ingresados.
4. **Bucles (`for`):** Automatización de repeticiones y proyecciones numéricas.
```

### 📌 Paso 4: Guardar los cambios (Commit)

1. Baja hasta el final de la página de GitHub.
2. Haz clic en el botón verde **Commit changes...**
3. En la ventana emergente, vuelve a presionar **Commit changes** para confirmar.
4. ¡Listo! Copia la URL de tu repositorio de GitHub para enviarla en la plataforma del curso.
