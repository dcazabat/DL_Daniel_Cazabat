# TRABAJO PRÁCTICO N° 1 (PARTE 1)

## Introducción a la Programación con Google Colab: Tu Primer Asistente Financiero

- **Requisito previo:** Cuenta de Google
- **Plataforma:** Google Colaboratory
- **Nivel:** Principiante (Sin código previo)

---

## 1. ¿QUÉ ES GOOGLE COLAB?

Google Colab es una herramienta gratuita en la nube que te permite escribir y ejecutar código en Python directamente desde tu navegador web. No requiere instalar nada en tu computadora y funciona igual que una hoja de trabajo digital dividida en bloques interactivos llamados **celdas**.

### 📌 Paso 1: Crear tu primer archivo en Colab

1. Abre tu navegador e ingresa a: <https://colab.research.google.com>
2. Inicia sesión con tu cuenta de correo de Google/Gmail si aún no lo has hecho.
3. En la ventana emergente que aparece, haz clic en el botón "Nuevo cuaderno" (o **New Notebook**).
4. En la esquina superior izquierda, haz clic sobre el nombre por defecto (ejemplo: `Untitled0.ipynb`) y cámbialo por:

```
TP1_Nombre_Apellido.ipynb
```

---

## 2. DESARROLLO DEL TRABAJO PRÁCTICO

En este ejercicio construirás paso a paso tu propio programa para gestionar gastos personales y calcular aumentos de sueldo sin necesidad de hacer cuentas a mano.

### Bloque 1: Variables y Cálculos Automáticos

Copia el siguiente código en la primera celda limpia de tu cuaderno de Colab. Luego presiona el botón de Play (▶) ubicado a la izquierda de la celda para ejecutarlo:

```python
# 1. Definimos nuestros datos de entrada (Variables)
ingreso_mensual = 500000
alquiler = 180000
servicios = 45000
comida = 120000

# 2. La computadora realiza los cálculos matemáticos
gastos_totales = alquiler + servicios + comida
dinero_libre = ingreso_mensual - gastos_totales

# 3. Mostramos los resultados por pantalla
print("--- RESUMEN DE TU MES ---")
print("Tus gastos totales son: $", gastos_totales)
print("Te quedan libres para disfrutar: $", dinero_libre)
```

### Bloque 2: La Computadora Toma Decisiones (Condicionales If / Else)

Crea una nueva celda de código en Colab presionando el botón **"+ Código"** en la barra superior. Copia este bloque y ejecútalo:

```python
# Evaluamos la cantidad de dinero libre que nos queda
if dinero_libre > 100000:
    print("🎉 ¡Excelente mes! Puedes guardar un poco para ahorrar o darte un gusto.")
elif dinero_libre > 0:
    print("⚠️ Llegas a fin de mes, pero cuidado con los gastos extras.")
else:
    print("🚨 ¡Atención! Estás gastando más de lo que ingresa. Hay que ajustar el presupuesto.")
```

### Bloque 3: Simular Proyección de Sueldo (Bucles For)

Agrega otra celda de código para proyectar un incremento del 10% mensual en tu sueldo a lo largo de los próximos 6 meses:

```python
sueldo_actual = 500000
porcentaje_aumento = 1.10  # Representa un aumento del 10%

print("--- PROYECCIÓN DE SUELDO A 6 MESES ---")

# Repetimos el cálculo automáticamente durante 6 meses
for mes in range(1, 7):
    sueldo_actual = sueldo_actual * porcentaje_aumento
    print("Mes", mes, ": Ganarás $", round(sueldo_actual))
```

### 🏆 Desafío Final: Repartir la cuenta de una cena

Agrega una última celda en tu Colab y resuelve el siguiente ejercicio práctico por tu cuenta:

```python
# Completa los valores y calcula cuánto debe pagar cada amigo
precio_pizza = 12000
cantidad_amigos = 4
pago_por_persona = precio_pizza / cantidad_amigos
print("Cada amigo debe pagar: $", pago_por_persona)
```

---

### 💡 ¿Qué aprendimos sin darnos cuenta?

1. **Variables:** Guardar información en palabras fáciles de entender (ejemplo: `ingreso_mensual`, `comida`).
2. **Operaciones:** Realizar matemática automática con los símbolos `+`, `-`, `*` y `/`.
3. **Condicionales (`if` / `else`):** Darle criterio o lógica al programa para tomar decisiones automáticamente según los datos.
4. **Bucles (`for`):** Hacer que la computadora repita tareas repetitivas por nosotros miles de veces en un segundo.
