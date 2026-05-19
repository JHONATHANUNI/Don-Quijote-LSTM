# Generador de Texto Literario con LSTM — Don Quijote de la Mancha

## Equipo
Jhonathan Uní  
David Pinto

## Asignatura
Inteligencia Artificial — Unidad 4

---

## Descripción del proyecto

Este proyecto implementa un modelo de red neuronal recurrente (LSTM) para la generación de texto a nivel de palabra, entrenado con el corpus de *Don Quijote de la Mancha* de Miguel de Cervantes.

El objetivo es aprender patrones lingüísticos del español clásico para generar texto nuevo con estructura similar al estilo del autor original.

El problema abordado corresponde al modelado de lenguaje (Language Modeling), donde el modelo predice la siguiente palabra en una secuencia.

---

## Problema que resuelve

- Predicción de la siguiente palabra en una secuencia de texto
- Aprendizaje de patrones gramaticales del español
- Generación de texto sintético estilo literario
- Simulación de escritura narrativa basada en corpus histórico

---

## Arquitectura del modelo

- Embedding (256 dimensiones)
- LSTM (512 unidades, return_sequences=True)
- Dropout (0.3)
- LSTM (512 unidades)
- Dropout (0.3)
- Dense (256, activación ReLU)
- Dense (tamaño del vocabulario, activación Softmax)

---

## Técnicas utilizadas

### Preprocesamiento
- Tokenización de texto
- Secuencias de longitud fija (40 tokens)
- Padding de secuencias
- Vectorización a nivel de palabra

### Entrenamiento
- Optimizador Adam
- Función de pérdida: Sparse Categorical Crossentropy
- Early Stopping
- ReduceLROnPlateau

### Generación de texto
- Temperature sampling
- Top-k sampling
- Nucleus sampling (top-p)
- Penalización de repetición
- Bloqueo de n-gramas

---

## Dataset

Fuente: Project Gutenberg  
https://www.gutenberg.org/files/2000/2000-0.txt

Características:
- Aproximadamente 2 MB de texto
- ~400,000 palabras
- Español clásico del siglo XVII

---

## Pipeline del proyecto

1. Carga del corpus
2. Limpieza del texto
3. Tokenización
4. Creación de secuencias
5. Padding
6. División en entrenamiento y validación
7. Entrenamiento del modelo
8. Evaluación
9. Generación de texto

---

## Resultados

- Validation Loss: ~6.3 (dependiente del entrenamiento final)
- Accuracy: ~5%
- Perplexity: alta (esperada en modelos LSTM word-level)

Interpretación:
El modelo aprende patrones básicos del lenguaje, pero presenta limitaciones en coherencia global y repetición de palabras funcionales.

---

## Demo

Aplicación desarrollada con Streamlit para generación de texto en tiempo real.

### Ejecución local

```bash
pip install -r app/requirements.txt
streamlit run app/app.py