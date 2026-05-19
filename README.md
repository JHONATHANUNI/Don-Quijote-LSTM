# Generador de Texto Literario — Don Quijote LSTM

## Equipo
[Jhonathan y David]

## Asignatura
Inteligencia Artificial — Unidad 4

## Descripción

Proyecto de generación de texto literario utilizando redes neuronales recurrentes LSTM apiladas entrenadas con el corpus completo de Don Quijote de la Mancha de Miguel de Cervantes.

El modelo aprende patrones:
- gramaticales,
- estilísticos,
- narrativos,
- y semánticos

para generar texto nuevo inspirado en el estilo cervantino.

---

# Arquitectura del modelo

- Embedding(128)
- LSTM(256)
- Dropout(0.3)
- LSTM(128)
- Dropout(0.3)
- Dense + Softmax

---

# Dataset

Fuente oficial:
https://www.gutenberg.org/files/2000/2000-0.txt

Corpus:
- ~2 MB
- ~400,000 palabras
- Español clásico del siglo XVII

---

# Tecnologías utilizadas

- Python
- TensorFlow
- Keras
- Streamlit
- NumPy
- Pandas
- Matplotlib
- WordCloud

---

# Resultados

- Validation Loss: X.XX
- Accuracy: XX%
- Perplexity: XXX

---

# Demo Streamlit

## Ejecutar localmente

```bash
pip install -r app/requirements.txt
streamlit run app/app.py