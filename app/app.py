import streamlit as st
import numpy as np
import pickle
import os

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ==================================================
# ⚙️ CONFIG
# ==================================================

SEQ_LEN = 40

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUTA_MODELO = os.path.join(BASE_DIR, "..", "modelo", "modelo_quijote.keras")
RUTA_TOKENIZER = os.path.join(BASE_DIR, "..", "preprocesadores", "tokenizer.pkl")

# ==================================================
# 🚀 CACHE DE MODELO Y TOKENIZER
# ==================================================

@st.cache_resource
def cargar_modelo():
    return load_model(RUTA_MODELO)

@st.cache_resource
def cargar_tokenizer():
    with open(RUTA_TOKENIZER, "rb") as f:
        return pickle.load(f)

model = cargar_modelo()
tokenizer = cargar_tokenizer()

vocab_size = len(tokenizer.word_index) + 1

# ==================================================
# 🔥 TOP-K + TOP-P (NUCLEUS SAMPLING PRO)
# ==================================================

def nucleus_sampling(preds, top_p=0.9):
    sorted_idx = np.argsort(preds)[::-1]
    sorted_probs = preds[sorted_idx]

    cumulative = np.cumsum(sorted_probs)

    cutoff = np.where(cumulative >= top_p)[0][0] + 1

    selected_idx = sorted_idx[:cutoff]
    selected_probs = preds[selected_idx]

    selected_probs = selected_probs / np.sum(selected_probs)

    return np.random.choice(selected_idx, p=selected_probs)

# ==================================================
# 🧠 GENERADOR SENIOR
# ==================================================

def generar_texto(model, tokenizer, seed_text, n_palabras=80, temperatura=0.8):

    resultado = seed_text
    seq_len = model.input_shape[1]

    seen = {}  # 🔥 control de repetición real

    for _ in range(n_palabras):

        seq = tokenizer.texts_to_sequences([resultado])[0]

        seq = pad_sequences(
            [seq],
            maxlen=seq_len,
            padding="post",
            truncating="post"
        )

        pred = model.predict(seq, verbose=0)[0]

        # ==================================================
        # 🔥 ESTABILIDAD NUMÉRICA + TEMPERATURA CORRECTA
        # ==================================================

        pred = np.log(pred + 1e-10) / temperatura
        pred = np.exp(pred)
        pred = pred / np.sum(pred)

        # ==================================================
        # 🔥 NUCLEUS SAMPLING (MEJOR QUE TOP-K)
        # ==================================================

        idx = nucleus_sampling(pred, top_p=0.9)

        # ==================================================
        # 🔥 PENALIZACIÓN DE REPETICIÓN
        # ==================================================

        seen[idx] = seen.get(idx, 0) + 1

        if seen[idx] > 2:
            continue

        palabra = tokenizer.index_word.get(idx, "")

        if palabra in ["", "<OOV>"]:
            continue

        resultado += " " + palabra

    return resultado

# ==================================================
# 🎨 UI
# ==================================================

st.title("🪶 Don Quijote LSTM - Generator PRO")

st.markdown("""
Modelo LSTM optimizado con **Nucleus Sampling + Temperature Control**  
Mejora significativa en coherencia y reducción de repetición.
""")

# ==================================================
# INPUTS
# ==================================================

texto_inicial = st.text_area(
    "Texto inicial:",
    "En un lugar de la Mancha cuyo nombre no quiero recordar"
)

if texto_inicial.strip() == "":
    texto_inicial = "En un lugar de la Mancha"

temperatura = st.slider(
    "🔥 Creatividad (temperatura)",
    0.1, 1.5, 0.8, 0.1
)

n_palabras = st.slider(
    "📝 Número de palabras",
    20, 200, 100, 10
)

# ==================================================
# BOTÓN
# ==================================================

if st.button("🚀 Generar texto"):

    with st.spinner("Generando texto estilo Quijote..."):

        resultado = generar_texto(
            model,
            tokenizer,
            texto_inicial,
            n_palabras,
            temperatura
        )

    st.success("Texto generado correctamente ✔")

    st.markdown("## 📜 Resultado")

    st.write(resultado)

# ==================================================
# SIDEBAR
# ==================================================

st.sidebar.title("⚙️ Configuración avanzada")

st.sidebar.write("""
### 🔥 Técnicas usadas
- Nucleus Sampling (top-p)
- Temperature scaling
- Repetition penalty
- LSTM stacked model

### 📚 Dataset
- Don Quijote de la Mancha

### 🧠 Mejora clave
- Control de diversidad real
- Menos loops de palabras
""")