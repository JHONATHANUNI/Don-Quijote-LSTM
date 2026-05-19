"""
Don Quijote LSTM Text Generator — Production Refactor
======================================================
Mejoras sobre el código original:
  • Decoding híbrido: top-k + top-p (nucleus) con umbral dinámico
  • Frequency penalty tipo GPT-2 (penalización logarítmica por frecuencia)
  • N-gram blocking (bi-gramas y tri-gramas)
  • Memory buffer de tokens recientes para penalización de ventana corta
  • Anti-stopword collapse con penalización extra a palabras funcionales
  • Softmax numéricamente estable (log-sum-exp)
  • Temperature scaling correcto en espacio logit
  • Logit clipping para evitar overflow/underflow extremos
  • Separación limpia de lógica vs UI
  • Caching correcto con st.cache_resource
"""

from __future__ import annotations

import os
import pickle
import collections
from typing import Optional

import numpy as np
import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUTA_MODELO     = os.path.join(BASE_DIR, "..", "modelo",          "modelo_quijote.keras")
RUTA_TOKENIZER  = os.path.join(BASE_DIR, "..", "preprocesadores", "tokenizer.pkl")

# Palabras funcionales en castellano que tienden a dominar la distribución
# cuando el modelo colapsa. Se les aplica una penalización adicional.
STOPWORDS_ES = frozenset({
    "de", "la", "el", "en", "y", "a", "que", "los", "las", "un", "una",
    "con", "del", "al", "se", "su", "por", "es", "le", "lo", "no", "más",
    "como", "pero", "era", "fue", "una", "sus", "o", "ni", "si", "me",
    "te", "nos", "les", "ya", "sin", "sobre", "entre", "también",
})

# ──────────────────────────────────────────────────────────────────────────────
# CARGA DE RECURSOS (cacheados para toda la sesión)
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Cargando modelo LSTM…")
def cargar_modelo():
    return load_model(RUTA_MODELO)


@st.cache_resource(show_spinner="Cargando tokenizer…")
def cargar_tokenizer():
    with open(RUTA_TOKENIZER, "rb") as f:
        return pickle.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# UTILIDADES NUMÉRICAS
# ──────────────────────────────────────────────────────────────────────────────

def logits_from_probs(probs: np.ndarray) -> np.ndarray:
    """Convierte probabilidades en logits de forma estable (evita log(0))."""
    return np.log(np.clip(probs, 1e-10, 1.0))


def softmax_stable(logits: np.ndarray) -> np.ndarray:
    """Softmax numéricamente estable usando el truco log-sum-exp."""
    shifted = logits - np.max(logits)
    exp_vals = np.exp(np.clip(shifted, -50.0, 50.0))   # logit clipping
    return exp_vals / (np.sum(exp_vals) + 1e-10)


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    """Escala logits por temperatura (siempre en espacio logit, no en probs)."""
    temperature = max(temperature, 1e-3)   # evita división por cero
    return logits / temperature


# ──────────────────────────────────────────────────────────────────────────────
# PENALIZACIONES
# ──────────────────────────────────────────────────────────────────────────────

def apply_frequency_penalty(
    logits: np.ndarray,
    token_counts: dict[int, int],
    frequency_penalty: float = 0.5,
) -> np.ndarray:
    """
    Frequency penalty tipo GPT-2:
        logit[i] -= frequency_penalty * count[i]

    Penaliza proporcionalmente a cuántas veces apareció el token en el texto
    generado hasta ahora. Efectivo para evitar el loop de largo alcance.
    """
    if frequency_penalty == 0.0:
        return logits
    logits = logits.copy()
    for token_id, count in token_counts.items():
        logits[token_id] -= frequency_penalty * count
    return logits


def apply_presence_penalty(
    logits: np.ndarray,
    seen_tokens: set[int],
    presence_penalty: float = 0.3,
) -> np.ndarray:
    """
    Presence penalty: penaliza con un valor fijo cualquier token que ya
    haya aparecido al menos una vez (independientemente de la frecuencia).
    Complementa el frequency penalty.
    """
    if presence_penalty == 0.0:
        return logits
    logits = logits.copy()
    for token_id in seen_tokens:
        logits[token_id] -= presence_penalty
    return logits


def apply_repetition_window_penalty(
    logits: np.ndarray,
    recent_tokens: list[int],
    window: int = 10,
    penalty: float = 1.2,
) -> np.ndarray:
    """
    Penaliza tokens que aparecieron en la ventana reciente de `window` tokens.
    Divides el logit por `penalty` para cada ocurrencia en la ventana.
    Esto es lo que usa la mayoría de modelos modernos (similar a HF transformers).
    """
    if penalty == 1.0:
        return logits
    logits = logits.copy()
    window_counts: dict[int, int] = collections.Counter(recent_tokens[-window:])
    for token_id, count in window_counts.items():
        # División repetida: logit positivo baja, logit negativo sube (coherente)
        if logits[token_id] > 0:
            logits[token_id] /= (penalty ** count)
        else:
            logits[token_id] *= (penalty ** count)
    return logits


def apply_stopword_penalty(
    logits: np.ndarray,
    index_word: dict[int, str],
    recent_tokens: list[int],
    window: int = 4,
    penalty: float = 2.5,
) -> np.ndarray:
    """
    Anti-stopword collapse: si las últimas `window` palabras son todas
    stopwords, aplica una penalización extra a todas las stopwords del
    vocabulario para forzar al modelo a elegir una palabra de contenido.
    """
    recent_words = [index_word.get(t, "") for t in recent_tokens[-window:]]
    all_stops = all(w in STOPWORDS_ES for w in recent_words if w)
    if not all_stops or len(recent_words) < window:
        return logits
    logits = logits.copy()
    for token_id, word in index_word.items():
        if word in STOPWORDS_ES:
            logits[token_id] -= penalty
    return logits


# ──────────────────────────────────────────────────────────────────────────────
# N-GRAM BLOCKING
# ──────────────────────────────────────────────────────────────────────────────

def get_blocked_by_ngrams(
    token_history: list[int],
    n: int = 3,
) -> set[int]:
    """
    Devuelve el conjunto de tokens cuya adición generaría un n-grama
    que ya existe en el historial generado.
    Ejemplo con n=3: si la historia termina en [A, B], bloquea cualquier C
    tal que [A, B, C] ya apareció antes.
    """
    if len(token_history) < n - 1:
        return set()

    tail = tuple(token_history[-(n - 1):])
    blocked: set[int] = set()

    for i in range(len(token_history) - n + 1):
        gram = tuple(token_history[i: i + n])
        if gram[:-1] == tail:
            blocked.add(gram[-1])

    return blocked


# ──────────────────────────────────────────────────────────────────────────────
# SAMPLING
# ──────────────────────────────────────────────────────────────────────────────

def top_k_top_p_sampling(
    probs: np.ndarray,
    top_k: int = 50,
    top_p: float = 0.92,
) -> int:
    """
    Híbrido top-k + nucleus (top-p):
      1. Filtra a los top-k tokens más probables.
      2. Dentro de esos top-k, aplica nucleus: conserva los tokens cuya
         probabilidad acumulada no supera top_p.
      3. Muestrea del subconjunto resultante.

    El top-k actúa como un techo de seguridad que evita que el nucleus
    sampling incluya accidentalmente tokens de muy larga cola.
    """
    vocab_size = len(probs)

    # Paso 1: Top-K
    k = min(top_k, vocab_size)
    top_k_indices = np.argpartition(probs, -k)[-k:]
    top_k_probs = probs[top_k_indices]

    # Ordenar por probabilidad descendente
    order = np.argsort(top_k_probs)[::-1]
    top_k_indices = top_k_indices[order]
    top_k_probs   = top_k_probs[order]

    # Paso 2: Nucleus dentro del top-K
    cumulative = np.cumsum(top_k_probs)
    cutoff_idx = np.searchsorted(cumulative, top_p) + 1
    cutoff_idx = max(cutoff_idx, 1)

    nucleus_indices = top_k_indices[:cutoff_idx]
    nucleus_probs   = top_k_probs[:cutoff_idx]
    nucleus_probs   = nucleus_probs / (nucleus_probs.sum() + 1e-10)

    return int(np.random.choice(nucleus_indices, p=nucleus_probs))


# ──────────────────────────────────────────────────────────────────────────────
# GENERADOR PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def generar_texto(
    model,
    tokenizer,
    seed_text: str,
    n_palabras: int = 100,
    temperatura: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.92,
    frequency_penalty: float = 0.45,
    presence_penalty: float = 0.25,
    repetition_penalty: float = 1.3,
    repetition_window: int = 12,
    ngram_block_size: int = 3,
) -> str:
    """
    Genera texto autoregressivo con todas las técnicas de control de calidad.

    Parámetros
    ----------
    temperature         : escala de entropía (0.1 = muy determinista, 1.5 = muy caótico)
    top_k               : número máximo de candidatos antes del nucleus
    top_p               : umbral de probabilidad acumulada para nucleus sampling
    frequency_penalty   : cuánto penalizar tokens por su frecuencia global (GPT-2 style)
    presence_penalty    : penalización fija por haber aparecido al menos una vez
    repetition_penalty  : factor de penalización para tokens en ventana reciente
    repetition_window   : tamaño de la ventana de memoria corta
    ngram_block_size    : tamaño de n-grama para bloqueo duro de repeticiones
    """
    seq_len = model.input_shape[1]
    index_word = tokenizer.index_word

    resultado_tokens: list[int] = []
    resultado_palabras: list[str] = list(seed_text.split())

    # Historial de tokens del seed para inicializar penalizaciones
    seed_seq = tokenizer.texts_to_sequences([seed_text])[0]
    token_counts: dict[int, int] = collections.Counter(seed_seq)
    seen_tokens: set[int] = set(seed_seq)

    for _ in range(n_palabras):
        # ── Construir secuencia de entrada ───────────────────────────────────
        current_text = " ".join(resultado_palabras)
        seq = tokenizer.texts_to_sequences([current_text])[0]
        seq_padded = pad_sequences(
            [seq], maxlen=seq_len, padding="post", truncating="post"
        )

        # ── Inferencia ───────────────────────────────────────────────────────
        raw_probs = model.predict(seq_padded, verbose=0)[0]

        # ── Espacio logit: todas las penalizaciones se aplican aquí ──────────
        logits = logits_from_probs(raw_probs)

        # 1. Temperature scaling (siempre en logit, no en prob)
        logits = apply_temperature(logits, temperatura)

        # 2. Frequency penalty (largo alcance)
        logits = apply_frequency_penalty(logits, token_counts, frequency_penalty)

        # 3. Presence penalty
        logits = apply_presence_penalty(logits, seen_tokens, presence_penalty)

        # 4. Repetition window penalty (corto alcance, factor multiplicativo)
        logits = apply_repetition_window_penalty(
            logits, resultado_tokens, repetition_window, repetition_penalty
        )

        # 5. Anti-stopword collapse
        logits = apply_stopword_penalty(
            logits, index_word, resultado_tokens, window=4, penalty=2.5
        )

        # ── Convertir de vuelta a probabilidades ─────────────────────────────
        probs = softmax_stable(logits)

        # ── N-gram blocking duro (zeroing) ───────────────────────────────────
        all_tokens = seed_seq + resultado_tokens
        blocked = get_blocked_by_ngrams(all_tokens, n=ngram_block_size)

        # También bloquear bi-gramas
        if ngram_block_size > 2:
            blocked |= get_blocked_by_ngrams(all_tokens, n=2)

        if blocked:
            probs[list(blocked)] = 0.0
            total = probs.sum()
            if total > 1e-10:
                probs /= total
            else:
                # Fallback: distribución uniforme si todo fue bloqueado
                probs = np.ones(len(probs)) / len(probs)

        # ── Sampling híbrido top-k + nucleus ────────────────────────────────
        chosen_idx = top_k_top_p_sampling(probs, top_k=top_k, top_p=top_p)

        # ── Actualizar contadores ────────────────────────────────────────────
        token_counts[chosen_idx] = token_counts.get(chosen_idx, 0) + 1
        seen_tokens.add(chosen_idx)
        resultado_tokens.append(chosen_idx)

        # ── Decodificar token ────────────────────────────────────────────────
        palabra = index_word.get(chosen_idx, "")
        if not palabra or palabra == "<OOV>":
            continue

        resultado_palabras.append(palabra)

    return " ".join(resultado_palabras)


# ──────────────────────────────────────────────────────────────────────────────
# INTERFAZ STREAMLIT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Don Quijote LSTM Generator",
        page_icon="🪶",
        layout="centered",
    )

    st.title("🪶 Don Quijote LSTM — Generator Pro")
    st.caption(
        "Generación narrativa con nucleus sampling híbrido, "
        "frequency penalty y n-gram blocking."
    )

    # ── Cargar recursos ──────────────────────────────────────────────────────
    model     = cargar_modelo()
    tokenizer = cargar_tokenizer()

    # ── Inputs principales ───────────────────────────────────────────────────
    texto_inicial = st.text_area(
        "Texto inicial",
        value="En un lugar de la Mancha cuyo nombre no quiero recordar",
        height=80,
    )
    if not texto_inicial.strip():
        texto_inicial = "En un lugar de la Mancha"

    col1, col2 = st.columns(2)
    with col1:
        temperatura = st.slider("🌡️ Temperatura", 0.3, 1.5, 0.8, 0.05,
                                help="Más alto = más creativo; más bajo = más conservador")
        n_palabras = st.slider("📝 Palabras a generar", 20, 300, 100, 10)
        top_k = st.slider("🎯 Top-K", 10, 100, 50, 5,
                          help="Número máximo de candidatos antes del nucleus sampling")

    with col2:
        top_p = st.slider("🔵 Top-P (nucleus)", 0.70, 0.99, 0.92, 0.01,
                          help="Masa de probabilidad acumulada para nucleus sampling")
        frequency_penalty = st.slider("🔁 Frequency penalty", 0.0, 1.5, 0.45, 0.05,
                                       help="Penaliza tokens usados frecuentemente")
        repetition_penalty = st.slider("🚫 Repetition penalty", 1.0, 2.0, 1.3, 0.05,
                                        help="Penaliza tokens recientes (ventana corta)")

    # ── Generación ───────────────────────────────────────────────────────────
    if st.button("🚀 Generar texto", type="primary", use_container_width=True):
        with st.spinner("Generando texto estilo Quijote…"):
            resultado = generar_texto(
                model=model,
                tokenizer=tokenizer,
                seed_text=texto_inicial,
                n_palabras=n_palabras,
                temperatura=temperatura,
                top_k=top_k,
                top_p=top_p,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
            )

        st.success("Texto generado ✔")
        st.markdown("### 📜 Resultado")
        st.write(resultado)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Técnicas activas")
        st.markdown("""
**Decoding**
- Hybrid Top-K + Nucleus (Top-P)
- Temperature scaling en logits

**Penalizaciones**
- Frequency penalty (GPT-2 style)
- Presence penalty
- Repetition window penalty
- Anti-stopword collapse

**Bloqueo duro**
- Bi-gram blocking
- Tri-gram blocking

**Estabilidad**
- Softmax log-sum-exp
- Logit clipping `[-50, 50]`
- Fallback de distribución uniforme
        """)
        st.divider()
        st.caption("Modelo: LSTM stacked — Dataset: Don Quijote de la Mancha")


if __name__ == "__main__":
    main()