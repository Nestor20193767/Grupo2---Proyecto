"""
ARRYS — TCN-cVAE Frontend v2
Generación de señales ECG sintéticas condicionadas + clasificación downstream

Archivos necesarios en ./Modelo/:
    tcncvae_decoder_physionet.onnx
    clf_aug_physionet.onnx
    latent_bank.npz          ← exportado desde el notebook
    label_encoder_physionet.pkl
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import onnxruntime as ort

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ARRYS — Generador ECG Sintético",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* Fondo principal */
.stApp { background-color: #0a1628; }

/* Sidebar */
[data-testid="stSidebar"] { background-color: #0d2240; border-right: 1px solid #1a4a7a; }

/* Texto general */
body, .stMarkdown, .stText, p, li { color: #e0eaff; }

/* Títulos */
h1 { color: #00c9a7 !important; font-size: 1.6rem !important; }
h2 { color: #ffffff !important; font-size: 1.2rem !important; }
h3 { color: #7ecfff !important; font-size: 1rem !important; }

/* Selectbox, slider labels */
.stSelectbox label, .stSlider label, .stNumberInput label,
.stRadio label, .stCheckbox label { color: #a0b8d8 !important; font-size: 0.85rem; }

/* Botón primario */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00c9a7, #028090);
    color: #0a1628; font-weight: 700; border: none;
    border-radius: 6px; padding: 0.6rem 1.2rem; font-size: 1rem;
    width: 100%; transition: all 0.2s;
}
.stButton > button[kind="primary"]:hover { opacity: 0.85; transform: translateY(-1px); }

/* Botón secundario */
.stButton > button[kind="secondary"] {
    background: transparent; color: #7ecfff;
    border: 1px solid #1a4a7a; border-radius: 6px; width: 100%;
}

/* Métricas */
[data-testid="stMetricValue"] { color: #00c9a7 !important; font-size: 1.4rem !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #a0b8d8 !important; font-size: 0.78rem !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* Info / Warning / Success */
.stAlert { border-radius: 6px; }

/* Separadores */
hr { border-color: #1a4a7a; }

/* Tags de clase */
.class-tag {
    display: inline-block; padding: 2px 10px; border-radius: 12px;
    font-weight: 700; font-size: 0.8rem; margin: 2px;
}
/* Cards */
.metric-card {
    background: #112030; border: 1px solid #1a4a7a;
    border-radius: 8px; padding: 12px; text-align: center;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES DEL MODELO 
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "Modelo")

DECODER_PATH = os.path.join(MODEL_DIR, "tcncvae_decoder_physionet.onnx")
CLF_PATH     = os.path.join(MODEL_DIR, "clf_aug_physionet.onnx")
BANK_PATH    = os.path.join(MODEL_DIR, "latent_bank.npz")

CLASS_NAMES  = ['AF', 'AFL', 'NSR', 'Others', 'SB', 'ST']
NUM_CLASSES  = len(CLASS_NAMES)
LATENT_DIM   = 32
SEQ_LEN      = 325
FS           = 500   # Hz

# Descripción clínica de cada clase
CLASS_INFO = {
    "AF":     {"desc": "Fibrilación Auricular",      "color": "#e84545", "risk": "Alto"},
    "AFL":    {"desc": "Flutter Auricular",          "color": "#f5a623", "risk": "Medio-Alto"},
    "NSR":    {"desc": "Ritmo Sinusal Normal",       "color": "#00c9a7", "risk": "Normal"},
    "Others": {"desc": "Otras arritmias agrupadas",  "color": "#7c7c7c", "risk": "Variable"},
    "SB":     {"desc": "Bradicardia Sinusal",        "color": "#7ecfff", "risk": "Bajo-Medio"},
    "ST":     {"desc": "Taquicardia Sinusal",        "color": "#c084fc", "risk": "Bajo-Medio"},
}

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS (cacheada)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    decoder, clf, bank = None, None, None

    if os.path.exists(DECODER_PATH):
        decoder = ort.InferenceSession(DECODER_PATH, sess_options=opts,
                                        providers=["CPUExecutionProvider"])
    if os.path.exists(CLF_PATH):
        clf = ort.InferenceSession(CLF_PATH, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    if os.path.exists(BANK_PATH):
        raw  = np.load(BANK_PATH)
        bank = {}
        for ci in range(NUM_CLASSES):
            mu_key = f"mu_{ci}"
            lv_key = f"lv_{ci}"
            if mu_key in raw and lv_key in raw:
                bank[ci] = {"mu": raw[mu_key], "log_var": raw[lv_key]}
    return decoder, clf, bank


@st.cache_resource(show_spinner=False)
def _load_label_encoder():
    pkl_path = os.path.join(MODEL_DIR, "label_encoder_physionet.pkl")
    if os.path.exists(pkl_path):
        import joblib
        return joblib.load(pkl_path)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════
def sample_z_from_bank(bank, cls_idx: int, n: int, noise: float = 1.0) -> np.ndarray:
    """Muestrea z del banco latente real (no ruido gaussiano puro)."""
    if bank is None or cls_idx not in bank:
        return np.random.randn(n, LATENT_DIM).astype(np.float32)
    mu  = bank[cls_idx]["mu"]
    lv  = bank[cls_idx]["log_var"]
    idx = np.random.randint(0, len(mu), size=n)
    mu_s  = mu[idx]
    std_s = np.exp(0.5 * lv[idx])
    return (mu_s + noise * std_s * np.random.randn(*std_s.shape)).astype(np.float32)


def sample_z_gaussian(n: int, mu: float, sigma: float) -> np.ndarray:
    return (np.random.randn(n, LATENT_DIM) * sigma + mu).astype(np.float32)


def make_condition(cls_idx: int, n: int) -> np.ndarray:
    c = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    c[:, cls_idx] = 1.0
    return c


def normalize_beat(beat: np.ndarray) -> np.ndarray:
    mu  = beat.mean()
    std = beat.std() + 1e-8
    return (beat - mu) / std


def generate_beats(decoder, z: np.ndarray, cls_idx: int) -> np.ndarray:
    """Genera latidos. Devuelve (N, SEQ_LEN)."""
    n   = z.shape[0]
    c   = make_condition(cls_idx, n)
    out = decoder.run(None, {"z": z, "condition": c})[0]  # (N, 1, 325)
    return out[:, 0, :]


def classify_beats(clf, beats: np.ndarray) -> tuple:
    """Clasifica batch de latidos. Devuelve (probs NxC, preds N)."""
    if clf is None or beats is None:
        return None, None
    # Normalizar y añadir canal
    mu   = beats.mean(axis=1, keepdims=True)
    std  = beats.std(axis=1, keepdims=True) + 1e-8
    x    = ((beats - mu) / std)[:, np.newaxis, :].astype(np.float32)
    logits = clf.run(None, {"signal": x})[0]   # (N, C)
    e      = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs  = e / e.sum(axis=1, keepdims=True)
    preds  = probs.argmax(axis=1)
    return probs, preds


def compute_stats(beats: np.ndarray) -> dict:
    """Estadísticas rápidas sobre los latidos generados."""
    mean_beat = beats.mean(axis=0)
    std_beat  = beats.std(axis=0)
    return {
        "mean_amp":  float(beats.max(axis=1).mean()),
        "std_amp":   float(beats.max(axis=1).std()),
        "mean_noise": float(std_beat.mean()),
        "diversity":  float(np.mean([np.std(beats[:, i]) for i in range(SEQ_LEN)])),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
DARK_LAYOUT = dict(
    plot_bgcolor="#050d1a",
    paper_bgcolor="#0a1628",
    font=dict(color="#a0b8d8", family="monospace", size=11),
    xaxis=dict(gridcolor="#0d2240", zerolinecolor="#1a4a7a"),
    yaxis=dict(gridcolor="#0d2240", zerolinecolor="#1a4a7a"),
    margin=dict(l=40, r=20, t=40, b=30),
)


def plot_beats(beats: np.ndarray, cls_name: str, n_show: int = 8) -> go.Figure:
    t  = np.arange(SEQ_LEN) / FS * 1000   # ms
    color = CLASS_INFO[cls_name]["color"]
    n = min(n_show, len(beats))

    fig = go.Figure()
    # Media sombreada
    mean_b = beats.mean(axis=0)
    std_b  = beats.std(axis=0)
    fig.add_trace(go.Scatter(
        x=np.concatenate([t, t[::-1]]),
        y=np.concatenate([mean_b + std_b, (mean_b - std_b)[::-1]]),
        fill="toself", fillcolor=f"rgba({_hex_to_rgb(color)},0.12)",
        line=dict(width=0), showlegend=False, hoverinfo="skip"
    ))
    # Latidos individuales
    for i in range(n):
        fig.add_trace(go.Scatter(
            x=t, y=beats[i], mode="lines",
            line=dict(color=color, width=1.2),
            opacity=0.45, showlegend=(i == 0),
            name=f"Latido {i+1}",
        ))
    # Media
    fig.add_trace(go.Scatter(
        x=t, y=mean_b, mode="lines",
        line=dict(color="white", width=2.5, dash="solid"),
        name="Media", showlegend=True,
    ))

    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text=f"ECG Sintético — {cls_name} · {CLASS_INFO[cls_name]['desc']}",
                   font=dict(color=color, size=14)),
        xaxis_title="Tiempo (ms)",
        yaxis_title="Amplitud (Z-score)",
        height=360,
        legend=dict(orientation="h", y=-0.18, font=dict(size=10)),
    )
    # Marcadores en el pico R (muestra 125 = 250 ms)
    r_peak_t = 125 / FS * 1000
    fig.add_vline(x=r_peak_t, line_dash="dot", line_color="#ffffff", opacity=0.3,
                  annotation_text="R", annotation_font_color="#ffffff")
    return fig


def plot_multi_class(beats_dict: dict) -> go.Figure:
    """Compara la señal media de varias clases."""
    t = np.arange(SEQ_LEN) / FS * 1000
    fig = go.Figure()
    for cls_name, beats in beats_dict.items():
        color = CLASS_INFO[cls_name]["color"]
        fig.add_trace(go.Scatter(
            x=t, y=beats.mean(axis=0), mode="lines",
            line=dict(color=color, width=2),
            name=f"{cls_name} — {CLASS_INFO[cls_name]['desc']}",
        ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Comparación morfológica entre clases (señal media)",
                   font=dict(color="#00c9a7", size=13)),
        xaxis_title="Tiempo (ms)", yaxis_title="Amplitud (Z-score)",
        height=320,
        legend=dict(orientation="h", y=-0.25, font=dict(size=10)),
    )
    return fig


def plot_probs_bar(probs: np.ndarray, target_cls: str) -> go.Figure:
    """Gráfico de barras de probabilidades del clasificador."""
    mean_probs = probs.mean(axis=0)
    colors = [CLASS_INFO[c]["color"] for c in CLASS_NAMES]
    border = ["rgba(255,255,255,0.8)" if c == target_cls else "rgba(0,0,0,0)"
              for c in CLASS_NAMES]

    fig = go.Figure(go.Bar(
        x=CLASS_NAMES, y=mean_probs * 100,
        marker=dict(color=colors, line=dict(color=border, width=2)),
        text=[f"{p*100:.1f}%" for p in mean_probs],
        textposition="outside",
        textfont=dict(size=11, color="#e0eaff"),
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Verificación del clasificador (prob. media)", font=dict(color="#7ecfff", size=13)),
        yaxis_title="Confianza (%)", yaxis=dict(range=[0, 115], **DARK_LAYOUT["yaxis"]),
        height=260,
    )
    return fig


def plot_z_heatmap(z: np.ndarray) -> go.Figure:
    """Heatmap del espacio latente generado."""
    fig = go.Figure(go.Heatmap(
        z=z, colorscale="RdBu_r",
        zmid=0, showscale=True,
        colorbar=dict(tickfont=dict(color="#a0b8d8"), title=dict(text="z", font=dict(color="#a0b8d8"))),
        hovertemplate="dim=%{x}<br>latido=%{y}<br>z=%{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        **DARK_LAYOUT,
        title=dict(text="Mapa del espacio latente z inyectado", font=dict(color="#7ecfff", size=12)),
        xaxis_title="Dimensión latente", yaxis_title="Muestra",
        height=220,
    )
    return fig


def _hex_to_rgb(h: str) -> str:
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    st.markdown("<h1 style='font-size:2.5rem; margin:0; padding-top:0.3rem;'>🫀</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1 style='margin-bottom:0; padding-bottom:0;'>ARRYS — Generador de Señales ECG Sintéticas</h1>
    <p style='color:#5a7a9a; font-size:0.82rem; margin:0;'>
    TCN-cVAE condicionado por clase clínica SNOMED-CT · PhysioNet 12-Lead 500 Hz ·
    <span style='color:#e84545; font-weight:600;'>⚠ Uso académico exclusivo — No apto para diagnóstico clínico</span>
    </p>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:0.5rem 0 1rem 0;'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ══════════════════════════════════════════════════════════════════════════════
if "beats"      not in st.session_state: st.session_state.beats      = None
if "z_used"     not in st.session_state: st.session_state.z_used     = None
if "cls_used"   not in st.session_state: st.session_state.cls_used   = None
if "probs"      not in st.session_state: st.session_state.probs      = None
if "preds"      not in st.session_state: st.session_state.preds      = None
if "multi_beats"not in st.session_state: st.session_state.multi_beats= {}
if "tab"        not in st.session_state: st.session_state.tab        = "generate"

# ══════════════════════════════════════════════════════════════════════════════
# CARGAR MODELOS
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Cargando modelos ONNX…"):
    decoder, clf, bank = load_models()

# Estado de los modelos
st.sidebar.markdown("### 📦 Estado de los modelos")
col_s1, col_s2, col_s3 = st.sidebar.columns(3)
col_s1.metric("Decoder",     "✅" if decoder else "❌")
col_s2.metric("Clasif.",     "✅" if clf     else "❌")
col_s3.metric("Banco z",     "✅" if bank    else "❌")

if bank is None:
    st.sidebar.warning(
        "**Banco latente no encontrado.**\n\n"
        "Exporta el archivo latent_bank.npz desde tu entorno de entrenamiento y súbelo a la carpeta `Modelo/`.\n\n"
        "Sin el banco se usará N(0,I) — menos fiel a la distribución real.",
        icon="⚠️"
    )

if decoder is None:
    st.error(
        "**Decoder ONNX no encontrado.**\n\n"
        "Coloca `tcncvae_decoder_physionet.onnx` en la carpeta `Modelo/`.",
        icon="❌"
    )
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — CONTROLES
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Panel de Control")

# ── Selección de clase ────────────────────────────────────────────────────────
st.sidebar.markdown("### 1. Clase clínica")
cls_name = st.sidebar.selectbox(
    "Arritmia a sintetizar:",
    CLASS_NAMES,
    format_func=lambda c: f"{c} — {CLASS_INFO[c]['desc']}",
    key="cls_select",
)
cls_idx = CLASS_NAMES.index(cls_name)

info = CLASS_INFO[cls_name]
risk_color = {"Alto": "#e84545", "Medio-Alto": "#f5a623", "Normal": "#00c9a7",
              "Bajo-Medio": "#7ecfff", "Variable": "#7c7c7c"}[info["risk"]]
st.sidebar.markdown(
    f"<div style='background:#112030;border-left:3px solid {info['color']};"
    f"border-radius:4px;padding:8px 10px;font-size:0.82rem;'>"
    f"<b style='color:{info['color']};'>{cls_name}</b> · {info['desc']}<br>"
    f"<span style='color:{risk_color};font-size:0.75rem;'>Nivel de riesgo: {info['risk']}</span>"
    f"</div>",
    unsafe_allow_html=True
)

# ── Cantidad de latidos ───────────────────────────────────────────────────────
st.sidebar.markdown("### 2. Cantidad de latidos")
n_beats = st.sidebar.slider(
    "Latidos a generar:", min_value=1, max_value=32, value=8, step=1,
    help="Cada latido tiene 325 muestras (650 ms a 500 Hz)."
)

# ── Origen del vector z ───────────────────────────────────────────────────────
st.sidebar.markdown("### 3. Origen del vector z")
z_source = st.sidebar.radio(
    "Fuente del espacio latente:",
    ["Banco latente (recomendado)" if bank else "Banco latente (no disponible)", "Gaussiana N(μ,σ)", "CSV personalizado"],
    key="z_source_radio",
    disabled=False,
)
if "no disponible" in z_source:
    z_source = "Gaussiana N(μ,σ)"

noise_level = 1.0
if z_source == "Banco latente (recomendado)":
    st.sidebar.markdown("<small style='color:#5a7a9a;'>Muestrea z desde la distribución posterior del encoder entrenado, preservando la morfología real de la clase.</small>", unsafe_allow_html=True)
    noise_level = st.sidebar.slider(
        "Nivel de variabilidad (noise):", 0.1, 2.5, 1.0, 0.1,
        help="0.1 = latidos muy similares al prototipo · 2.0 = alta variabilidad"
    )

elif z_source == "Gaussiana N(μ,σ)":
    st.sidebar.markdown("<small style='color:#f5a623;'>Muestrea z desde una gaussiana arbitraria. Puede generar señales menos fieles a la clase.</small>", unsafe_allow_html=True)
    g_mu    = st.sidebar.slider("Media (μ):",          -3.0, 3.0,  0.0, 0.1)
    g_sigma = st.sidebar.slider("Desviación Std (σ):", 0.1,  5.0,  1.0, 0.1)

elif z_source == "CSV personalizado":
    st.sidebar.markdown(f"<small>Sube un CSV de forma `({n_beats}, {LATENT_DIM})`</small>", unsafe_allow_html=True)
    csv_file = st.sidebar.file_uploader(f"CSV ({n_beats}×{LATENT_DIM}):", type=["csv"])

# ── Opciones avanzadas ────────────────────────────────────────────────────────
st.sidebar.markdown("### 4. Opciones avanzadas")
show_clf    = st.sidebar.checkbox("Verificar con clasificador", value=True if clf else False, disabled=(clf is None))
show_latent = st.sidebar.checkbox("Mostrar mapa latente",       value=False)
seed_fixed  = st.sidebar.checkbox("Semilla fija (reproducible)", value=False)
if seed_fixed:
    seed_val = st.sidebar.number_input("Semilla:", value=42, min_value=0, max_value=9999)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#5a7a9a;'>ARRYS v2 · TCN-cVAE · PhysioNet ECG-Arrhythmia 12-Lead<br>"
    "⚠ Solo uso académico — No diagnóstico clínico</small>",
    unsafe_allow_html=True
)

# ══════════════════════════════════════════════════════════════════════════════
# BOTÓN GENERAR
# ══════════════════════════════════════════════════════════════════════════════
col_btn, col_cmp_btn, _ = st.columns([1, 1, 2])
btn_generate = col_btn.button("⚡ Generar ECG", type="primary", key="btn_gen")
btn_compare  = col_cmp_btn.button("🔄 Comparar todas las clases", type="secondary", key="btn_cmp")

# ══════════════════════════════════════════════════════════════════════════════
# LÓGICA DE GENERACIÓN
# ══════════════════════════════════════════════════════════════════════════════
if btn_generate:
    if seed_fixed:
        np.random.seed(seed_val)

    # Construir z según la fuente seleccionada
    z = None
    if z_source == "Banco latente (recomendado)":
        z = sample_z_from_bank(bank, cls_idx, n_beats, noise=noise_level)
    elif z_source == "Gaussiana N(μ,σ)":
        z = sample_z_gaussian(n_beats, g_mu, g_sigma)
    elif z_source == "CSV personalizado":
        if "csv_file" in dir() and csv_file is not None:
            try:
                df_z = pd.read_csv(csv_file, header=None)
                if df_z.shape == (n_beats, LATENT_DIM):
                    z = df_z.values.astype(np.float32)
                else:
                    st.error(f"Shape incorrecto. Esperado: ({n_beats}, {LATENT_DIM}), obtenido: {df_z.shape}")
            except Exception as e:
                st.error(f"Error al leer CSV: {e}")
        if z is None:
            z = sample_z_from_bank(bank, cls_idx, n_beats, noise=1.0)

    if z is None:
        z = sample_z_gaussian(n_beats, 0.0, 1.0)

    # Generar latidos
    with st.spinner(f"Generando {n_beats} latidos de {cls_name}…"):
        beats = generate_beats(decoder, z, cls_idx)

    # Clasificar
    probs, preds = None, None
    if show_clf and clf is not None:
        probs, preds = classify_beats(clf, beats)

    # Guardar en sesión
    st.session_state.beats   = beats
    st.session_state.z_used  = z
    st.session_state.cls_used= cls_name
    st.session_state.probs   = probs
    st.session_state.preds   = preds

# Comparación multi-clase
if btn_compare:
    if seed_fixed:
        np.random.seed(seed_val)
    multi = {}
    with st.spinner("Generando una muestra de cada clase…"):
        for ci_cmp, cls_cmp in enumerate(CLASS_NAMES):
            z_cmp = sample_z_from_bank(bank, ci_cmp, 8, noise=1.0) if bank else \
                    sample_z_gaussian(8, 0.0, 1.0)
            multi[cls_cmp] = generate_beats(decoder, z_cmp, ci_cmp)
    st.session_state.multi_beats = multi

# ══════════════════════════════════════════════════════════════════════════════
# ÁREA DE RESULTADOS
# ══════════════════════════════════════════════════════════════════════════════
tab_gen, tab_cmp, tab_info = st.tabs(["📈 Señal generada", "🔬 Comparación de clases", "ℹ️ Info del modelo"])

# ─── TAB 1: Señal generada ────────────────────────────────────────────────────
with tab_gen:
    if st.session_state.beats is not None:
        beats    = st.session_state.beats
        cls_used = st.session_state.cls_used
        z_used   = st.session_state.z_used
        probs    = st.session_state.probs
        preds    = st.session_state.preds
        color    = CLASS_INFO[cls_used]["color"]

        # ── Métricas rápidas ──────────────────────────────────────────────────
        stats = compute_stats(beats)
        n_gen = beats.shape[0]

        mc1, mc2, mc3, mc4, mc5 = st.columns(5)
        mc1.metric("Latidos generados",  n_gen)
        mc2.metric("Amplitud media",     f"{stats['mean_amp']:.3f}")
        mc3.metric("Variabilidad σ",      f"{stats['std_amp']:.3f}")
        mc4.metric("Diversidad intra",    f"{stats['diversity']:.4f}")

        if probs is not None:
            target_idx     = CLASS_NAMES.index(cls_used)
            mean_conf      = probs[:, target_idx].mean()
            n_correct      = (preds == target_idx).sum()
            mc5.metric(
                "Verificación clf",
                f"{n_correct}/{n_gen} ✓",
                delta=f"Conf. {mean_conf*100:.1f}%",
                delta_color="normal" if mean_conf > 0.5 else "inverse",
            )
        else:
            mc5.metric("Clasificador", "Desactivado")

        st.markdown("---")

        # ── Gráfico principal de latidos ───────────────────────────────────────
        n_show = min(8, n_gen)
        st.plotly_chart(plot_beats(beats, cls_used, n_show=n_show), width="stretch")

        # ── Columnas: clasificador + mapa latente ──────────────────────────────
        col_clf, col_lat = st.columns([1.2, 1])

        with col_clf:
            if probs is not None:
                st.plotly_chart(plot_probs_bar(probs, cls_used), width="stretch")

                # Alerta si la clase predicha mayoritaria difiere de la solicitada
                target_idx = CLASS_NAMES.index(cls_used)
                majority_pred = int(np.bincount(preds).argmax())
                if majority_pred != target_idx:
                    st.warning(
                        f"⚠ El clasificador predice mayoritariamente "
                        f"**{CLASS_NAMES[majority_pred]}** en lugar de **{cls_used}**. "
                        f"Considera aumentar el banco latente o reducir el nivel de ruido.",
                        icon="⚠️"
                    )
                else:
                    st.success(
                        f"✅ {(preds == target_idx).sum()}/{n_gen} latidos verificados "
                        f"como **{cls_used}** por el clasificador.",
                        icon="✅"
                    )
            else:
                st.info("Activa 'Verificar con clasificador' en el panel lateral para ver la verificación.")

        with col_lat:
            if show_latent and z_used is not None:
                st.plotly_chart(plot_z_heatmap(z_used), width="stretch")
            else:
                st.markdown(
                    f"<div style='background:#112030;border-radius:8px;padding:20px;"
                    f"border-left:3px solid {color};height:230px;display:flex;"
                    f"flex-direction:column;justify-content:center;'>"
                    f"<p style='color:{color};font-weight:700;font-size:1rem;margin:0;'>{cls_used}</p>"
                    f"<p style='color:#a0b8d8;font-size:0.82rem;margin:4px 0 0 0;'>{CLASS_INFO[cls_used]['desc']}</p>"
                    f"<p style='color:#5a7a9a;font-size:0.78rem;margin:8px 0 0 0;'>"
                    f"SEQ_LEN = {SEQ_LEN} muestras · {SEQ_LEN/FS*1000:.0f} ms<br>"
                    f"LATENT_DIM = {LATENT_DIM}<br>"
                    f"FS = {FS} Hz<br>"
                    f"Fuente z: {st.session_state.get('z_source_radio','—')[:25]}</p>"
                    f"</div>",
                    unsafe_allow_html=True
                )

        # ── Descarga de datos ─────────────────────────────────────────────────
        st.markdown("---")
        dl_col1, dl_col2, dl_col3 = st.columns(3)

        csv_beats = pd.DataFrame(
            beats,
            columns=[f"t_{i}" for i in range(SEQ_LEN)]
        )
        csv_beats.insert(0, "clase", cls_used)

        dl_col1.download_button(
            "⬇ Descargar señales (.csv)",
            data=csv_beats.to_csv(index=False).encode(),
            file_name=f"arrys_{cls_used}_{n_gen}beats.csv",
            mime="text/csv",
        )

        if z_used is not None:
            csv_z = pd.DataFrame(z_used, columns=[f"z{i}" for i in range(LATENT_DIM)])
            dl_col2.download_button(
                "⬇ Descargar vector z (.csv)",
                data=csv_z.to_csv(index=False).encode(),
                file_name=f"arrys_z_{cls_used}.csv",
                mime="text/csv",
            )

        if probs is not None:
            csv_clf = pd.DataFrame(probs, columns=CLASS_NAMES)
            csv_clf.insert(0, "prediccion", [CLASS_NAMES[p] for p in preds])
            csv_clf.insert(0, "clase_objetivo", cls_used)
            dl_col3.download_button(
                "⬇ Descargar predicciones clf (.csv)",
                data=csv_clf.to_csv(index=False).encode(),
                file_name=f"arrys_clf_{cls_used}.csv",
                mime="text/csv",
            )

    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#3a5a7a;'>
            <p style='font-size:3rem;'>🫀</p>
            <p style='font-size:1.1rem;font-weight:600;'>Ninguna señal generada aún</p>
            <p style='font-size:0.85rem;'>Selecciona una clase y haz clic en <b>⚡ Generar ECG</b></p>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 2: Comparación de clases ─────────────────────────────────────────────
with tab_cmp:
    if st.session_state.multi_beats:
        st.plotly_chart(plot_multi_class(st.session_state.multi_beats), width="stretch")

        # Grilla de señales individuales por clase
        st.markdown("### Señal media por clase")
        cols = st.columns(3)
        for i, (cn, bts) in enumerate(st.session_state.multi_beats.items()):
            col = cols[i % 3]
            t   = np.arange(SEQ_LEN) / FS * 1000
            fig_mini = go.Figure(go.Scatter(
                x=t, y=bts.mean(axis=0), mode="lines",
                line=dict(color=CLASS_INFO[cn]["color"], width=2),
            ))
            fig_mini.update_layout(
                **DARK_LAYOUT,
                title=dict(text=f"{cn} — {CLASS_INFO[cn]['desc'][:20]}",
                           font=dict(color=CLASS_INFO[cn]["color"], size=11)),
                height=180,
                xaxis=dict(showticklabels=False, **DARK_LAYOUT["xaxis"]),
                yaxis=dict(showticklabels=True, **DARK_LAYOUT["yaxis"]),
                margin=dict(l=30, r=10, t=35, b=10),
                showlegend=False,
            )
            col.plotly_chart(fig_mini, width="stretch")
    else:
        st.markdown("""
        <div style='text-align:center;padding:60px 20px;color:#3a5a7a;'>
            <p style='font-size:3rem;'>🔬</p>
            <p style='font-size:1.1rem;font-weight:600;'>Haz clic en <b>🔄 Comparar todas las clases</b></p>
        </div>
        """, unsafe_allow_html=True)

# ─── TAB 3: Info del modelo ───────────────────────────────────────────────────
with tab_info:
    col_i1, col_i2 = st.columns(2)

    with col_i1:
        st.markdown("### 🏗️ Arquitectura TCN-cVAE")
        st.markdown(f"""
| Parámetro | Valor |
|-----------|-------|
| Arquitectura | TCN-cVAE (Temporal Conv. Net) |
| LATENT_DIM | `{LATENT_DIM}` |
| SEQ_LEN | `{SEQ_LEN}` muestras (`650 ms`) |
| Frecuencia de muestreo | `{FS} Hz` |
| Dilataciones | `(1, 2, 4, 8, 16, 32)` |
| Campo receptivo | `505 muestras` > SEQ_LEN ✅ |
| Normalización | `WeightNorm` (Bai et al., 2018) |
| N° de clases | `{NUM_CLASSES}` (SNOMED-CT) |
| β-VAE | `β=0.035`, warmup 40 épocas |
        """)

        st.markdown("### 📦 Archivos ONNX")
        for path, label in [(DECODER_PATH, "Decoder"), (CLF_PATH, "Clasificador"), (BANK_PATH, "Banco latente")]:
            exists = os.path.exists(path)
            icon   = "✅" if exists else "❌"
            size   = f"{os.path.getsize(path)/1024:.0f} KB" if exists else "no encontrado"
            st.markdown(f"- {icon} **{label}**: `{os.path.basename(path)}` — {size}")

    with col_i2:
        st.markdown("### 🩺 Clases clínicas")
        for cn, ci in CLASS_INFO.items():
            st.markdown(
                f"<div style='background:#112030;border-left:4px solid {ci['color']};"
                f"border-radius:4px;padding:7px 12px;margin:4px 0;font-size:0.83rem;'>"
                f"<b style='color:{ci['color']};'>{cn}</b> — {ci['desc']}"
                f" &nbsp;<span style='color:#5a7a9a;font-size:0.75rem;'>Riesgo: {ci['risk']}</span>"
                f"</div>",
                unsafe_allow_html=True
            )

        st.markdown("### ℹ️ Guía de uso")
        st.markdown("""
**Banco latente (recomendado):**
Muestrea z desde la distribución aprendida por el encoder. Las señales preservan la morfología real de la clase. Usa `noise < 1.0` para señales más prototípicas y `noise > 1.0` para más variabilidad.

**Gaussiana N(μ,σ):**
Muestrea z desde una distribución libre. Útil para exploración del espacio latente, pero puede generar señales menos fieles.

**CSV personalizado:**
Inyecta un vector z específico. Requiere un CSV de forma `(N_beats, 32)`.

**Verificación con clasificador:**
Usa el modelo TCN entrenado en modo C (REAL+SINT) para verificar que los latidos generados son reconocibles como la clase objetivo.
        """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#3a5a7a;font-size:0.78rem;'>"
        "⚠️ <b>Advertencia:</b> Las señales generadas son sintéticas y de uso exclusivamente académico. "
        "No representan señales reales de pacientes y no deben utilizarse para diagnóstico clínico.<br>"
        "Dataset: PhysioNet ECG-Arrhythmia 1.0.0 · Modelo: TCN-cVAE (ARRYS 7.0) · "
        "Frontend: Streamlit v2</p>",
        unsafe_allow_html=True
    )
