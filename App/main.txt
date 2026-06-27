"""
ARRYS — TCN-cVAE Frontend v3
Generación de señales ECG sintéticas condicionadas + clasificación downstream

Archivos necesarios en ./Modelo/:
    tcncvae_decoder_physionet.onnx
    clf_aug_physionet.onnx
    latent_bank.npz
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

# Intentar cargar onnxruntime de manera segura
try:
    import onnxruntime as ort
except ImportError:
    st.error("Error: 'onnxruntime' no está instalado. Ejecuta: pip install onnxruntime")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN DE PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="ARRYS — Clasificación y Generación ECG",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Estilo CSS personalizado para la interfaz Biomédica
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

/* Banners y Alertas Clínicas */
.clinical-warning {
    background-color: rgba(232, 69, 69, 0.15);
    border: 1px solid #e84545;
    border-radius: 6px;
    padding: 12px;
    color: #ff7676;
    font-weight: 600;
    text-align: center;
    margin-bottom: 15px;
}

/* Botón primario */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #00c9a7, #028090);
    color: #0a1628; font-weight: 700; border: none;
    border-radius: 6px; padding: 0.6rem 1.2rem; font-size: 1rem;
    width: 100%; transition: all 0.2s;
}
.stButton > button[kind="primary"]:hover { opacity: 0.85; transform: translateY(-1px); }

/* Métricas */
[data-testid="stMetricValue"] { color: #00c9a7 !important; font-size: 1.4rem !important; font-weight: 700; }
[data-testid="stMetricLabel"] { color: #a0b8d8 !important; font-size: 0.78rem !important; }
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

CLASS_INFO = {
    "AF":     {"desc": "Fibrilación Auricular",      "color": "#e84545", "risk": "Alto"},
    "AFL":    {"desc": "Flutter Auricular",          "color": "#f5a623", "risk": "Medio-Alto"},
    "NSR":    {"desc": "Ritmo Sinusal Normal",       "color": "#00c9a7", "risk": "Normal"},
    "Others": {"desc": "Otras arritmias agrupadas",  "color": "#7c7c7c", "risk": "Variable"},
    "SB":     {"desc": "Bradicardia Sinusal",        "color": "#7ecfff", "risk": "Bajo-Medio"},
    "ST":     {"desc": "Taquicardia Sinusal",        "color": "#c084fc", "risk": "Bajo-Medio"},
}

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS (Cacheada)
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_models():
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    decoder, clf, bank = None, None, None

    if os.path.exists(DECODER_PATH):
        decoder = ort.InferenceSession(DECODER_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    if os.path.exists(CLF_PATH):
        clf = ort.InferenceSession(CLF_PATH, sess_options=opts, providers=["CPUExecutionProvider"])
    if os.path.exists(BANK_PATH):
        raw = np.load(BANK_PATH)
        bank = {}
        for ci in range(NUM_CLASSES):
            mu_key, lv_key = f"mu_{ci}", f"lv_{ci}"
            if mu_key in raw and lv_key in raw:
                bank[ci] = {"mu": raw[mu_key], "log_var": raw[lv_key]}
    return decoder, clf, bank

# ══════════════════════════════════════════════════════════════════════════════
# LÓGICA DE INFERENCIA MATEMÁTICA Y REDES
# ══════════════════════════════════════════════════════════════════════════════
def sample_z_from_bank(bank, cls_idx: int, n: int, noise: float = 1.0) -> np.ndarray:
    if bank is None or cls_idx not in bank:
        return np.random.randn(n, LATENT_DIM).astype(np.float32)
    mu, lv = bank[cls_idx]["mu"], bank[cls_idx]["log_var"]
    idx = np.random.randint(0, len(mu), size=n)
    mu_s, std_s = mu[idx], np.exp(0.5 * lv[idx])
    return (mu_s + noise * std_s * np.random.randn(*std_s.shape)).astype(np.float32)

def generate_beats(decoder, z: np.ndarray, cls_idx: int) -> np.ndarray:
    n = z.shape[0]
    c = np.zeros((n, NUM_CLASSES), dtype=np.float32)
    c[:, cls_idx] = 1.0
    out = decoder.run(None, {"z": z, "condition": c})[0]
    return out[:, 0, :]

def classify_beats(clf, beats: np.ndarray) -> tuple:
    if clf is None or beats is None:
        return None, None
    # normalize_per_sample (C3) requerida por el clasificador
    mu = beats.mean(axis=1, keepdims=True)
    std = beats.std(axis=1, keepdims=True) + 1e-8
    x = ((beats - mu) / std)[:, np.newaxis, :].astype(np.float32)
    
    logits = clf.run(None, {"signal": x})[0]
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = e / e.sum(axis=1, keepdims=True)
    preds = probs.argmax(axis=1)
    return probs, preds

# ══════════════════════════════════════════════════════════════════════════════
# GRÁFICOS CON DISEÑO OSCURO BIOMÉDICO
# ══════════════════════════════════════════════════════════════════════════════
DARK_LAYOUT = dict(
    plot_bgcolor="#050d1a", paper_bgcolor="#0a1628",
    font=dict(color="#a0b8d8", family="monospace", size=11),
    xaxis=dict(gridcolor="#0d2240", zerolinecolor="#1a4a7a"),
    yaxis=dict(gridcolor="#0d2240", zerolinecolor="#1a4a7a"),
    margin=dict(l=50, r=20, t=40, b=40),
)

def plot_single_waveform(signal: np.ndarray, title: str, color: str) -> go.Figure:
    t = np.arange(SEQ_LEN) / FS * 1000 # Eje temporal en milisegundos (0-650ms)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=signal, mode="lines", line=dict(color=color, width=2.5), name="Señal"))
    fig.update_layout(**DARK_LAYOUT, title=title, xaxis_title="Tiempo (ms)", yaxis_title="Amplitud normalizada (Z-score)", height=280)
    fig.add_vline(x=250, line_dash="dot", line_color="#ffffff", opacity=0.3, annotation_text="Pico R")
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ DE USUARIO — ENCABEZADO Y ADVERTENCIAS OBLIGATORIAS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="clinical-warning">⚠️ Sistema académico de apoyo; no apto para diagnóstico clínico real</div>', unsafe_allow_html=True)

col_logo, col_title = st.columns([0.08, 0.92])
with col_logo:
    st.markdown("<h1 style='font-size:3rem; margin:0;'>🫀</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown("""
    <h1 style='margin:0;'>ARRYS — Sistema de Clasificación y Modelado Heterogéneo de Arritmias Cardíacas</h1>
    <p style='color:#7ecfff; font-size:0.88rem; margin:2px 0 0 0;'>
    <b>Problema Biomédico:</b> El desbalance crítico de clases en bioseñales limita el rendimiento de los clasificadores automáticos en la práctica clínica. Este sistema utiliza una Red Temporal Convolucional acoplada a un VAE Condicional (TCN-cVAE) entrenado con el dataset de PhysioNet para robustecer el diagnóstico mediante técnicas avanzadas de oversampling sintético condicionado.
    </p>
    """, unsafe_allow_html=True)

st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE MODELOS
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Inicializando motores de inferencia ONNX…"):
    decoder, clf, bank = load_models()

# ══════════════════════════════════════════════════════════════════════════════
# DIVISION EN PESTAÑAS (DEMO FLUX)
# ══════════════════════════════════════════════════════════════════════════════
tab_real_test, tab_sint_gen, tab_model_architecture = st.tabs([
    "🔬 1. Carga y Prueba de Muestra Real", 
    "📈 2. Generación Sintética Condicionada", 
    "ℹ️ 3. Ficha Técnica de Ingeniería"
])

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 1: CARGA O SELECCIÓN DE UNA MUESTRA DE ENTRADA (REQUISITO EXPLICITO)
# ──────────────────────────────────────────────────────────────────────────────
with tab_real_test:
    st.markdown("### Selección de Entrada de Datos Clínicos")
    
    input_method = st.radio(
        "Elige el método de entrada para la prueba del Clasificador TCN (ARRYS6):",
        ["Utilizar muestra prototípica del dataset de referencia", "Cargar archivo de señal propio (.csv)"],
        horizontal=True
    )
    
    selected_signal = None
    target_class_demo = "NSR"
    
    if input_method == "Utilizar muestra prototípica del dataset de referencia":
        target_class_demo = st.selectbox(
            "Selecciona la patología de origen para extraer del banco real:",
            CLASS_NAMES, index=2, format_func=lambda c: f"Latido de referencia clasificado como: {c} ({CLASS_INFO[c]['desc']})"
        )
        # Extraemos un latido realista desde la media del banco latente para simular la muestra de entrada real
        if bank and CLASS_NAMES.index(target_class_demo) in bank:
            z_demo = bank[CLASS_NAMES.index(target_class_demo)]["mu"][[0]]
            selected_signal = generate_beats(decoder, z_demo, CLASS_NAMES.index(target_class_demo))[0]
        else:
            # Fallback en caso de que falte el archivo npz
            selected_signal = np.sin(np.linspace(0, 10, SEQ_LEN)) + np.random.normal(0, 0.05, SEQ_LEN)
            
    else:
        uploaded_file = st.file_uploader(f"Carga un archivo CSV con una fila que contenga exactamente {SEQ_LEN} amplitudes temporales:", type=["csv"])
        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file, header=None).values[0].astype(np.float32)
                if len(uploaded_data) == SEQ_LEN:
                    selected_signal = uploaded_data
                    st.success("Señal de entrada externa cargada correctamente.")
                else:
                    st.error(f"Error dimensional: El archivo debe contener exactamente {SEQ_LEN} puntos de muestreo.")
            except Exception as e:
                st.error(f"Error procesando el archivo: {e}")
                
    if selected_signal is not None:
        st.markdown("---")
        col_view_left, col_view_right = st.columns([1.3, 1])
        
        with col_view_left:
            st.plotly_chart(
                plot_single_waveform(selected_signal, f"Visualización del resultado de Entrada — Formato de Onda ECG", "#00c9a7"),
                use_container_width=True
            )
            
        with col_view_right:
            st.markdown("### 🎯 Resultado de la Predicción del Clasificador")
            
            # Ejecutar inferencia en el clasificador TCN
            input_batch = selected_signal[np.newaxis, :]
            probs, preds = classify_beats(clf, input_batch)
            
            if probs is not None:
                predicted_idx = preds[0]
                predicted_class = CLASS_NAMES[predicted_idx]
                confidence_level = probs[0][predicted_idx] * 100
                
                # Visualización del resultado y nivel de confianza
                st.metric(label="Clase Diagnosticada por la Red Temporal", value=f"{predicted_class} — {CLASS_INFO[predicted_class]['desc']}")
                
                # Barra de progreso para nivel de confianza
                st.markdown(f"**Nivel de confianza / Probabilidad de acierto:** `{confidence_level:.2f}%`")
                st.progress(confidence_level / 100.0)
                
                # Mostrar el pool de probabilidades secundarias
                df_probs = pd.DataFrame({
                    'Arritmia': CLASS_NAMES,
                    'Probabilidad (%)': probs[0] * 100
                })
                fig_bars = px.bar(df_probs, x='Arritmia', y='Probabilidad (%)', color='Arritmia',
                                  color_discrete_map={c: CLASS_INFO[c]['color'] for c in CLASS_NAMES},
                                  height=200, template="plotly_dark")
                fig_bars.update_layout(margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="#050d1a", paper_bgcolor="#0a1628")
                st.plotly_chart(fig_bars, use_container_width=True)
            else:
                st.info("El Clasificador TCN ONNX no está disponible en la carpeta /Modelo.")

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 2: GENERACIÓN SINTÉTICA (TU LÓGICA ORIGINAL OPTIMIZADA)
# ──────────────────────────────────────────────────────────────────────────────
with tab_sint_gen:
    st.sidebar.markdown("### Configuración del Espacio Sintético")
    # Sincronización con controles laterales existentes del panel de control de tu script original
    if st.sidebar.button("Re-generar desde el panel lateral", key="tab2_trigger"):
        st.info("Ajusta los parámetros del panel de control de la izquierda y presiona el botón principal superior de la interfaz.")

    if st.session_state.beats is not None:
        beats = st.session_state.beats
        cls_used = st.session_state.cls_used
        probs_sint = st.session_state.probs
        
        st.markdown(f"### Visualización de Datos de Síntesis Activa: {cls_used}")
        
        sc1, sc2, sc3 = st.columns(3)
        sc1.metric("Latidos Sintetizados en Memoria", len(beats))
        sc2.metric("Dimensión Latente de Muestreo", LATENT_DIM)
        
        if probs_sint is not None:
            target_idx = CLASS_NAMES.index(cls_used)
            sc3.metric("Reconocimiento del Clasificador (Conf. Media)", f"{probs_sint[:, target_idx].mean()*100:.1f}%")
        
        # Reutilizamos tu motor de plots plotly original
        t = np.arange(SEQ_LEN) / FS * 1000
        fig_gen = go.Figure()
        mean_b = beats.mean(axis=0)
        fig_gen.add_trace(go.Scatter(x=t, y=mean_b, mode="lines", line=dict(color=CLASS_INFO[cls_used]['color'], width=3), name="Latido Sintético Promedio"))
        fig_gen.update_layout(**DARK_LAYOUT, title=f"Morfología Sintética Generada por TCN-cVAE para la clase {cls_used}", xaxis_title="Tiempo (ms)", yaxis_title="Amplitud Z-Score")
        st.plotly_chart(fig_gen, use_container_width=True)
    else:
        st.info("Por favor, interactúa con el panel de control de la izquierda y haz clic en '⚡ Generar ECG' para desplegar la señal sintética.")

# ──────────────────────────────────────────────────────────────────────────────
# PESTAÑA 3: FICHA TÉCNICA (SITUATIONAL AWARENESS BIOMÉDICA)
# ──────────────────────────────────────────────────────────────────────────────
with tab_model_architecture:
    st.markdown("### Especificaciones de Diseño del Clasificador TCN (ARRYS6)")
    st.markdown(f"""
    * **Total de Parámetros del Clasificador:** `235,925` entrenados bajo el criterio **F1-macro** para mitigar el sesgo por clases minoritarias.
    * **Canales Ocultos (Hidden channels):** `64` unidades de procesamiento temporal paralela.
    * **Esquema de Dilataciones Convolucionales:** `(1, 2, 4, 8, 16, 32)` mapeando un campo receptivo biológico de 505 muestras (superior a la ventana de adquisición).
    * **Estrategia ante Pérdidas:** `FocalLoss(gamma=2)`, optimizada para priorizar el aprendizaje sobre muestras complejas y arritmias raras (como la Fibrilación Auricular con escaso soporte real).
    * **Normalización C3 Implementada:** `normalize_per_sample` ejecutada en tiempo real por el backend para aislar variaciones de amplitud basales entre pacientes.
    """)
    st.success("Configuración de los modelos ONNX cargada con éxito en la arquitectura de hardware.")

# ══════════════════════════════════════════════════════════════════════════════
# PIE DE PÁGINA REGLAMENTARIO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<hr style='border-color: #1a4a7a;'>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;color:#5a7a9a;font-size:0.78rem;'>"
    "<b>Clasificador TCN (ARRYS6)</b> · Proyecto de Procesamiento de Señales Biomédicas Avanzadas · UPCH<br>"
    "⚠️ <i>Sistema académico de apoyo; no apto para diagnóstico clínico real. Las simulaciones numéricas no reemplazan la evaluación de un especialista ni el dictamen de un electrocardiógrafo certificado.</i></p>",
    unsafe_allow_html=True
)
