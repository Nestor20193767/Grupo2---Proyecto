"""
ARRYS — TCN-cVAE · Frontend v3
Sistema académico de apoyo; no apto para diagnóstico clínico real.

Archivos requeridos en ./Modelo/:
    tcncvae_decoder_physionet.onnx
    clf_aug_physionet.onnx          (opcional)

Los centroides del espacio latente están embebidos (extraídos de ARRYS_7_0.ipynb).
"""

import os, time, io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import onnxruntime as ort

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTES
# ══════════════════════════════════════════════════════════════════════════════
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR    = os.path.join(BASE_DIR, "Modelo")
DECODER_PATH = os.path.join(MODEL_DIR, "tcncvae_decoder_physionet.onnx")
CLF_PATH     = os.path.join(MODEL_DIR, "clf_aug_physionet.onnx")

CLASS_NAMES = ['AF', 'AFL', 'NSR', 'Others', 'SB', 'ST']
NUM_CLASSES = len(CLASS_NAMES)
LATENT_DIM  = 32
SEQ_LEN     = 325
FS          = 500   # Hz

CLASS_INFO = {
    "AF":     {"desc": "Fibrilación Auricular",    "color": "#e84545", "risk": "Alto",
               "clinic": "Ausencia de onda P, intervalos RR irregulares. Alta prevalencia (2-3% adultos mayores)."},
    "AFL":    {"desc": "Flutter Auricular",        "color": "#f5a623", "risk": "Medio-Alto",
               "clinic": "Ondas en dientes de sierra a 300 bpm. Patrón morfológico muy característico."},
    "NSR":    {"desc": "Ritmo Sinusal Normal",     "color": "#00c9a7", "risk": "Normal",
               "clinic": "FC 60-100 bpm. Onda P precediendo cada QRS. Morfología PR, QRS y QT dentro del rango normal."},
    "Others": {"desc": "Otras arritmias",          "color": "#7c7c7c", "risk": "Variable",
               "clinic": "Grupo heterogéneo: NSIVCB, PAC, QTIE, STD. Morfología y riesgo variables."},
    "SB":     {"desc": "Bradicardia Sinusal",      "color": "#7ecfff", "risk": "Bajo-Medio",
               "clinic": "FC < 60 bpm. Morfología normal pero ritmo lento. Puede indicar disfunción sinusal."},
    "ST":     {"desc": "Taquicardia Sinusal",      "color": "#c084fc", "risk": "Bajo-Medio",
               "clinic": "FC > 100 bpm. Onda P normal, acortamiento del intervalo RR. Generalmente secundaria."},
}

# Centroides del espacio latente — extraídos de latent_bank[ci]['mu'].mean(dim=0) en ARRYS_7_0.ipynb
CENTROIDES = {
    'AF':np.array([[-0.45274,0.12979,-0.36698,-0.20618,-0.33701,-0.49930,0.28936,-0.30462,
                     0.61714,-0.03478,0.44779,-0.01792,0.27789,-0.12837,0.13702,-0.06316,
                     0.03612,0.24127,-0.02829,0.22373,0.12686,0.04979,0.02184,0.07542,
                    -0.08244,-0.11614,0.29652,-0.04590,0.24790,-0.05893,-0.06507,0.06827]],dtype=np.float32),
    'AFL':np.array([[-0.46822,0.01207,-0.24564,-0.29479,-0.35478,-0.56675,0.17853,-0.34281,
                      0.68105,-0.09535,0.65846,0.00680,0.21590,-0.24656,0.18850,0.03202,
                     -0.10759,0.26279,0.02122,0.33652,0.21488,-0.16333,0.08644,0.01938,
                     -0.01209,-0.21684,0.51227,-0.03345,0.18047,-0.01645,-0.07718,0.05152]],dtype=np.float32),
    'NSR':np.array([[-0.35770,0.17537,-0.19021,-0.15303,0.18387,-0.55847,-0.25865,0.24933,
                      0.32696,-0.02686,0.40702,0.06353,0.60168,-0.11741,0.26159,0.08093,
                     -0.27300,0.49637,0.03498,0.18123,0.38554,-0.33685,0.36239,0.15658,
                     -0.25133,-0.28957,0.65774,0.26021,0.48496,0.34026,-0.46982,-0.19253]],dtype=np.float32),
    'Others':np.array([[-0.41515,0.27205,-0.08812,-0.22254,0.02018,-0.45404,0.34956,-0.17729,
                         0.31346,-0.15250,0.35173,-0.14402,0.25643,0.04308,0.09929,-0.13133,
                        -0.07570,0.12475,0.17536,-0.03283,0.20698,-0.10650,0.32385,0.09128,
                        -0.04681,0.12780,0.50049,0.10278,0.21503,0.14744,-0.07260,-0.10959]],dtype=np.float32),
    'SB':np.array([[-0.32199,0.02263,-0.47944,-0.04235,-0.12238,-0.26056,-0.20471,0.03219,
                     0.14413,-0.00046,0.35935,0.06089,0.15402,-0.28663,0.07502,0.06268,
                    -0.04474,0.40752,-0.12572,0.07899,0.04433,-0.17326,0.20072,-0.13257,
                     0.08033,-0.04504,0.26452,0.16794,0.04873,-0.18928,-0.39454,-0.10671]],dtype=np.float32),
    'ST':np.array([[-0.54659,0.21795,0.09014,-0.38076,0.43375,-0.31646,-0.18444,0.68856,
                     0.47541,0.18764,0.01922,-0.24525,0.78680,0.28961,0.24222,0.11643,
                    -0.01504,0.39154,0.31510,0.11472,0.37185,-0.51854,0.19191,0.74124,
                    -0.27744,-0.43856,0.57710,0.09196,0.71150,0.61575,-0.30115,-0.00706]],dtype=np.float32),
}

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(page_title="ARRYS — TCN-cVAE",page_icon="🫀",
                   layout="wide",initial_sidebar_state="expanded")

st.markdown("""<style>
.stApp{background:#0a1628}
[data-testid="stSidebar"]{background:#0b1e36;border-right:1px solid #1a3a5c}
body,.stMarkdown,.stText,p,li{color:#dde8f5}
h1{color:#00c9a7!important;font-size:1.55rem!important}
h2{color:#fff!important;font-size:1.15rem!important}
h3{color:#7ecfff!important;font-size:.97rem!important}
.stSelectbox label,.stSlider label,.stNumberInput label,
.stRadio label,.stCheckbox label{color:#9ab5d0!important;font-size:.83rem}
.stButton>button[kind="primary"]{
    background:linear-gradient(135deg,#00c9a7,#028090);
    color:#071320;font-weight:700;border:none;border-radius:6px;
    padding:.55rem 1rem;font-size:.95rem;width:100%;transition:all .18s}
.stButton>button[kind="primary"]:hover{opacity:.82;transform:translateY(-1px)}
.stButton>button[kind="secondary"]{
    background:transparent;color:#7ecfff;
    border:1px solid #1a4a7a;border-radius:6px;width:100%}
[data-testid="stMetricValue"]{color:#00c9a7!important;font-size:1.3rem!important;font-weight:700}
[data-testid="stMetricLabel"]{color:#9ab5d0!important;font-size:.74rem!important}
[data-testid="stMetricDelta"]{font-size:.74rem!important}
.stAlert{border-radius:6px}
hr{border-color:#1a3a5c!important}
div[data-testid="stTabs"] button{color:#9ab5d0!important}
div[data-testid="stTabs"] button[aria-selected="true"]{color:#00c9a7!important;border-bottom-color:#00c9a7!important}
.warn-box{background:#1a0a0a;border:1.5px solid #e84545;border-radius:8px;
          padding:10px 14px;font-size:.82rem;color:#f5b5b5;line-height:1.5}
.info-pill{display:inline-block;padding:2px 9px;border-radius:10px;
           font-size:.73rem;font-weight:700;margin:2px}
</style>""",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODELOS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_sessions():
    opts=ort.SessionOptions()
    opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
    opts.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    dec=ort.InferenceSession(DECODER_PATH,sess_options=opts,
        providers=["CPUExecutionProvider"]) if os.path.exists(DECODER_PATH) else None
    clf=ort.InferenceSession(CLF_PATH,sess_options=opts,
        providers=["CPUExecutionProvider"]) if os.path.exists(CLF_PATH) else None
    return dec,clf

# ══════════════════════════════════════════════════════════════════════════════
# INFERENCIA
# ══════════════════════════════════════════════════════════════════════════════
def build_z(cls_name,n,noise,seed):
    rng=np.random.default_rng(seed)
    c=CENTROIDES[cls_name]
    return (c+rng.standard_normal((n,LATENT_DIM)).astype(np.float32)*noise)

def make_c(ci,n):
    c=np.zeros((n,NUM_CLASSES),dtype=np.float32);c[:,ci]=1.;return c

def decode(dec,z,ci):
    out=dec.run(None,{"z":z,"condition":make_c(ci,z.shape[0])})[0]
    return out[:,0,:]   # (N,325)

def run_clf(clf,beats):
    if clf is None:return None,None
    mu=beats.mean(1,keepdims=True);std=beats.std(1,keepdims=True)+1e-8
    x=((beats-mu)/std)[:,np.newaxis,:].astype(np.float32)
    logits=clf.run(None,{"signal":x})[0]
    e=np.exp(logits-logits.max(1,keepdims=True))
    probs=e/e.sum(1,keepdims=True)
    return probs,probs.argmax(1)

def classify_single(clf,beat):
    """Clasifica un solo latido 1D (SEQ_LEN,)."""
    if clf is None:return None,None
    b=beat.astype(np.float32)
    mu,std=b.mean(),b.std()+1e-8
    x=((b-mu)/std)[np.newaxis,np.newaxis,:].astype(np.float32)
    logits=clf.run(None,{"signal":x})[0]
    e=np.exp(logits-logits.max(1,keepdims=True))
    probs=(e/e.sum(1,keepdims=True))[0]
    return probs,int(probs.argmax())

# ══════════════════════════════════════════════════════════════════════════════
# VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════════════
_DL=dict(plot_bgcolor="#040d1a",paper_bgcolor="#0a1628",
         font=dict(color="#9ab5d0",family="monospace",size=11),
         margin=dict(l=42,r=18,t=44,b=32),
         xaxis=dict(gridcolor="#0c1f33",zerolinecolor="#1a3a5c"),
         yaxis=dict(gridcolor="#0c1f33",zerolinecolor="#1a3a5c"))

def _rgb(h):
    h=h.lstrip("#")
    return ",".join(str(int(h[i:i+2],16)) for i in (0,2,4))

def empty_monitor():
    fig=go.Figure()
    fig.update_layout(**_DL,height=340,
        xaxis=dict(range=[0,SEQ_LEN/FS*1000],**_DL["xaxis"],showticklabels=False),
        yaxis=dict(range=[-3.5,3.5],**_DL["yaxis"]),
        annotations=[dict(text="Pulsa ⚡ Generar ECG para ver la señal",
            xref="paper",yref="paper",x=.5,y=.5,
            font=dict(color="#2a4a6a",size=14),showarrow=False)])
    return fig

def static_fig(beats,cls_name):
    """Figura estática de todos los latidos + media + banda ±σ."""
    t=np.arange(SEQ_LEN)/FS*1000
    color=CLASS_INFO[cls_name]["color"]
    mean_b,std_b=beats.mean(0),beats.std(0)
    fig=go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([t,t[::-1]]),
        y=np.concatenate([mean_b+std_b,(mean_b-std_b)[::-1]]),
        fill="toself",fillcolor=f"rgba({_rgb(color)},.10)",
        line=dict(width=0),showlegend=False,hoverinfo="skip"))
    for i,b in enumerate(beats[:8]):
        fig.add_trace(go.Scatter(x=t,y=b,mode="lines",
            line=dict(color=color,width=1.1),opacity=.38,
            showlegend=(i==0),name="Latidos"))
    fig.add_trace(go.Scatter(x=t,y=mean_b,mode="lines",
        line=dict(color="white",width=2.5),name="Media"))
    fig.add_vline(x=125/FS*1000,line_dash="dot",line_color="#fff",opacity=.22,
                  annotation_text="R",annotation_font_color="#fff",annotation_font_size=10)
    fig.update_layout(**_DL,height=340,
        title=dict(text=f"ECG Sintético — {cls_name} · {CLASS_INFO[cls_name]['desc']}",
                   font=dict(color=color,size=13)),
        xaxis_title="Tiempo (ms)",yaxis_title="Amplitud (Z-score)",
        legend=dict(orientation="h",y=-.2,font=dict(size=10)))
    return fig

def animated_fig(beat,cls_name,step):
    """Figura parcial para animación trazo-a-trazo."""
    t=np.arange(SEQ_LEN)/FS*1000
    color=CLASS_INFO[cls_name]["color"]
    fig=go.Figure()
    # Segmento ya dibujado
    fig.add_trace(go.Scatter(x=t[:step],y=beat[:step],mode="lines",
        line=dict(color=color,width=2.2),showlegend=False))
    # Punto "cursor" en la punta
    if step>0 and step<SEQ_LEN:
        fig.add_trace(go.Scatter(x=[t[step-1]],y=[beat[step-1]],mode="markers",
            marker=dict(color="white",size=7,symbol="circle"),showlegend=False))
    fig.update_layout(**_DL,height=340,
        title=dict(text=f"⚡ Sintetizando — {cls_name} · {CLASS_INFO[cls_name]['desc']}",
                   font=dict(color=color,size=13)))
    fig.update_xaxes(range=[0,SEQ_LEN/FS*1000], title_text="Tiempo (ms)")
    fig.update_yaxes(range=[-3.5,3.5], title_text="Amplitud (Z-score)")
    return fig

def probs_fig(probs,target):
    mean_p=probs.mean(0) if probs.ndim==2 else probs
    colors=[CLASS_INFO[c]["color"] for c in CLASS_NAMES]
    border=["rgba(255,255,255,.9)" if c==target else "rgba(0,0,0,0)" for c in CLASS_NAMES]
    fig=go.Figure(go.Bar(x=CLASS_NAMES,y=mean_p*100,
        marker=dict(color=colors,line=dict(color=border,width=2)),
        text=[f"{p*100:.1f}%" for p in mean_p],
        textposition="outside",textfont=dict(size=10.5,color="#dde8f5")))
    fig.update_layout(**_DL,height=240,
        title=dict(text="Confianza diagnóstica del clasificador (%)",
                   font=dict(color="#7ecfff",size=12)),
        yaxis_title="%")
    fig.update_yaxes(range=[0,118])
    return fig

def compare_fig(beats_dict):
    t=np.arange(SEQ_LEN)/FS*1000
    fig=go.Figure()
    for cn,bts in beats_dict.items():
        fig.add_trace(go.Scatter(x=t,y=bts.mean(0),mode="lines",
            line=dict(color=CLASS_INFO[cn]["color"],width=2),
            name=f"{cn} — {CLASS_INFO[cn]['desc']}"))
    fig.update_layout(**_DL,height=320,
        title=dict(text="Morfología media por clase clínica",font=dict(color="#00c9a7",size=13)),
        xaxis_title="Tiempo (ms)",yaxis_title="Amplitud (Z-score)",
        legend=dict(orientation="h",y=-.28,font=dict(size=10)))
    return fig

def latent_fig(z,cls_name):
    zm=z.mean(0);centroid=CENTROIDES[cls_name][0]
    colors=["#e84545" if v<0 else "#7ecfff" for v in zm]
    fig=go.Figure()
    fig.add_trace(go.Bar(x=list(range(LATENT_DIM)),y=zm,marker_color=colors,
        name="z inyectado",
        hovertemplate="dim %{x}: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=list(range(LATENT_DIM)),y=centroid,mode="markers",
        marker=dict(color="white",size=5,symbol="diamond"),name="Centroide clase"))
    fig.update_layout(**_DL,height=210,
        title=dict(text="Vector z vs centroide de clase",font=dict(color="#7ecfff",size=11)),
        xaxis_title="Dimensión latente",yaxis_title="Valor",
        legend=dict(orientation="h",y=-.3,font=dict(size=10)))
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# ESTADO DE SESIÓN
# ══════════════════════════════════════════════════════════════════════════════
for k,v in [("beats",None),("z",None),("cls",None),("probs",None),
             ("preds",None),("multi",{}),("anim_done",False),
             ("real_beat",None),("real_probs",None),("real_pred",None)]:
    if k not in st.session_state: st.session_state[k]=v

# ══════════════════════════════════════════════════════════════════════════════
# CARGAR MODELOS
# ══════════════════════════════════════════════════════════════════════════════
with st.spinner("Cargando modelos ONNX…"):
    dec,clf=load_sessions()

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
c0,c1=st.columns([.07,.93])
c0.markdown("<p style='font-size:2.5rem;margin:0;padding-top:.2rem;'>🫀</p>",
            unsafe_allow_html=True)
c1.markdown(
    "<h1 style='margin:0;padding:0;'>ARRYS — Generador de Señales ECG Sintéticas Condicionadas</h1>"
    "<p style='color:#3a5a7a;font-size:.78rem;margin:.15rem 0 0 0;'>"
    "TCN-cVAE · PhysioNet 12-Lead 500 Hz · SNOMED-CT · Ingeniería Biomédica</p>",
    unsafe_allow_html=True)

# Advertencia obligatoria (plan Gemini §2 ítem 7)
st.markdown(
    "<div class='warn-box'>⚕️ <b>Sistema académico de apoyo; no apto para diagnóstico clínico real.</b> "
    "Las señales generadas son sintéticas, producidas por un modelo de IA no validado clínicamente. "
    "Ninguna salida de esta herramienta debe interpretarse como diagnóstico médico.</div>",
    unsafe_allow_html=True)
st.markdown("<hr style='margin:.5rem 0 .8rem 0;'>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown("### 📦 Modelos ONNX")
cs1,cs2=st.sidebar.columns(2)
cs1.metric("Decoder","✅" if dec else "❌")
cs2.metric("Clasif.","✅" if clf else "❌")

if dec is None:
    st.sidebar.error("Decoder no encontrado.\nColoca `tcncvae_decoder_physionet.onnx` en `./Modelo/`.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.markdown("## 🎛️ Parámetros del generador")

# ── 1. Clase clínica ──────────────────────────────────────────────────────────
st.sidebar.markdown("### 1. Clase clínica (condición c)")
cls_name=st.sidebar.selectbox("Arritmia objetivo:",CLASS_NAMES,
    format_func=lambda c:f"{c} — {CLASS_INFO[c]['desc']}")
cls_idx=CLASS_NAMES.index(cls_name)
info=CLASS_INFO[cls_name]
risk_c={"Alto":"#e84545","Medio-Alto":"#f5a623","Normal":"#00c9a7",
         "Bajo-Medio":"#7ecfff","Variable":"#7c7c7c"}[info["risk"]]
st.sidebar.markdown(
    f"<div style='background:#0d1f33;border-left:3px solid {info['color']};"
    f"border-radius:4px;padding:7px 10px;font-size:.80rem;'>"
    f"<b style='color:{info['color']};'>{cls_name}</b> · {info['desc']}<br>"
    f"<span style='color:{risk_c};font-size:.73rem;'>Riesgo clínico: {info['risk']}</span><br>"
    f"<span style='color:#5a7a9a;font-size:.71rem;'>{info['clinic'][:70]}…</span>"
    f"</div>",unsafe_allow_html=True)

# ── 2. Cantidad de latidos ────────────────────────────────────────────────────
st.sidebar.markdown("### 2. Cantidad de latidos")
n_beats=st.sidebar.slider("Latidos a generar:",1,32,8,1,
    help="Cada latido: 325 muestras · 650 ms · 500 Hz")

# ── 3. Parámetros del espacio latente z ──────────────────────────────────────
st.sidebar.markdown("### 3. Espacio latente z  (ruido)")
noise=st.sidebar.slider("Variabilidad morfológica (noise):",0.0,2.5,0.8,.05,
    help="0 = centroide puro · 0.8 = variabilidad natural · >1.5 = exploración extrema")

if noise<0.25:
    st.sidebar.info("🎯 **Prototipo** — señal canónica de la clase.",icon="🎯")
elif noise<1.1:
    st.sidebar.success("✅ **Natural** — variabilidad dentro de la distribución entrenada.",icon="✅")
else:
    st.sidebar.warning("⚡ **Exploración** — morfologías posiblemente atípicas.",icon="⚡")

# ── 4. Parámetros avanzados del modelo ───────────────────────────────────────
st.sidebar.markdown("### 4. Parámetros del modelo")
with st.sidebar.expander("⚙️ Opciones avanzadas",expanded=False):
    beta_info  = st.number_input("β-VAE (informativo):",value=0.035,step=.005,
        help="Solo visual — muestra el β con el que fue entrenado el modelo.",disabled=True)
    show_clf   = st.checkbox("Verificar con clasificador downstream",value=bool(clf),disabled=(clf is None))
    show_z     = st.checkbox("Mostrar vector z generado",value=False)
    seed_on    = st.checkbox("Semilla reproducible",value=False)
    seed_val   = st.number_input("Semilla:",0,9999,42,disabled=not seed_on)
    anim_speed = st.select_slider("Velocidad animación:",
        options=["Rápida","Normal","Lenta"],value="Normal")

CHUNK = {"Rápida":25,"Normal":14,"Lenta":6}[anim_speed]

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<small style='color:#2a4a6a;'>ARRYS v3 · TCN-cVAE · PhysioNet ECG-Arrhythmia<br>"
    "Centroides embebidos desde ARRYS_7_0.ipynb<br>"
    "⚠ Solo uso académico</small>",unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# BOTONES
# ══════════════════════════════════════════════════════════════════════════════
bc1,bc2,bc3=st.columns([1.1,1.1,1.8])
btn_gen=bc1.button("⚡ Generar ECG",type="primary",key="btn_gen")
btn_cmp=bc2.button("🔄 Comparar clases",type="secondary",key="btn_cmp")

# ══════════════════════════════════════════════════════════════════════════════
# LÓGICA DE GENERACIÓN  +  ANIMACIÓN
# ══════════════════════════════════════════════════════════════════════════════
if btn_gen:
    seed=int(seed_val) if seed_on else None
    z=build_z(cls_name,n_beats,noise,seed)
    beats=decode(dec,z,cls_idx)
    probs,preds=(run_clf(clf,beats) if show_clf and clf else (None,None))
    st.session_state.beats=beats; st.session_state.z=z
    st.session_state.cls=cls_name; st.session_state.probs=probs
    st.session_state.preds=preds; st.session_state.anim_done=False

if btn_cmp:
    seed=int(seed_val) if seed_on else None
    multi={}
    with st.spinner("Generando una muestra de cada clase…"):
        for ci_c,cls_c in enumerate(CLASS_NAMES):
            z_c=build_z(cls_c,8,0.9,seed)
            multi[cls_c]=decode(dec,z_c,ci_c)
    st.session_state.multi=multi

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_gen,tab_real,tab_cmp,tab_info=st.tabs([
    "📈 Generador ECG",
    "🔬 Clasificar muestra real",
    "📊 Comparación de clases",
    "ℹ️ Arquitectura & Dataset",
])

# ─── TAB 1: GENERADOR ─────────────────────────────────────────────────────────
with tab_gen:
    monitor=st.empty()

    if st.session_state.beats is None:
        monitor.plotly_chart(empty_monitor(),use_container_width=True)

    elif not st.session_state.anim_done:
        # ── ANIMACIÓN TRAZO-A-TRAZO ──────────────────────────────────────────
        beat_anim=st.session_state.beats[0]   # animar el primer latido
        color=CLASS_INFO[st.session_state.cls]["color"]

        for step in range(CHUNK,SEQ_LEN+CHUNK,CHUNK):
            step=min(step,SEQ_LEN)
            monitor.plotly_chart(
                animated_fig(beat_anim,st.session_state.cls,step),
                use_container_width=True)
            time.sleep(0.02)

        st.session_state.anim_done=True
        # Tras la animación, mostrar figura estática completa
        monitor.plotly_chart(
            static_fig(st.session_state.beats,st.session_state.cls),
            use_container_width=True)
    else:
        # Figura estática (ya animada)
        monitor.plotly_chart(
            static_fig(st.session_state.beats,st.session_state.cls),
            use_container_width=True)

    if st.session_state.beats is not None:
        beats=st.session_state.beats
        cls_u=st.session_state.cls
        probs=st.session_state.probs
        preds=st.session_state.preds
        z_u  =st.session_state.z
        n=beats.shape[0]
        color=CLASS_INFO[cls_u]["color"]

        # ── Métricas rápidas ─────────────────────────────────────────────────
        m1,m2,m3,m4,m5=st.columns(5)
        m1.metric("Latidos",n)
        m2.metric("Amp. pico media",f"{beats.max(1).mean():.3f}")
        m3.metric("Diversidad σ",f"{beats.std(0).mean():.4f}")
        m4.metric("Duración",f"{SEQ_LEN/FS*1000:.0f} ms")
        if probs is not None:
            ti=CLASS_NAMES.index(cls_u)
            m5.metric("Verificación clf",f"{(preds==ti).sum()}/{n} ✓",
                      delta=f"Conf. {probs[:,ti].mean()*100:.1f}%",
                      delta_color="normal" if probs[:,ti].mean()>.5 else "inverse")
        else:
            m5.metric("Clasificador","—")

        st.markdown("<hr style='margin:.4rem 0;'>",unsafe_allow_html=True)

        # ── Clasificador + Vector z ──────────────────────────────────────────
        ca,cb=st.columns([1.35,1])
        with ca:
            if probs is not None:
                st.plotly_chart(probs_fig(probs,cls_u),use_container_width=True)
                ti=CLASS_NAMES.index(cls_u)
                maj=int(np.bincount(preds).argmax())
                if maj!=ti:
                    st.warning(f"⚠ El clasificador predice mayoritariamente **{CLASS_NAMES[maj]}** "
                               f"en lugar de **{cls_u}**. Reduce el nivel de ruido.",icon="⚠️")
                else:
                    st.success(f"✅ {(preds==ti).sum()}/{n} latidos verificados como **{cls_u}**.",icon="✅")
            else:
                st.info("Activa *Verificar con clasificador* en las opciones avanzadas.",icon="ℹ️")
        with cb:
            if show_z and z_u is not None:
                st.plotly_chart(latent_fig(z_u,cls_u),use_container_width=True)
            else:
                st.markdown(
                    f"<div style='background:#0d1f33;border-left:4px solid {color};"
                    f"border-radius:6px;padding:16px 14px;'>"
                    f"<p style='color:{color};font-size:1rem;font-weight:700;margin:0;'>{cls_u}</p>"
                    f"<p style='color:#9ab5d0;font-size:.80rem;margin:4px 0 10px 0;'>"
                    f"{CLASS_INFO[cls_u]['desc']}</p>"
                    f"<p style='color:#3a5a7a;font-size:.73rem;line-height:1.8;'>"
                    f"SEQ_LEN = {SEQ_LEN} muestras · {SEQ_LEN/FS*1000:.0f} ms<br>"
                    f"LATENT_DIM = {LATENT_DIM} · FS = {FS} Hz<br>"
                    f"Noise = {noise:.2f} · Latidos = {n}<br>"
                    f"β-VAE = 0.035 · WeightNorm ✓<br>"
                    f"Dilataciones = (1,2,4,8,16,32)</p>"
                    f"</div>",unsafe_allow_html=True)

        # ── Descargas ────────────────────────────────────────────────────────
        st.markdown("<hr style='margin:.5rem 0;'>",unsafe_allow_html=True)
        dc1,dc2,dc3=st.columns(3)
        df_b=pd.DataFrame(beats,columns=[f"t{i}" for i in range(SEQ_LEN)])
        df_b.insert(0,"clase",cls_u)
        dc1.download_button("⬇ Señales (.csv)",
            df_b.to_csv(index=False).encode(),
            f"arrys_{cls_u}_{n}beats.csv","text/csv")
        if z_u is not None:
            df_z=pd.DataFrame(z_u,columns=[f"z{i}" for i in range(LATENT_DIM)])
            dc2.download_button("⬇ Vector z (.csv)",
                df_z.to_csv(index=False).encode(),
                f"arrys_z_{cls_u}.csv","text/csv")
        if probs is not None:
            df_c=pd.DataFrame(probs,columns=CLASS_NAMES)
            df_c.insert(0,"pred",[CLASS_NAMES[p] for p in preds])
            df_c.insert(0,"objetivo",cls_u)
            dc3.download_button("⬇ Predicciones clf (.csv)",
                df_c.to_csv(index=False).encode(),
                f"arrys_clf_{cls_u}.csv","text/csv")

# ─── TAB 2: CLASIFICAR MUESTRA REAL ──────────────────────────────────────────
with tab_real:
    st.markdown("### 🔬 Carga y clasificación de muestra real")
    st.markdown(
        "Sube un CSV con **una fila** y **325 columnas** (un latido de 650 ms a 500 Hz, "
        "Z-score normalizado), o genera una muestra de prueba con el botón inferior.",
        unsafe_allow_html=True)

    colA,colB=st.columns([1.5,1])
    with colA:
        uploaded=st.file_uploader(
            "Cargar señal ECG real (.csv — 1 × 325 muestras):",
            type=["csv"],key="upload_real")

        st.markdown("**— ó —**")
        demo_cls=st.selectbox("Generar latido sintético de prueba:",CLASS_NAMES,
            format_func=lambda c:f"{c} — {CLASS_INFO[c]['desc']}",key="demo_cls")
        btn_demo=st.button("🧪 Usar latido sintético como muestra",
            type="secondary",key="btn_demo")

    with colB:
        # Instrucciones de formato
        st.markdown(
            "<div style='background:#0d1f33;border-left:3px solid #7ecfff;"
            "border-radius:5px;padding:10px 12px;font-size:.79rem;'>"
            "<b style='color:#7ecfff;'>Formato esperado</b><br><br>"
            "• Archivo CSV sin encabezado (o con encabezado ignorado)<br>"
            "• 1 fila × 325 columnas<br>"
            "• Valores float32 (Z-score por latido)<br>"
            "• Frecuencia de muestreo: 500 Hz<br>"
            "• Ventana: 250 ms antes del pico R + 400 ms después<br><br>"
            "<b style='color:#f5a623;'>Si no tienes datos reales</b> usa el botón "
            "de muestra sintética para ver cómo funciona la clasificación."
            "</div>",unsafe_allow_html=True)

    # Carga desde CSV
    if uploaded is not None:
        try:
            df=pd.read_csv(uploaded,header=None)
            if df.shape[1]==SEQ_LEN:
                beat=df.values[0].astype(np.float32)
            elif df.shape[0]==SEQ_LEN:
                beat=df.values[:,0].astype(np.float32)
            else:
                st.error(f"Shape incorrecto: {df.shape}. Necesito 1×325 o 325×1.")
                beat=None
            if beat is not None:
                probs_r,pred_r=classify_single(clf,beat)
                st.session_state.real_beat=beat
                st.session_state.real_probs=probs_r
                st.session_state.real_pred=pred_r
                st.success("✅ Señal cargada y clasificada.",icon="✅")
        except Exception as e:
            st.error(f"Error al leer el CSV: {e}")

    # Generar latido de prueba
    if btn_demo:
        di=CLASS_NAMES.index(demo_cls)
        z_demo=build_z(demo_cls,1,0.6,None)
        beat_demo=decode(dec,z_demo,di)[0]
        probs_r,pred_r=classify_single(clf,beat_demo)
        st.session_state.real_beat=beat_demo
        st.session_state.real_probs=probs_r
        st.session_state.real_pred=pred_r

    # Mostrar resultado
    if st.session_state.real_beat is not None:
        beat_r=st.session_state.real_beat
        probs_r=st.session_state.real_probs
        pred_r=st.session_state.real_pred

        st.markdown("<hr style='margin:.6rem 0;'>",unsafe_allow_html=True)

        # Señal
        t=np.arange(SEQ_LEN)/FS*1000
        fig_r=go.Figure()
        fig_r.add_trace(go.Scatter(x=t,y=beat_r,mode="lines",
            line=dict(color="#00c9a7",width=2.2),name="Señal cargada"))
        fig_r.add_vline(x=125/FS*1000,line_dash="dot",line_color="#fff",
                        opacity=.22,annotation_text="R (esperado)",
                        annotation_font_color="#fff",annotation_font_size=9)
        fig_r.update_layout(**_DL,height=260,
            title=dict(text="Señal ECG cargada (visualización)",
                       font=dict(color="#00c9a7",size=13)),
            xaxis_title="Tiempo (ms)",yaxis_title="Amplitud (Z-score)")
        st.plotly_chart(fig_r,use_container_width=True)

        if probs_r is not None and pred_r is not None:
            cls_pred=CLASS_NAMES[pred_r]
            conf=float(probs_r[pred_r])
            color_pred=CLASS_INFO[cls_pred]["color"]

            # Resultado principal
            rc1,rc2,rc3=st.columns(3)
            rc1.metric("Clase predicha",cls_pred)
            rc2.metric("Confianza",f"{conf*100:.1f}%",
                delta="Alta" if conf>.70 else ("Media" if conf>.45 else "Baja"),
                delta_color="normal" if conf>.70 else ("off" if conf>.45 else "inverse"))
            rc3.metric("Descripción",CLASS_INFO[cls_pred]["desc"][:22])

            # Diagnóstico visual
            st.markdown(
                f"<div style='background:#0d1f33;border-left:5px solid {color_pred};"
                f"border-radius:6px;padding:12px 16px;margin:.5rem 0;'>"
                f"<b style='color:{color_pred};font-size:1.05rem;'>"
                f"Diagnóstico sugerido: {cls_pred} — {CLASS_INFO[cls_pred]['desc']}</b><br>"
                f"<span style='color:#9ab5d0;font-size:.82rem;'>"
                f"{CLASS_INFO[cls_pred]['clinic']}</span><br>"
                f"<span style='color:{color_pred};font-size:.78rem;font-weight:600;'>"
                f"Nivel de riesgo clínico: {CLASS_INFO[cls_pred]['risk']}</span>"
                f"</div>",unsafe_allow_html=True)

            # Barras de probabilidad
            st.plotly_chart(probs_fig(probs_r,cls_pred),use_container_width=True)

            # Advertencia
            st.markdown(
                "<div class='warn-box'>⚕️ <b>Sistema académico de apoyo; no apto para diagnóstico clínico real.</b> "
                "Esta predicción es generada por un modelo de IA con fines de investigación. "
                "No reemplaza la evaluación de un profesional médico calificado.</div>",
                unsafe_allow_html=True)
        else:
            st.warning("Clasificador no disponible. Sube `clf_aug_physionet.onnx` a `./Modelo/`.")

# ─── TAB 3: COMPARACIÓN DE CLASES ─────────────────────────────────────────────
with tab_cmp:
    if st.session_state.multi:
        st.plotly_chart(compare_fig(st.session_state.multi),use_container_width=True)
        st.markdown("### Detalle por clase")
        cols=st.columns(3)
        for i,(cn,bts) in enumerate(st.session_state.multi.items()):
            t=np.arange(SEQ_LEN)/FS*1000
            fm=go.Figure(go.Scatter(x=t,y=bts.mean(0),mode="lines",
                line=dict(color=CLASS_INFO[cn]["color"],width=2)))
            fm.update_layout(**_DL,height=170,showlegend=False,
                title=dict(text=f"{cn} · {CLASS_INFO[cn]['desc'][:20]}",
                           font=dict(color=CLASS_INFO[cn]["color"],size=10)),
                margin=dict(l=28,r=8,t=32,b=8),
                xaxis=dict(showticklabels=False,**_DL["xaxis"]))
            cols[i%3].plotly_chart(fm,use_container_width=True)
    else:
        st.markdown(
            "<div style='text-align:center;padding:60px 20px;color:#1a3a5a;'>"
            "<p style='font-size:2.6rem;'>📊</p>"
            "<p style='font-size:1rem;font-weight:600;'>Pulsa <b>🔄 Comparar clases</b></p>"
            "</div>",unsafe_allow_html=True)

# ─── TAB 4: ARQUITECTURA & DATASET ───────────────────────────────────────────
with tab_info:
    ti1,ti2=st.columns(2)

    with ti1:
        st.markdown("### 🏗️ Arquitectura TCN-cVAE")
        st.markdown(f"""
| Componente | Detalle |
|------------|---------|
| **Modelo** | TCN-cVAE — Temporal Convolutional Network |
| **Encoder** | Señal (1×325) + c (1×6) → μ,σ ∈ ℝ³² |
| **Decoder** | z (1×32) + c (1×6) → señal (1×1×325) |
| **LATENT_DIM** | `32` |
| **SEQ_LEN** | `325` muestras · 650 ms |
| **FS** | `500 Hz` |
| **Dilataciones** | `(1, 2, 4, 8, 16, 32)` |
| **Campo receptivo** | `505 muestras` > SEQ_LEN ✅ |
| **Normalización** | `WeightNorm` (Bai et al., 2018) |
| **β-VAE** | `β = 0.035`, warmup 40 épocas |
| **Clases** | `{NUM_CLASSES}` (SNOMED-CT) |
| **Pérdida** | MSE + β·KL + LabelSmooth ε=0.05 |
        """)

        st.markdown("### 🧠 Clasificador downstream")
        st.markdown("""
| Parámetro | Valor |
|-----------|-------|
| **Parámetros** | 235,925 |
| **Hidden CH** | 64 |
| **Dilataciones** | (1, 2, 4, 8, 16, 32) |
| **Pérdida** | FocalLoss (γ=2) |
| **Normalización** | normalize_per_sample |
| **Modo evaluación** | TSTR (Train-on-Synthetic-Test-on-Real) |
| **Mejor modo** | C — REAL+SINT (+4.48 pp recall macro) |
        """)

        st.markdown("### 📦 Archivos ONNX")
        for path,lbl in [(DECODER_PATH,"Decoder"),(CLF_PATH,"Clasificador")]:
            ex=os.path.exists(path)
            sz=f"{os.path.getsize(path)/1024:.0f} KB" if ex else "no encontrado"
            st.markdown(f"- {'✅' if ex else '❌'} **{lbl}**: `{os.path.basename(path)}` — {sz}")
        st.markdown("- ✅ **Centroides**: embebidos en `main.py` (sin archivo externo)")

    with ti2:
        st.markdown("### 📊 Estrategia del dataset")
        st.markdown("""
**PhysioNet ECG-Arrhythmia 1.0.0** · 12 derivaciones · 500 Hz · SNOMED-CT

| Split | % | Uso |
|-------|---|-----|
| GEN_SPLIT | 60% | Entrenamiento del TCN-cVAE |
| CLF_REAL  | 20% | Pool de datos reales del clasificador |
| CLF_HELD  | 20% | Test held-out (nunca visto) |

**Criterio de 7,000 latidos (subset):** Se restringió el pool del clasificador a ~1,400 latidos
reales por split para simular el escenario clínico de *data scarcity* que justifica el uso del
generador. Con datos ilimitados, el generador pierde valor demostrativo.

**Desbalance extremo:** AFL/SB dominan (>35% y >17.5% respectivamente). Ratio AFL:LVQRS = 435:1.
""")

        st.markdown("### 🩺 Clases SNOMED-CT")
        for cn,ci in CLASS_INFO.items():
            st.markdown(
                f"<div style='background:#0d1f33;border-left:4px solid {ci['color']};"
                f"border-radius:4px;padding:7px 12px;margin:4px 0;font-size:.80rem;'>"
                f"<b style='color:{ci['color']};'>{cn}</b> — {ci['desc']}"
                f" <span style='color:#2a4a6a;font-size:.71rem;'>· Riesgo: {ci['risk']}</span><br>"
                f"<span style='color:#3a5a7a;font-size:.70rem;'>{ci['clinic'][:75]}…</span>"
                f"</div>",unsafe_allow_html=True)

        st.markdown("### 🎚️ Guía de parámetros")
        st.markdown("""
| Noise | Efecto en la señal |
|-------|--------------------|
| `0.0` | Centroide puro — prototipo de la clase |
| `0.5–1.0` | Variabilidad natural del banco latente entrenado |
| `1.0–1.5` | Alta diversidad morfológica |
| `> 1.5` | Exploración fuera de distribución — morfologías atípicas |

Los centroides están extraídos del **banco latente real** de ARRYS_7_0.ipynb:
`latent_bank[ci]['mu'].mean(dim=0)` sobre el conjunto de entrenamiento completo.
Añadir ruido gaussiano alrededor del centroide es equivalente a la reparametrización
VAE sin necesidad del encoder en tiempo de inferencia.
        """)

    st.markdown("---")
    st.markdown(
        "<p style='text-align:center;color:#1a3a5a;font-size:.75rem;'>"
        "⚠️ <b>Sistema académico de apoyo; no apto para diagnóstico clínico real.</b> · "
        "Modelo: TCN-cVAE (ARRYS 7.0) · Dataset: PhysioNet ECG-Arrhythmia 1.0.0 · "
        "Frontend: Streamlit v3 · Centroides: banco latente ARRYS_7_0</p>",
        unsafe_allow_html=True)
