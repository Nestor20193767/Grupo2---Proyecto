import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
import plotly.graph_objects as go
import os
import time

# ==========================================
# 1. CONFIGURACIÓN DEL DASHBOARD
# ==========================================
st.set_page_config(page_title="TCN-cVAE Advanced Demo", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #00ff00; }
    h1, h2, h3 { color: #ffffff; }
    .stAlert { background-color: #1e1e1e; color: #ffffff; }
    /* Estilizar sliders */
    .stSlider > div > div > div > div { background-color: #00ff00; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PARÁMETROS DEL MODELO Y ESPACIO LATENTE
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "Modelo", "tcncvae_decoder_physionet.onnx")

# Las 6 clases originales con las que se entrenó el generador
CLASES_ARRITMIA = ['AF', 'AFL', 'NSR', 'Others', 'SB', 'ST']
NUM_CLASES = len(CLASES_ARRITMIA)
LATENT_DIM = 32 

# =====================================================================
# ANCLAS LATENTES (Centroides Ideales calculados desde PhysioNet)
# =====================================================================
VECTORES_BASE = {
    'AF': np.array([[-0.45274, 0.12979, -0.36698, -0.20618, -0.33701, -0.49930, 0.28936, -0.30462, 0.61714, -0.03478, 0.44779, -0.01792, 0.27789, -0.12837, 0.13702, -0.06316, 0.03612, 0.24127, -0.02829, 0.22373, 0.12686, 0.04979, 0.02184, 0.07542, -0.08244, -0.11614, 0.29652, -0.04590, 0.24790, -0.05893, -0.06507, 0.06827]], dtype=np.float32),
    'AFL': np.array([[-0.46822, 0.01207, -0.24564, -0.29479, -0.35478, -0.56675, 0.17853, -0.34281, 0.68105, -0.09535, 0.65846, 0.00680, 0.21590, -0.24656, 0.18850, 0.03202, -0.10759, 0.26279, 0.02122, 0.33652, 0.21488, -0.16333, 0.08644, 0.01938, -0.01209, -0.21684, 0.51227, -0.03345, 0.18047, -0.01645, -0.07718, 0.05152]], dtype=np.float32),
    'NSR': np.array([[-0.35770, 0.17537, -0.19021, -0.15303, 0.18387, -0.55847, -0.25865, 0.24933, 0.32696, -0.02686, 0.40702, 0.06353, 0.60168, -0.11741, 0.26159, 0.08093, -0.27300, 0.49637, 0.03498, 0.18123, 0.38554, -0.33685, 0.36239, 0.15658, -0.25133, -0.28957, 0.65774, 0.26021, 0.48496, 0.34026, -0.46982, -0.19253]], dtype=np.float32),
    'Others': np.array([[-0.41515, 0.27205, -0.08812, -0.22254, 0.02018, -0.45404, 0.34956, -0.17729, 0.31346, -0.15250, 0.35173, -0.14402, 0.25643, 0.04308, 0.09929, -0.13133, -0.07570, 0.12475, 0.17536, -0.03283, 0.20698, -0.10650, 0.32385, 0.09128, -0.04681, 0.12780, 0.50049, 0.10278, 0.21503, 0.14744, -0.07260, -0.10959]], dtype=np.float32),
    'SB': np.array([[-0.32199, 0.02263, -0.47944, -0.04235, -0.12238, -0.26056, -0.20471, 0.03219, 0.14413, -0.00046, 0.35935, 0.06089, 0.15402, -0.28663, 0.07502, 0.06268, -0.04474, 0.40752, -0.12572, 0.07899, 0.04433, -0.17326, 0.20072, -0.13257, 0.08033, -0.04504, 0.26452, 0.16794, 0.04873, -0.18928, -0.39454, -0.10671]], dtype=np.float32),
    'ST': np.array([[-0.54659, 0.21795, 0.09014, -0.38076, 0.43375, -0.31646, -0.18444, 0.68856, 0.47541, 0.18764, 0.01922, -0.24525, 0.78680, 0.28961, 0.24222, 0.11643, -0.01504, 0.39154, 0.31510, 0.11472, 0.37185, -0.51854, 0.19191, 0.74124, -0.27744, -0.43856, 0.57710, 0.09196, 0.71150, 0.61575, -0.30115, -0.00706]], dtype=np.float32),
}

@st.cache_resource
def load_onnx_session(model_path):
    return ort.InferenceSession(model_path)

# ==========================================
# 3. INTERFAZ GRÁFICA (UI)
# ==========================================
st.title("🫀 Inferencia Generativa Avanzada: Exploración del Espacio Latente")

col_inputs, col_monitor = st.columns([1.2, 2.3])

with col_inputs:
    st.header("1. Panel de Control Latente")
    
    # ENTRADA C: Condición
    st.subheader("A. Condición Clínica ($c$)")
    clase_seleccionada = st.selectbox("Arritmia principal a sintetizar:", CLASES_ARRITMIA)
    clase_idx = CLASES_ARRITMIA.index(clase_seleccionada)
    
    st.markdown("---")
    
    # ENTRADA Z: Ruido Avanzado
    st.subheader(f"B. Vector de Ruido ($z$) - Dimensión Fija: {LATENT_DIM}")
    modo_ruido = st.selectbox(
        "Tipo de Exploración Matemática:", 
        [
            "Distribución Guiada (Recomendado)", 
            "Interpolación de Arritmias",
            "Distribución Normal Pura", 
            "Distribución Uniforme", 
            "Cargar CSV Personalizado"
        ]
    )
    
    z_input = None
    c_input = np.zeros((1, NUM_CLASES), dtype=np.float32)
    c_input[0, clase_idx] = 1.0 # Inicializar condición pura por defecto
    
    # --- LÓGICA DE CONTROL DINÁMICO ---
    if modo_ruido == "Distribución Guiada (Recomendado)":
        st.caption("Aplica variabilidad biológica sobre el latido ideal del banco latente.")
        variabilidad = st.slider("Variabilidad Fisiológica (Ruido)", min_value=0.0, max_value=2.0, value=0.15, step=0.05)
        z_base = VECTORES_BASE[clase_seleccionada]
        # Inyecta ruido controlado alrededor del centroide
        z_input = z_base + (np.random.randn(1, LATENT_DIM) * variabilidad)
        z_input = z_input.astype(np.float32)

    elif modo_ruido == "Interpolación de Arritmias":
        st.caption("Transición matemática continua entre dos morfologías.")
        clase_destino = st.selectbox("Arritmia de destino (Hacia dónde transicionar):", CLASES_ARRITMIA, index=(clase_idx + 1) % NUM_CLASES)
        alpha = st.slider("Nivel de Transición (α)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
        
        # Mezcla el espacio latente (Z)
        z_origen = VECTORES_BASE[clase_seleccionada]
        z_dest = VECTORES_BASE[clase_destino]
        z_input = ((1 - alpha) * z_origen + (alpha * z_dest)).astype(np.float32)
        
        # Mezcla el One-Hot Encoder Clínico (C)
        idx_dest = CLASES_ARRITMIA.index(clase_destino)
        c_input = np.zeros((1, NUM_CLASES), dtype=np.float32)
        c_input[0, clase_idx] = (1.0 - alpha)
        c_input[0, idx_dest] = alpha

    elif modo_ruido == "Distribución Normal Pura":
        st.caption("Fuerza un latido desde cero explorando a ciegas el espacio latente.")
        col1, col2 = st.columns(2)
        mu = col1.slider("Media (μ)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
        sigma = col2.slider("Desviación Std (σ)", min_value=0.1, max_value=5.0, value=0.5, step=0.1)
        z_input = (np.random.randn(1, LATENT_DIM) * sigma + mu).astype(np.float32)
        
    elif modo_ruido == "Distribución Uniforme":
        rango = st.slider("Rango [Min, Max]", min_value=-5.0, max_value=5.0, value=(-1.0, 1.0), step=0.5)
        z_input = np.random.uniform(low=rango[0], high=rango[1], size=(1, LATENT_DIM)).astype(np.float32)

    elif modo_ruido == "Cargar CSV Personalizado":
        archivo_z = st.file_uploader(f"Sube un CSV de 1x{LATENT_DIM}", type=['csv'])
        if archivo_z is not None:
            try:
                df_z = pd.read_csv(archivo_z, header=None)
                if df_z.shape == (1, LATENT_DIM):
                    z_input = df_z.values.astype(np.float32)
                    st.success("Ruido cargado correctamente.")
                else:
                    st.error(f"Error de forma. Esperado: (1, {LATENT_DIM})")
            except:
                st.error("Error al leer el archivo CSV.")
    
    # Fallback de seguridad si no hay input válido
    if z_input is None:
        z_input = VECTORES_BASE[clase_seleccionada]
    
    # Visualización del Vector Z
    st.markdown("**Inspección del vector $z$ a inyectar:**")
    fig_z = go.Figure(data=[go.Bar(
        y=z_input[0], 
        marker_color=['#ff3333' if val < 0 else '#33ccff' for val in z_input[0]]
    )])
    fig_z.update_layout(
        height=150, margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        xaxis=dict(visible=False), 
        yaxis=dict(range=[-5, 5], visible=False)
    )
    st.plotly_chart(fig_z, width="stretch")

    st.markdown("---")
    render_mode = st.radio("Opciones de Renderizado:", ["Generación Instantánea", "Animación en Vivo"], horizontal=True)
    btn_generar = st.button("⚡ SINTETIZAR SEÑAL ECG", width="stretch", type="primary")

with col_monitor:
    st.header("2. Monitor de Salida")
    monitor_placeholder = st.empty()
    
    if not btn_generar:
        fig_vacia = go.Figure()
        fig_vacia.update_layout(
            xaxis=dict(range=[0, 325], showgrid=False, visible=False),
            yaxis=dict(range=[-3, 3], showgrid=True, gridcolor='#1a1a1a', zeroline=True, zerolinecolor='#333333'),
            plot_bgcolor='black', paper_bgcolor='black', height=500, margin=dict(l=20, r=20, t=40, b=20)
        )
        monitor_placeholder.plotly_chart(fig_vacia, width="stretch")
        st.info("👈 Selecciona una arritmia, ajusta el espacio latente y presiona **Sintetizar Señal**.")

# ==========================================
# 4. LÓGICA DE INFERENCIA
# ==========================================
if btn_generar:
    session = load_onnx_session(ONNX_MODEL_PATH)
    
    if session is None:
        with col_monitor:
            st.error("❌ No se encontró el modelo ONNX. Verifica la ruta o el archivo .data")
    else:
        # Extraer nombres dinámicamente
        input_names = [inp.name for inp in session.get_inputs()]
        output_name = session.get_outputs()[0].name
        
        # Inferencia limpia con dimensiones correctas
        output = session.run([output_name], {input_names[0]: z_input, input_names[1]: c_input})[0]
        senal_generada = output.flatten()
        
        # Preparar título del gráfico
        titulo_monitor = f"MONITOR ECG - {clase_seleccionada} | {modo_ruido.split('(')[0]}"
        if modo_ruido == "Interpolación de Arritmias":
            titulo_monitor = f"MONITOR ECG - INTERPOLACIÓN ({clase_seleccionada} ➔ {clase_destino} | α={alpha})"
            
        # --- LÓGICA DE RENDERIZADO ---
        if render_mode == "Generación Instantánea":
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(senal_generada)), y=senal_generada, mode='lines', line=dict(color='#00ff00', width=2.5)))
            fig.update_layout(
                title=dict(text=titulo_monitor, font=dict(color='#00ff00', size=16)),
                xaxis=dict(range=[0, 325], showgrid=False, visible=False),
                yaxis=dict(range=[-3, 3], showgrid=True, gridcolor='#1a1a1a', zeroline=True, zerolinecolor='#004400'),
                plot_bgcolor='black', paper_bgcolor='black', height=500, margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            monitor_placeholder.plotly_chart(fig, width="stretch")
            
        else:
            # Animación por Chunks (Actualización de frame fluida)
            chunk_size = 25 
            for i in range(0, 325 + chunk_size, chunk_size):
                x_data = np.arange(i)
                y_data = senal_generada[:i]
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=x_data, y=y_data, mode='lines', 
                    line=dict(color='#00ff00', width=2.5)
                ))
                
                if i > 0 and i < 325:
                    fig.add_trace(go.Scatter(
                        x=[x_data[-1]], y=[y_data[-1]], mode='markers',
                        marker=dict(color='white', size=8, symbol='circle'), showlegend=False
                    ))
                
                fig.update_layout(
                    title=dict(text=titulo_monitor, font=dict(color='#00ff00', size=16)),
                    xaxis=dict(range=[0, 325], showgrid=False, visible=False),
                    yaxis=dict(range=[-3, 3], showgrid=True, gridcolor='#1a1a1a', zeroline=True, zerolinecolor='#004400'),
                    plot_bgcolor='black', paper_bgcolor='black', height=500, margin=dict(l=20, r=20, t=40, b=20),
                    showlegend=False
                )
                
                monitor_placeholder.plotly_chart(fig, width="stretch")
                time.sleep(0.005)
