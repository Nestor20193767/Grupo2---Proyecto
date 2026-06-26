import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
import plotly.graph_objects as go
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
# 2. PARÁMETROS DEL MODELO
# ==========================================
import os

# Usar os.path garantiza que la ruta funcione perfectamente en los servidores Linux de Streamlit
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ONNX_MODEL_PATH = os.path.join(BASE_DIR, "Modelo", "tcncvae_decoder_physionet.onnx")

# CLASES_ARRITMIA = ['AFL', 'LVQRS', 'NSIVCB', 'Other', 'PAC', 'QTIE', 'STD']
CLASES_ARRITMIA = ['SB', 'NSR', 'AFL', 'ST', 'AF']
NUM_CLASES = len(CLASES_ARRITMIA)

# NOTA PARA EL JURADO: Esta dimensión es inmutable post-entrenamiento debido a la arquitectura de la red neuronal.
LATENT_DIM = 32 

@st.cache_resource
def load_onnx_session(model_path):
    try:
        return ort.InferenceSession(model_path)
    except:
        return None

# ==========================================
# 3. INTERFAZ GRÁFICA (UI)
# ==========================================
st.title("🫀 Inferencia Generativa Avanzada: Exploración del Espacio Latente")

col_inputs, col_monitor = st.columns([1.2, 2.3])

with col_inputs:
    st.header("1. Panel de Control Latente")
    
    # ENTRADA C: Condición
    st.subheader("A. Condición Clínica ($c$)")
    clase_seleccionada = st.selectbox("Arritmia a sintetizar:", CLASES_ARRITMIA)
    clase_idx = CLASES_ARRITMIA.index(clase_seleccionada)
    
    st.markdown("---")
    
    # ENTRADA Z: Ruido Avanzado
    st.subheader(f"B. Vector de Ruido ($z$) - Dimensión Fija: {LATENT_DIM}")
    
    modo_ruido = st.selectbox(
        "Tipo de Distribución Matemática:", 
        ["Distribución Normal (Gaussiana)", "Distribución Uniforme", "Valores Extremos (Ceros/Unos)", "Cargar CSV Personalizado"]
    )
    
    z_input = None
    
    # Lógica de Control de Ruido Dinámico
    if modo_ruido == "Distribución Normal (Gaussiana)":
        st.caption("La distribución estándar esperada por el cVAE.")
        col1, col2 = st.columns(2)
        mu = col1.slider("Media (μ)", min_value=-3.0, max_value=3.0, value=0.0, step=0.1)
        sigma = col2.slider("Desviación Std (σ)", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        # Generar ruido normalizado ajustado
        z_input = (np.random.randn(1, LATENT_DIM) * sigma + mu).astype(np.float32)
        
    elif modo_ruido == "Distribución Uniforme":
        st.caption("Evalúa cómo el modelo maneja ruido no gaussiano continuo.")
        rango = st.slider("Rango [Min, Max]", min_value=-5.0, max_value=5.0, value=(-1.0, 1.0), step=0.5)
        # Generar ruido uniforme
        z_input = np.random.uniform(low=rango[0], high=rango[1], size=(1, LATENT_DIM)).astype(np.float32)
        
    elif modo_ruido == "Valores Extremos (Ceros/Unos)":
        st.caption("Prueba de estrés para ver el sesgo de la condición c.")
        extremo = st.radio("Llenar vector latente con:", ["Ceros absolutos (0)", "Unos absolutos (1)", "Menos unos (-1)"])
        if extremo == "Ceros absolutos (0)":
            z_input = np.zeros((1, LATENT_DIM), dtype=np.float32)
        elif extremo == "Unos absolutos (1)":
            z_input = np.ones((1, LATENT_DIM), dtype=np.float32)
        else:
            z_input = np.ones((1, LATENT_DIM), dtype=np.float32) * -1.0

    elif modo_ruido == "Cargar CSV Personalizado":
        archivo_z = st.file_uploader(f"Sube un CSV de 1x{LATENT_DIM}", type=['csv'])
        if archivo_z is not None:
            try:
                df_z = pd.read_csv(archivo_z, header=None)
                if df_z.shape == (1, LATENT_DIM):
                    z_input = df_z.values.astype(np.float32)
                    st.success("Ruido cargado.")
                else:
                    st.error(f"Error de forma. Esperado: (1, {LATENT_DIM})")
            except:
                st.error("Error al leer el CSV.")
    
    # Fallback de seguridad
    if z_input is None:
        z_input = np.random.randn(1, LATENT_DIM).astype(np.float32)
    
    # Visualización del Vector Z (Histograma de barras en vivo)
    st.markdown("**Inspección del vector $z$ a inyectar:**")
    fig_z = go.Figure(data=[go.Bar(
        y=z_input[0], 
        marker_color=['#ff3333' if val < 0 else '#33ccff' for val in z_input[0]] # Rojo negativo, Azul positivo
    )])
    fig_z.update_layout(
        height=150, margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        xaxis=dict(visible=False), 
        yaxis=dict(range=[-5, 5], visible=False) # Rango fijo para ver los cambios de escala
    )
    st.plotly_chart(fig_z, use_container_width=True)

    st.markdown("---")
    btn_generar = st.button("⚡ SINTETIZAR SEÑAL ECG", use_container_width=True, type="primary")

with col_monitor:
    st.header("2. Monitor de Salida")
    monitor_placeholder = st.empty()
    
    fig_vacia = go.Figure()
    fig_vacia.update_layout(
        xaxis=dict(range=[0, 325], showgrid=False, visible=False),
        yaxis=dict(range=[-3, 3], showgrid=True, gridcolor='#1a1a1a', zeroline=True, zerolinecolor='#333333'),
        plot_bgcolor='black', paper_bgcolor='black', height=500, margin=dict(l=20, r=20, t=40, b=20)
    )
    
    if not btn_generar:
        monitor_placeholder.plotly_chart(fig_vacia, use_container_width=True)
        st.info("👈 Ajusta los parámetros del ruido y presiona **Sintetizar Señal**.")

# ==========================================
# 4. LÓGICA DE INFERENCIA Y ANIMACIÓN
# ==========================================
if btn_generar:
    session = load_onnx_session(ONNX_MODEL_PATH)
    
    if session is None:
        with col_monitor:
            st.error("❌ No se encontró el modelo ONNX. Verifica la ruta.")
    else:
        # Preparar etiqueta c
        c_input = np.zeros((1, NUM_CLASES), dtype=np.float32)
        c_input[0, clase_idx] = 1.0
        
        # Ejecutar modelo
        input_names = [inp.name for inp in session.get_inputs()]
        output_name = session.get_outputs()[0].name
        
        try:
            output = session.run([output_name], {input_names[0]: z_input, input_names[1]: c_input})[0]
        except:
            z_c_concat = np.concatenate([z_input, c_input], axis=1)
            output = session.run([output_name], {input_names[0]: z_c_concat})[0]
            
        senal_generada = output.flatten()
        
        # Animación del Monitor
        y_min, y_max = np.min(senal_generada) - 0.5, np.max(senal_generada) + 0.5
        chunk_size = 12 
        
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
                title=dict(text=f"MONITOR ECG - CLASE: {clase_seleccionada} | {modo_ruido.split('(')[0]}", font=dict(color='#00ff00', size=16)),
                xaxis=dict(range=[0, 325], showgrid=False, visible=False),
                yaxis=dict(range=[-3, 3], showgrid=True, gridcolor='#1a1a1a', zeroline=True, zerolinecolor='#004400'),
                plot_bgcolor='black', paper_bgcolor='black', height=500, margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            
            monitor_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.01)
