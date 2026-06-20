import streamlit as st
import numpy as np
import pandas as pd
import onnxruntime as ort
import plotly.graph_objects as go
import time

# ==========================================
# 1. CONFIGURACIÓN DEL DASHBOARD
# ==========================================
st.set_page_config(page_title="TCN-cVAE Demo", layout="wide", initial_sidebar_state="expanded")

# CSS personalizado para darle un aspecto más técnico/médico
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #00ff00; }
    h1, h2, h3 { color: #ffffff; }
    .stAlert { background-color: #1e1e1e; color: #ffffff; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. PARÁMETROS DEL MODELO
# ==========================================
ONNX_MODEL_PATH = "tcncvae_decoder_physionet.onnx"
CLASES_ARRITMIA = ['AFL', 'LVQRS', 'NSIVCB', 'Other', 'PAC', 'QTIE', 'STD']
NUM_CLASES = len(CLASES_ARRITMIA)
LATENT_DIM = 32 # Ajusta esto a la dimensión de tu cuello de botella

@st.cache_resource
def load_onnx_session(model_path):
    try:
        return ort.InferenceSession(model_path)
    except:
        return None

# ==========================================
# 3. INTERFAZ GRÁFICA (UI)
# ==========================================
st.title("🫀 Inferencia Generativa: TCN-cVAE en Tiempo Real")
st.markdown("Demostración de la reconstrucción de señales electrocardiográficas a partir de ruido latente y condicionamiento clínico.")

# Dividir la pantalla en dos columnas: Controles e Inputs (Izquierda) / Salida (Derecha)
col_inputs, col_monitor = st.columns([1, 2.5])

with col_inputs:
    st.header("1. Entradas del Modelo")
    
    # ENTRADA C: Condición
    st.subheader("Condición ($c$)")
    clase_seleccionada = st.selectbox("Clase de Arritmia a sintetizar:", CLASES_ARRITMIA)
    clase_idx = CLASES_ARRITMIA.index(clase_seleccionada)
    
    st.markdown("---")
    
    # ENTRADA Z: Ruido
    st.subheader("Espacio Latente ($z$)")
    modo_ruido = st.radio("Origen del ruido:", ["Generar Aleatoriamente", "Subir mi propio ruido (CSV)"])
    
    z_input = None
    if modo_ruido == "Subir mi propio ruido (CSV)":
        archivo_z = st.file_uploader(f"Sube un CSV de 1x{LATENT_DIM}", type=['csv'])
        if archivo_z is not None:
            try:
                df_z = pd.read_csv(archivo_z, header=None)
                if df_z.shape == (1, LATENT_DIM):
                    z_input = df_z.values.astype(np.float32)
                    st.success("Ruido cargado correctamente.")
                else:
                    st.error(f"El CSV debe tener exactamente 1 fila y {LATENT_DIM} columnas.")
            except:
                st.error("Error al leer el archivo CSV.")
    
    if z_input is None:
        # Generar ruido aleatorio si no se ha subido nada o se seleccionó la opción
        z_input = np.random.randn(1, LATENT_DIM).astype(np.float32)
    
    # Visualizar el ruido para el jurado
    st.markdown("**Visualización del vector $z$ (Input):**")
    fig_z = go.Figure(data=[go.Bar(y=z_input[0], marker_color='#0088ff')])
    fig_z.update_layout(
        height=150, margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor='#0e1117', paper_bgcolor='#0e1117',
        xaxis=dict(visible=False), yaxis=dict(visible=False)
    )
    st.plotly_chart(fig_z, use_container_width=True)

    st.markdown("---")
    btn_generar = st.button("⚡ SINTETIZAR SEÑAL", use_container_width=True, type="primary")

with col_monitor:
    st.header("2. Salida del Generador")
    monitor_placeholder = st.empty()
    
    # Monitor vacío inicial
    fig_vacia = go.Figure()
    fig_vacia.update_layout(
        xaxis=dict(range=[0, 325], showgrid=False, visible=False),
        yaxis=dict(range=[-3, 3], showgrid=True, gridcolor='#1a1a1a', zeroline=True, zerolinecolor='#333333'),
        plot_bgcolor='black', paper_bgcolor='black', height=500, margin=dict(l=20, r=20, t=40, b=20)
    )
    
    if not btn_generar:
        monitor_placeholder.plotly_chart(fig_vacia, use_container_width=True)
        st.info("👈 Selecciona los parámetros y presiona **Sintetizar Señal** para iniciar la inferencia.")

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
            # Intento: Entradas separadas
            output = session.run([output_name], {input_names[0]: z_input, input_names[1]: c_input})[0]
        except:
            # Fallback: Entradas concatenadas
            z_c_concat = np.concatenate([z_input, c_input], axis=1)
            output = session.run([output_name], {input_names[0]: z_c_concat})[0]
            
        senal_generada = output.flatten()
        
        # Animación del Monitor
        y_min, y_max = np.min(senal_generada) - 0.5, np.max(senal_generada) + 0.5
        chunk_size = 8 # Velocidad de la animación
        
        for i in range(0, 325 + chunk_size, chunk_size):
            x_data = np.arange(i)
            y_data = senal_generada[:i]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=x_data, y=y_data, mode='lines', 
                line=dict(color='#00ff00', width=2.5), 
                name="ECG"
            ))
            
            # Punto brilloso al final de la línea para dar efecto de "dibujado"
            if i > 0 and i < 325:
                fig.add_trace(go.Scatter(
                    x=[x_data[-1]], y=[y_data[-1]], mode='markers',
                    marker=dict(color='white', size=8, symbol='circle'), showlegend=False
                ))
            
            fig.update_layout(
                title=dict(text=f"MONITOR ECG - SINTETIZANDO CLASE: {clase_seleccionada}", font=dict(color='#00ff00', size=20)),
                xaxis=dict(range=[0, 325], showgrid=False, visible=False),
                yaxis=dict(range=[y_min, y_max], showgrid=True, gridcolor='#1a1a1a', zeroline=True, zerolinecolor='#004400'),
                plot_bgcolor='black', paper_bgcolor='black', height=500, margin=dict(l=20, r=20, t=40, b=20),
                showlegend=False
            )
            
            monitor_placeholder.plotly_chart(fig, use_container_width=True)
            time.sleep(0.01) # Pausa mínima para fluidez
