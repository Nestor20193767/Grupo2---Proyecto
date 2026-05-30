# Proyecto de Generación Sintética de Señales ECG con TCN-CVAE
Este repositorio/directorio contiene todo el código y la documentación necesaria para la descarga, preprocesamiento, entrenamiento y generación de datos sintéticos de señales electrocardiográficas (ECG) utilizando la base de datos MIT-BIH y una arquitectura TCN-CVAE.
## Estructura del Directorio
El proyecto está organizado en las siguientes carpetas principales:
### 1. `Analisis Database MIT-BIH/`
Esta carpeta contiene la **Fase 1** del proyecto (Adquisición y Preprocesamiento):
- **`Dabase_MIT_Arritmias.ipynb`**: Notebook de Jupyter original con la exploración interactiva de los datos.
- **`ecg_avance2_pipeline.py`**: Script de Python que automatiza la descarga de MIT-BIH, el filtrado (pasa-alto, notch, pasa-bajo), la segmentación de latidos y el manejo del desbalance de clases.
### 2. `Arquitectura y entrenamiento/`
Esta carpeta contiene la **Fase 2** del proyecto (Entrenamiento y Generación):
- **`ECG_TCNCVAE_Entrenamiento.ipynb`**: Notebook de Jupyter interactivo para la creación y entrenamiento del modelo.
- **`ecg_tcncvae_entrenamiento.py`**: Script de Python con la definición de la arquitectura de la red (Encoder, Decoder, TCN Stack), la función de pérdida (MSE + KL Divergence), el bucle de entrenamiento y la validación de latidos sintéticos.
### 3. `src/` (Integración)
Esta carpeta contiene los archivos unificados y limpios listos para uso local:
- **`ARRYS.py`**: Es el script principal e integrado. Combina todo el flujo de trabajo (preprocesamiento y entrenamiento) en un solo archivo `.py` que puede ejecutarse de manera local sin depender de Jupyter o Google Colab.
- **`explicacion_generacion_sintetica_ecg.md`**: Documento detallado que explica paso a paso la teoría y funcionamiento de la arquitectura TCN-CVAE empleada en `ARRYS.py`.
- **`requirements.txt`**: Archivo con todas las librerías de Python requeridas para ejecutar el proyecto en un entorno local (e.g. `numpy`, `wfdb`, `torch`, `scipy`, etc.).
---
## Cómo empezar
1. Instala las dependencias necesarias. Ve a la carpeta `src` (o donde tengas `requirements.txt`) y ejecuta:
   ```bash
   pip install -r requirements.txt
   ```
2. Si deseas ejecutar el flujo completo (desde la descarga hasta la generación sintética), puedes ejecutar el archivo unificado en la carpeta `src`:
   ```bash
   python src/ARRYS.py
   ```
3. Alternativamente, puedes explorar y ejecutar las fases de forma independiente usando los scripts de las carpetas `Analisis Database MIT-BIH` y `Arquitectura y entrenamiento`.
