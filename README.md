# Clasificación Automática de Arritmias Cardíacas

### Proyecto de Reconocimiento de Patrones

Este repositorio contiene el diseño e implementación de un modelo de inteligencia artificial para la clasificación automática de arritmias cardíacas a partir de señales de electrocardiograma (ECG) de una sola derivación. El proyecto evoluciona hacia una arquitectura híbrida que integra Redes Convolucionales Temporales (TCN) y Autoencoders Variacionales Condicionales (cVAE).

---

## Contexto del Problema

La revisión manual de registros de ECG prolongados (como el monitoreo Holter) consume una cantidad excesiva de tiempo y recursos hospitalarios. Además, es altamente susceptible a errores humanos derivados de la fatiga visual y cognitiva.

**Objetivo:** Mitigar las limitaciones del análisis manual mediante un sistema que captura la secuencia temporal de las señales con TCN y soluciona el desbalance de clases sintetizando datos para las categorías minoritarias con cVAE.

## Base de Datos / Datasets

El desarrollo y validación de este framework (TCN-cVAE) se fundamenta en el análisis comparativo de dos conjuntos de datos de electrocardiografía con características estructurales, de resolución y de origen clínico significativamente opuestas:

### 1. ECG5000 Dataset
* **Origen:** Derivado de la base de datos de insuficiencia cardíaca congestiva BIDMC disponible en PhysioNet.
* **Volumen:** 5,000 latidos individuales pre-extraídos.
* **Dimensiones:** Registro monocanal (unidimensional) con una longitud fija de 140 puntos de datos por latido.
* **Limitaciones Fisiológicas:** Este dataset experimental cuenta con un preprocesamiento agresivo donde se aplicó interpolación matemática para forzar la uniformidad de los vectores. Esto causó la pérdida de la tasa de muestreo original (Hz) y un enventanado asimétrico que desplaza el pico R hacia los extremos, distorsionando la morfología clásica de las ondas cardíacas. Fue utilizado estrictamente como base de validación inicial y prueba de concepto.

### 2. 12-Lead Arrhythmia Database (PhysioNet)
* **Origen:** *A large scale 12-lead electrocardiogram database for arrhythmia study*.
* **Volumen:** 45,152 registros clínicos correspondientes a pacientes independientes en condiciones de reposo.
* **Dimensiones:** Matriz multicanal de 12 derivaciones estándar simultáneas (I, II, III, aVR, aVL, aVF, V1-V6). Cada registro cuenta con una duración continua de 10 segundos.
* **Resolución Temporal:** Frecuencia de muestreo nativa de 500 Hz (5,000 muestras por derivación), manteniendo intactas las proporciones reales y la morfología clínica completa (complejo QRS, segmentos ST, ondas P y T).
* **Anotaciones:** Diagnósticos clínicos reales indexados mediante ontologías estandarizadas de códigos SNOMED CT.

---

### Análisis Comparativo e Impacto en el Pipeline

| Característica | ECG5000 | 12-Lead Arrhythmia Database |
| :--- | :--- | :--- |
| **Tipo de Señal** | Latido aislado (Interpolado) | Señal continua de 10 segundos |
| **Canales / Derivaciones**| 1 (Unidimensional) | 12 (Multicanal) |
| **Frecuencia de Muestreo**| Nula / Relativa a la interpolación | 500 Hz (Fija de hardware) |
| **Puntos por Registro** | 140 muestras | 60,000 muestras totales ($12 \times 5,000$) |
| **Morfología Clínica** | Deformada / Desfasada | Preservada (Fisiológicamente exacta) |
| **Distribución de Clases**| Sintética / Reducida | Desbalanceada (Escenario clínico real) |

### Justificación de la Migración
El uso original de **ECG5000** permitió verificar la viabilidad técnica de la red convolucional temporal (TCN) y el autoencoder variacional condicional (cVAE) para procesar secuencias numéricas homogeneizadas. Sin embargo, para evitar el sobreajuste a señales artificialmente deformadas y permitir que el modelo generalice ante electrocardiogramas reales, el proyecto migró hacia la **Base de Datos de 12 Derivaciones de PhysioNet**.

Esta transición requirió el diseño de un pipeline de preprocesamiento avanzado que incluye:
1. **Filtrado Digital:** Remoción de la deriva de línea base (*baseline wander*) mediante un filtro pasa-altas a 0.5 Hz e interferencia de red eléctrica mediante filtro Notch (50/60 Hz).
2. **Segmentación Fisiológica:** Localización precisa de complejos QRS mediante el algoritmo de Pan-Tompkins para extraer ventanas simétricas centradas en el pico R (preservando el ancho real en milisegundos).
3. **Optimización de Datos:** Ingesta vectorizada de archivos `.mat`/`.hea` y estructuración del dataset mediante un motor de datos de alto rendimiento para mitigar los cuellos de botella en la RAM antes de la conversión a tensores de entrenamiento.

## Instalación y Uso

*(Instrucciones preliminares)*

1. Clonar el repositorio:
```bash
git clone https://github.com/Nestor20193767/ReconocimientodePatrones.git
```

2. Instalar dependencias base:
```bash
pip install numpy pandas matplotlib scipy wfdb
```

---

## Equipo de Desarrollo

Este proyecto es una colaboración de estudiantes de Ingeniería Biomédica, realizada por:

* **Néstor Manuel Allende Heredia**  - [GitHub](https://www.google.com/search?q=https://github.com/Nestor20193767)
* **Marx Christian Ríos Morales**  - [GitHub](https://www.google.com/search?q=https://github.com/MarxRios)
* **Renzo William Luna Aliaga**  - [GitHub](https://www.google.com/search?q=https://github.com/RenzoLuna)
* **Luis Fernando Galván Nuñez**  - [GitHub](https://www.google.com/search?q=https://github.com/LuisGalvan)


## 📄 Licencia

Este proyecto se distribuye bajo la licencia [MIT/Libre], lo que permite su uso para fines académicos y de investigación.

