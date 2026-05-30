# Clasificación Automática de Arritmias Cardíacas

### Proyecto de Reconocimiento de Patrones

Este repositorio contiene el diseño e implementación de un modelo de inteligencia artificial para la clasificación automática de arritmias cardíacas a partir de señales de electrocardiograma (ECG) de una sola derivación. El proyecto evoluciona hacia una arquitectura híbrida que integra Redes Convolucionales Temporales (TCN) y Autoencoders Variacionales Condicionales (cVAE).

---

## Contexto del Problema

La revisión manual de registros de ECG prolongados (como el monitoreo Holter) consume una cantidad excesiva de tiempo y recursos hospitalarios. Además, es altamente susceptible a errores humanos derivados de la fatiga visual y cognitiva.

**Objetivo:** Mitigar las limitaciones del análisis manual mediante un sistema que captura la secuencia temporal de las señales con TCN y soluciona el desbalance de clases sintetizando datos para las categorías minoritarias con cVAE.

## Datos: MIT-BIH Arrhythmia Database

El sistema se entrena y valida utilizando el conjunto de datos MIT-BIH Arrhythmia Database. Este es el estándar de referencia e incluye las siguientes características:

* **Registros:** 48 extractos de media hora de duración correspondientes a señales ambulatorias de dos canales.
* **Anotaciones:** Aproximadamente 110,000 latidos individuales validados por al menos dos cardiólogos.
* **N (Normal):** Latidos normales o de paquete de rama.
* **S (Supraventricular):** Latidos ectópicos supraventriculares.
* **V (Ventricular):** Latidos ectópicos ventriculares.
* **F (Fusión):** Latidos de fusión entre un latido normal y uno ventricular.
* **Q (Desconocido/No clasificable):** Latidos cuyo origen no pudo ser determinado o con marcapasos.

---

## Pipeline del Proyecto (En Desarrollo)

El proyecto sigue una metodología de procesamiento dividida en fases clave:

1. **Acondicionamiento y Filtrado:** Aplicación de un filtro pasa-alto de 0.5 Hz, un filtro Notch de 60 Hz y un filtro pasa-bajo de 45 Hz para limpiar la señal.
2. **Segmentación:** Extracción de latidos utilizando una ventana fija de 256 muestras centrada en el pico R.
3. **Normalización:** Escalamiento Min-Max en el rango de 0 a 1 para homogenizar las amplitudes.
4. **Modelado Híbrido:** Uso de TCN para extraer características de largo alcance y cVAE para inyectar muestras sintéticas de alta fidelidad en el entrenamiento.

---

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

