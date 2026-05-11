# Clasificación Automática de Arritmias Cardíacas 🫀

### Proyecto de Reconocimiento de Patrones

Este repositorio contiene el desarrollo de un sistema inteligente diseñado para la detección y clasificación de anomalías en señales de **Electrocardiograma (ECG)**. El proyecto aborda el reto clínico de procesar grandes volúmenes de datos cardiacos para identificar latidos patológicos de manera eficiente.

-----

## 📖 Contexto del Problema

La interpretación manual de registros de ECG prolongados (como los estudios Holter) es una tarea que consume mucho tiempo y es propensa al error humano por fatiga. Las arritmias pueden variar desde hallazgos benignos hasta condiciones que ponen en riesgo la vida.

**Objetivo:** Construir un clasificador multiclase que analice segmentos de señal de una sola derivación y categorice cada latido automáticamente, facilitando la priorización de casos críticos en el entorno hospitalario.

## 📊 Datos: MIT-BIH Arrhythmia Database

El sistema se entrena y valida utilizando el dataset **MIT-BIH**, un estándar en la industria que incluye:

  * **Registros:** 48 extractos de 30 minutos de dos derivaciones.
  * **Anotaciones:** Etiquetas de latidos revisadas por expertos.
  * **Categorías (Estándar AAMI):**
      * **N**: Latidos normales.
      * **S**: Latidos supraventriculares.
      * **V**: Latidos ventriculares.
      * **F**: Latidos de fusión.
      * **Q**: Latidos no clasificables.

-----

## 🛠️ Pipeline del Proyecto (En Desarrollo)

El proyecto sigue una metodología de reconocimiento de patrones dividida en cuatro fases:

1.  **Adquisición y Limpieza:** Carga de señales desde PhysioNet y eliminación de ruidos (movimiento, respiración, interferencia eléctrica).
2.  **Segmentación:** Identificación de complejos QRS y extracción de ventanas temporales.
3.  **Extracción/Aprendizaje de Características:** Análisis de la morfología del latido y variabilidad del ritmo.
4.  **Clasificación:** Evaluación de distintas arquitecturas (Machine Learning Clásico vs. Deep Learning) para determinar la mayor precisión y sensibilidad clínica.

-----

## 💻 Instalación y Uso

*(Instrucciones preliminares)*

1.  Clonar el repositorio:
    ```bash
    git clone https://github.com/Nestor20193767/ReconocimientodePatrones.git
    ```
2.  Instalar dependencias base:
    ```bash
    pip install numpy pandas matplotlib scipy wfdb
    ```

-----

## 👥 Equipo de Desarrollo

Este proyecto es una colaboración realizada por:

  * **Néstor Manuel Allende Heredia** - [GitHub](https://www.google.com/search?q=https://github.com/usuario1)
  * **Marx Christian Ríos Morales** - [GitHub](https://www.google.com/search?q=https://github.com/usuario2)
  * **Renzo William Luna Aliaga** - [GitHub](https://www.google.com/search?q=https://github.com/usuario3)
  * **Nombre del Autor 4** - [GitHub](https://www.google.com/search?q=https://github.com/usuario4)

-----

## 📄 Licencia

Este proyecto se distribuye bajo la licencia [MIT/Libre], lo que permite su uso para fines académicos y de investigación.

