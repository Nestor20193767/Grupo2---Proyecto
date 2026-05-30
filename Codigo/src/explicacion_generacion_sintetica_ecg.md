# Generación Sintética de Señales ECG: Guía Detallada del Código (ARRYS.py)

Este documento explica de manera detallada el funcionamiento de la lógica contenida en `ARRYS.py`. El objetivo principal del código es la generación de datos sintéticos de señales electrocardiográficas (ECG) para abordar problemas como el desbalance de clases en diagnósticos automatizados.

Para lograrlo, el código se divide fundamentalmente en **dos grandes fases**:
1. El **pipeline de datos y preprocesamiento**, que prepara la señal cruda.
2. La **arquitectura TCN-CVAE (Temporal Convolutional Network - Conditional Variational Autoencoder)** que aprende la morfología de cada latido y genera nuevos datos sintéticos de alta fidelidad.

---

## Fase 1: Adquisición y Preprocesamiento de Datos

### 1.1. Descarga de la Base de Datos (MIT-BIH)
El proceso comienza descargando directamente los registros de la base de datos *MIT-BIH Arrhythmia Database* (desde PhysioNet) empleando la librería `wfdb`. Esta base de datos es el "gold-standard" (estándar de oro) en cardiología para el análisis de arritmias, proporcionando grabaciones etiquetadas a 360 Hz.

### 1.2. Filtrado de la Señal Cruda (Limpieza)
El ECG original suele venir con ruidos e interferencias. Para limpiar la señal se aplican 3 filtros secuenciales:
1. **Filtro Pasa-Alto (0.5 Hz):** Elimina el "baseline wander" (desviación o deriva de la línea base provocada por la respiración y movimientos del paciente).
2. **Filtro Notch (60 Hz):** Elimina la interferencia de la red eléctrica.
3. **Filtro Pasa-Bajo (45 Hz):** Elimina el ruido de alta frecuencia, principalmente electromiográfico (actividad de los músculos cercanos).

> [!TIP]
> Puedes visualizar la densidad espectral en la gráfica `psd_analisis.png` que genera el programa para confirmar cómo se "cortan" exactamente estas frecuencias de ruido sin afectar las ondas cardíacas importantes.

### 1.3. Segmentación de Latidos
En lugar de procesar grabaciones completas, el modelo se entrena **latido por latido**:
- Aprovechando que la base de datos ya provee las ubicaciones de los picos 'R' (el punto más alto del latido), el código recorta ventanas de tamaño fijo (**256 muestras**, que a 360Hz equivalen a ~711 milisegundos), centrando el pico R exactamente a la mitad.
- Inmediatamente, cada latido individual se **normaliza** (Min-Max) para que sus valores fluctúen estrictamente entre `[0, 1]`.

### 1.4. Mapeo AAMI y Prevención de "Data Leakage"
Las anotaciones de MIT-BIH se mapean al estándar AAMI, concentrándose en 5 súper-clases:
- **N**: Latidos normales
- **S**: Supraventriculares ectópicos
- **V**: Ventriculares ectópicos
- **F**: Latidos de Fusión
- **Q**: Artefactos o desconocidos

**División de Pacientes:** Para entrenar adecuadamente, se separa el dataset en Train/Validation/Test. Se asegura que un mismo **paciente** nunca esté en dos de estos conjuntos a la vez (*patient-level split*). Esto previene que el modelo simplemente se memorice la morfología particular del paciente (data leakage).

---

## Fase 2: El Modelo TCN-CVAE

El cerebro de la generación es un autoencoder variacional condicional impulsado por Redes Convolucionales Temporales (TCN).

### 2.1. Bloques Constructivos (TCNStack)
A diferencia de los modelos como LSTMs o RNNs (que procesan secuencialmente), el código usa un **CausalDilatedBlock** con convoluciones dilatadas. Al no tratar con señales "en tiempo real", el modelo usa un relleno (padding) simétrico para mirar "al pasado y al futuro" del pico R simultáneamente. Las dilataciones exponenciales permiten un campo receptivo lo suficientemente amplio como para captar el complejo QRS en su totalidad de forma muy rápida.

### 2.2. El Encoder (Codificador)
El Encoder se encarga de resumir el latido (señal) a sus componentes esenciales:
- Recibe un latido `x` (tensor de 256 puntos) y una etiqueta condicional `c` (ej. "Este latido es tipo Ventricular (V)").
- Utiliza la red TCN para procesarlo y proyecta la información a un **espacio latente** (z) de menor dimensionalidad (32 dimensiones). En lugar de emitir puntos absolutos, emite una media (`mu`) y una varianza (`log_var`).

### 2.3. El Truco de Reparametrización
El espacio latente es aleatorio (probabilístico) para permitir la "generación" y no solo la "copia". Usando la reparametrización $z = \mu + \sigma \cdot \epsilon$, donde $\epsilon$ es ruido aleatorio (distribución normal), la red puede ser entrenada de manera continua utilizando Backpropagation.

### 2.4. El Decoder (Decodificador)
El Decoder es el motor de creación:
- Toma los 32 valores del espacio latente ($z$) generados por el encoder **más** la condición ($c$, la clase deseada de latido).
- Utiliza un proceso de "Upsampling" y capas TCN para ir extendiendo la información progresivamente (de 32 → 64 → 128 → 256 puntos).
- La última capa genera la morfología final (un vector de 256 valores entre 0 y 1).

### 2.5. Función de Pérdida (Loss) y β-Scheduler
El aprendizaje ocurre al minimizar una ecuación de dos partes:
1. **Reconstrucción (MSE):** Mide qué tan igual es el latido generado al latido original.
2. **KL Divergence:** "Fuerza" a las variables del espacio latente a organizarse como una campana de Gauss, lo que mantiene el espacio ordenado para poder inventar latidos sin que sean puro ruido.

> [!NOTE]
> **β-Scheduler (KL Annealing):** Al inicio del entrenamiento, el peso de la divergencia KL ($\beta$) se ajusta a `0`. Esto le da oportunidad a la red de aprender a reconstruir primero antes de que le exijan que sus representaciones latentes sean "matemáticamente ordenadas". Posteriormente, $\beta$ sube gradualmente hasta `1`.

---

## Fase 3: Generación Sintética y Validación

Una vez que el modelo está entrenado, el **Encoder se desecha** para efectos prácticos. 

### 3.1. Proceso de Inferencia (Creación de nuevos datos)
El flujo final (implementado en el método `generate` de la clase `TCNCVAE`) es el siguiente:
1. El sistema escoge una clase AAMI que quiera crear, por ejemplo, "V" (Ventricular ectópico).
2. Samplea de manera aleatoria ruido de una distribución normal estándar: $z \sim \mathcal{N}(0, I)$. Es decir, se "inventa" un punto en el espacio latente.
3. El Decoder toma este punto aleatorio, junto con la etiqueta "V" inyectada.
4. El Decoder escupe una señal de 256 puntos completamente nueva y altamente realista.

### 3.2. Evaluación de los Resultados
Para garantizar que la data sintética es útil (ej. puede servir para entrenar clasificadores de médicos artificiales):
- Se compara visualmente en gráficas la curva original vs la sintética.
- Se hace un **Análisis Espectral PSD**: Verifica que el contenido frecuencial y armónico del latido falso sea estadísticamente el mismo al latido real (solapando las curvas).
- Se elabora un mapa **t-SNE**, que permite visualizar en 2D el espacio latente de 32D para certificar que el modelo supo agrupar las distintas cardiopatías por separado de forma semántica.

Finalmente, el programa salva paquetes de (por ejemplo) 1,000 latidos sintéticos de cada clase en el directorio `./ecg_synthetic/`, equilibrando con esto la base de datos de manera limpia, lista para un clasificador Downstream.
