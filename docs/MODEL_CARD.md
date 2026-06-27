# Model Card — ARRYS TCN-cVAE

## Descripción

Modelo generativo condicional para sintetizar latidos ECG de 325 muestras a 500 Hz. El encoder y el decoder usan bloques residuales TCN con dilataciones `[1, 2, 4, 8, 16, 32]`; el espacio latente tiene 32 dimensiones y el entrenamiento reporta `β = 0.035`.

## Uso previsto

- Aumentación de datos para investigación en clasificación de arritmias.
- Análisis de representaciones latentes y evaluación TSTR/TRTS.
- Demostración académica mediante Streamlit.

## Uso no previsto

- Diagnóstico, tamizaje, pronóstico o toma de decisiones clínicas.
- Sustitución de revisión por cardiólogos.
- Generación de historias clínicas o señales atribuidas a pacientes reales.

## Datos y clases

- Fuente: PhysioNet ECG-Arrhythmia.
- Generador: `AF`, `AFL`, `NSR`, `Others`, `SB`, `ST`.
- Clasificador final: ver `label_encoder_physionet.pkl`; el manuscrito excluye `Others`.

## Métricas reportadas

El modo aumentado real+sintético obtuvo accuracy 52.70%, recall macro 55.65% y F1 macro 51.21%. La validación morfológica fue preliminar y requiere evaluación clínica experta.

## Limitaciones

- Desbalance extremo y cobertura parcial del espacio real.
- AUC-ROC pendiente.
- No existe validación externa en PTB-XL o MIMIC-IV-ECG.
- Las clases agregadas y el orden de etiquetas deben verificarse contra los artefactos exportados.
