# Resultados principales

## Entrenamiento del generador

- 140 épocas.
- Pérdida de reconstrucción de validación: 0.11811.
- Gap train-validación: -0.01837.
- Unidades latentes activas: 32/32.
- Ratio de diversidad aproximado: 1.0.
- Memorization rate aproximada: 0.
- Coverage reportado: 0.45–0.56.

## Utilidad downstream

El modo C, entrenado con datos reales y sintéticos, reporta la mejor combinación de accuracy, recall macro y F1 macro. Frente al modo A, el recall macro aumenta 15.01 puntos porcentuales y el F1 macro 18.48 puntos porcentuales.

## Figuras

Las figuras extraídas del informe están en `figures/`. Los valores estructurados se encuentran en `results/metrics/`, de modo que puedan auditarse y regenerarse sin depender de capturas.
