# Asuntos pendientes antes de la entrega final

1. **F1 macro del modo C:** la Tabla II reporta 51.21%, mientras que la discusión menciona 30.31%. El repositorio usa 51.21% porque coincide con la tabla y la figura. Debe corregirse el texto del paper.
2. **Accuracy del modo C:** el texto afirma una caída respecto al modo A, pero los valores reportados son 52.70% frente a 35.47%. Debe reemplazarse esa explicación.
3. **AUC-ROC:** sigue pendiente y no debe presentarse como métrica calculada.
4. **Evaluación morfológica:** queda el marcador `[COMPLETAR: N/total]`. Debe sustituirse con el resultado real o eliminarse.
5. **Clases del análisis de errores:** se mencionan `LVQRS`, `PAC` y `NSIVCB`, aunque las figuras finales evalúan `SB`, `NSR`, `AFL`, `ST` y `AF`. Debe aclararse si corresponden a una corrida anterior.
6. **Orden de etiquetas:** verificar `label_encoder_physionet.pkl` y documentar sus `classes_`. La app ya evita asumir que el clasificador contiene las seis clases del generador.
7. **Figura 3:** el pie menciona modos A, B, C y D, pero la gráfica mostrada contiene A, B y C.
8. **Validación clínica:** las señales no están autorizadas para diagnóstico; la evaluación por especialistas sigue pendiente.
