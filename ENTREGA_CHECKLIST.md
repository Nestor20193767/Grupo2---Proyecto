# Checklist de entrega

- [ ] Ejecutar `scripts/migrate_repository.py` y copiar todos los modelos.
- [ ] Ejecutar `scripts/check_models.py --strict`.
- [ ] Confirmar el orden de clases de `label_encoder_physionet.pkl`.
- [ ] Ejecutar la app en un entorno limpio.
- [ ] Corregir todos los puntos de `docs/KNOWN_ISSUES.md`.
- [ ] Completar AUC-ROC o declarar formalmente que no se reporta.
- [ ] Sustituir el marcador de evaluación morfológica del paper.
- [ ] Confirmar que el notebook de entrenamiento está en `notebooks/02_...`.
- [ ] Ejecutar `pytest` y revisar CI.
- [ ] Crear release `v1.0.0` con enlace estable a modelos, si no se usa Git LFS.
- [ ] Probar clonación y ejecución en otro equipo.
