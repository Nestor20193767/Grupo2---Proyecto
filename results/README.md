# Resultados versionados

Los CSV de `results/metrics/` transcriben los valores reportados en el documento del proyecto. Las imágenes originales del informe están en `figures/`; las figuras regeneradas desde CSV se obtienen con:

```bash
python scripts/reproduce_figures.py
```

## Resumen TSTR/TRTS

| Modo | Entrenamiento → prueba | Accuracy | Recall macro | F1 macro |
|---|---|---:|---:|---:|
| A | Real → real | 35.47% | 40.64% | 32.73% |
| B | Sintético → real | 41.03% | 47.83% | 40.38% |
| C | Real + sintético → real | **52.70%** | **55.65%** | **51.21%** |
| D | Real → sintético | 34.56% | 50.30% | 32.69% |

AUC-ROC permanece pendiente en el manuscrito. Consulte `docs/KNOWN_ISSUES.md` antes de publicar la versión final.
