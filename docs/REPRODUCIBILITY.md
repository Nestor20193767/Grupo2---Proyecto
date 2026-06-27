# Guía de reproducibilidad

## Niveles de reproducción

### 1. Inferencia con modelos entrenados

Este es el camino plenamente soportado por el repositorio final:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1

pip install -r requirements-app.txt
pip install -e .
python scripts/check_models.py --strict
python scripts/generate_ecg.py --class-name NSR --n 8 --seed 42 --classify
```

### 2. Regeneración de tablas y gráficos reportados

```bash
pip install -r requirements.txt
python scripts/reproduce_figures.py
```

### 3. Reentrenamiento

El notebook de entrenamiento debe quedar en `notebooks/02_training_tcn_cvae_physionet.ipynb` después de ejecutar la migración. Para que el reentrenamiento sea auditable, registre al inicio del notebook:

- versión exacta del dataset;
- lista o hash de registros de cada split;
- semilla de Python, NumPy y PyTorch;
- versión de CUDA/cuDNN y GPU;
- hiperparámetros;
- ruta de salida de checkpoints y métricas.

## Semillas

Los ejemplos usan semilla 42. ONNX Runtime en CPU ofrece el camino de inferencia más estable entre equipos. El entrenamiento en GPU puede presentar pequeñas diferencias numéricas aun con semillas fijadas.

## Validación antes de entregar

```bash
python scripts/check_repository.py --allow-missing-models
python -m compileall app src scripts tests
pytest
```

Quite `--allow-missing-models` en la verificación definitiva después de copiar los modelos.
