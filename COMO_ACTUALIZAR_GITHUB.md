# Cómo aplicar esta actualización al repositorio GitHub

## Opción recomendada: rama de actualización

1. Descargue y descomprima `Grupo2-Proyecto-actualizado.zip` en una carpeta temporal.
2. Clone el repositorio actual:

```bash
git clone https://github.com/Nestor20193767/Grupo2---Proyecto.git
cd Grupo2---Proyecto
git checkout -b refactor/repositorio-final
```

3. **Antes de borrar `App/`**, copie los modelos y el notebook de entrenamiento usando el script incluido en la plantilla. Desde la raíz del repositorio:

```bash
python RUTA_A_LA_PLANTILLA/scripts/migrate_repository.py \
  --source . \
  --destination RUTA_A_LA_PLANTILLA \
  --dry-run

python RUTA_A_LA_PLANTILLA/scripts/migrate_repository.py \
  --source . \
  --destination RUTA_A_LA_PLANTILLA
```

4. Copie todo el contenido de la plantilla actualizada sobre la raíz del repositorio. No copie la carpeta contenedora; copie sus archivos y subcarpetas.

5. Instale y valide:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-dev.txt
pip install -e .
python scripts/check_models.py --strict
python scripts/check_repository.py
pytest
streamlit run app/streamlit_app.py
```

6. Cuando todo funcione, retire las carpetas duplicadas antiguas `App/` y `Codigo/`. Los archivos relevantes deben haber quedado en `models/`, `notebooks/` o `legacy/`.

7. Versione los binarios con Git LFS:

```bash
git lfs install
git lfs track "*.onnx" "*.onnx.data" "*.pt" "*.npz" "*.pkl"
git add .gitattributes
git add .
git commit -m "refactor: organizar repositorio final reproducible"
git push -u origin refactor/repositorio-final
```

8. Abra un Pull Request hacia `main`, revise el CI y cree el release `v1.0.0`.

## Ubicación actual de los modelos

Mientras se realiza la migración, los modelos entrenados existentes pueden consultarse en:

`https://github.com/Nestor20193767/Grupo2---Proyecto/tree/main/App/Modelo`

No elimine esa carpeta hasta confirmar que todos sus archivos fueron copiados a `models/`.
