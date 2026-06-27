# Migración del repositorio actual

## 1. Cree una rama de respaldo

```bash
git checkout main
git pull origin main
git checkout -b refactor/repositorio-final
```

## 2. Copie esta plantilla sobre la raíz del repositorio

No borre aún `App/` ni `Codigo/`. Después ejecute:

```bash
python scripts/migrate_repository.py --source . --destination . --dry-run
python scripts/migrate_repository.py --source . --destination .
```

La migración copia los artefactos de `App/Modelo/` a `models/`, conserva las aplicaciones antiguas en `legacy/` y coloca el notebook PhysioNet en `notebooks/`. No sobrescribe archivos existentes.

## 3. Verifique

```bash
pip install -r requirements-dev.txt
pip install -e .
python scripts/check_models.py --strict
python scripts/check_repository.py
pytest
streamlit run app/streamlit_app.py
```

## 4. Retire duplicados

Cuando la app y los scripts funcionen desde la nueva estructura, elimine del seguimiento de Git las carpetas antiguas `App/` y `Codigo/`. Los respaldos relevantes ya deben estar en `legacy/`.

## 5. Publique con Git LFS

```bash
git lfs install
git add .
git commit -m "refactor: organizar repositorio final reproducible"
git push -u origin refactor/repositorio-final
```

Abra un Pull Request hacia `main`, compruebe la ejecución limpia en otro equipo y cree un release `v1.0.0`.
