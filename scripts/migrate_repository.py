#!/usr/bin/env python3
"""Migra el repositorio actual a la estructura final sin borrar los originales."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

FILE_MAPPING = {
    "App/main.py": "legacy/app/streamlit_app_v3_original.py",
    "App/main2.py": "legacy/app/streamlit_app_v2.py",
    "Codigo/src/ARRYS-2.ipynb": "notebooks/02_training_tcn_cvae_physionet.ipynb",
    "Codigo/src/ARRYS.py": "legacy/mit_bih/ARRYS.py",
    "Codigo/Arquitectura y entrenamiento/Arquitectura y entrenamiento.ipynb": "legacy/mit_bih/Arquitectura_y_entrenamiento.ipynb",
    "Codigo/Arquitectura y entrenamiento/Arquitectura y entrenamiento.py": "legacy/mit_bih/Arquitectura_y_entrenamiento.py",
    "Codigo/Analisis Database MIT-BIH/Analisis Database MIT-BIH.ipynb": "legacy/mit_bih/Analisis_Database_MIT_BIH.ipynb",
    "Codigo/Analisis Database MIT-BIH/Pipeline Final.py": "legacy/mit_bih/Pipeline_Final.py",
}


def copy_one(source: Path, destination: Path, dry_run: bool) -> None:
    if not source.exists():
        print(f"[omite] No existe: {source}")
        return
    if destination.exists():
        print(f"[omite] Ya existe: {destination}")
        return
    print(f"[copia] {source} -> {destination}")
    if not dry_run:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("."), help="Raíz del repositorio antiguo")
    parser.add_argument("--destination", type=Path, default=Path("."), help="Raíz ya actualizada")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for old, new in FILE_MAPPING.items():
        copy_one(args.source / old, args.destination / new, args.dry_run)

    model_source = args.source / "App" / "Modelo"
    model_destination = args.destination / "models"
    if model_source.exists():
        for item in sorted(model_source.iterdir()):
            if item.is_file():
                copy_one(item, model_destination / item.name, args.dry_run)
    else:
        print(f"[omite] No existe el directorio de modelos: {model_source}")

    print("Migración terminada. Ejecuta: python scripts/check_models.py --strict")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
