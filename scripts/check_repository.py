#!/usr/bin/env python3
"""Comprueba que la entrega tenga la estructura mínima solicitada."""

from __future__ import annotations

import argparse
from pathlib import Path

REQUIRED = (
    "README.md",
    "requirements.txt",
    "requirements-app.txt",
    "app/streamlit_app.py",
    "src/arrys/inference.py",
    "models/README.md",
    "results/README.md",
    "results/metrics/classifier_modes.csv",
    "figures/01_training_curves.png",
    "docs/paper/paper_ieee_arrys.pdf",
    "notebooks/00_reproduce_reported_results.ipynb",
    "scripts/generate_ecg.py",
)
MODEL_REQUIRED = (
    "models/tcncvae_decoder_physionet.onnx",
    "models/tcncvae_decoder_physionet.onnx.data",
    "models/clf_aug_physionet.onnx",
    "models/clf_aug_physionet.onnx.data",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--allow-missing-models", action="store_true")
    args = parser.parse_args()

    missing = [item for item in REQUIRED if not (args.root / item).exists()]
    model_missing = [item for item in MODEL_REQUIRED if not (args.root / item).exists()]

    if missing:
        print("Archivos obligatorios faltantes:")
        for item in missing:
            print(f"  - {item}")
    if model_missing:
        print("Artefactos de modelo no encontrados:")
        for item in model_missing:
            print(f"  - {item}")
        if args.allow_missing_models:
            print("Se aceptan temporalmente porque se usó --allow-missing-models.")

    failed = bool(missing) or (bool(model_missing) and not args.allow_missing_models)
    if failed:
        return 1
    print("Estructura del repositorio: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
