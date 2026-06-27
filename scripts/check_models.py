#!/usr/bin/env python3
"""Verifica presencia, integridad básica e interfaces de los modelos ARRYS."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnxruntime as ort

EXPECTED = {
    "tcncvae_decoder_physionet.onnx": True,
    "tcncvae_decoder_physionet.onnx.data": True,
    "tcncvae_encoder_physionet.onnx": False,
    "clf_aug_physionet.onnx": True,
    "clf_aug_physionet.onnx.data": True,
    "clf_aug_physionet.pt": False,
    "label_encoder_physionet.pkl": False,
    "latent_bank.npz": False,
}


def shape_text(shape: list[object]) -> str:
    return " × ".join(str(value) for value in shape)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    missing_required = []
    print(f"Directorio: {args.model_dir.resolve()}")
    for name, required in EXPECTED.items():
        path = args.model_dir / name
        status = "OK" if path.exists() else ("FALTA" if required else "opcional")
        size = f"{path.stat().st_size / 1024:.1f} KiB" if path.exists() else "—"
        print(f"[{status:8}] {name:42} {size}")
        if required and not path.exists():
            missing_required.append(name)

    for name in ("tcncvae_decoder_physionet.onnx", "clf_aug_physionet.onnx"):
        path = args.model_dir / name
        if not path.exists():
            continue
        try:
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            print(f"\n{name}")
            for item in session.get_inputs():
                print(f"  input  {item.name}: {shape_text(item.shape)} ({item.type})")
            for item in session.get_outputs():
                print(f"  output {item.name}: {shape_text(item.shape)} ({item.type})")
        except Exception as exc:
            print(f"[ERROR] No se pudo abrir {name}: {exc}")
            if args.strict:
                return 2

    if missing_required:
        print("\nFaltan artefactos requeridos: " + ", ".join(missing_required))
        return 1 if args.strict else 0
    print("\nVerificación básica completada.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
