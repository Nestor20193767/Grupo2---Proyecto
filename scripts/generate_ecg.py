#!/usr/bin/env python3
"""Genera señales ECG sintéticas desde la línea de comandos."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from arrys import ArrysInference, GENERATOR_CLASSES, FS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--class-name", choices=GENERATOR_CLASSES, default="NSR")
    parser.add_argument("--n", type=int, default=8, help="Número de latidos")
    parser.add_argument("--noise", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--classify", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    engine = ArrysInference(args.model_dir)
    beats, z = engine.generate(args.class_name, args.n, args.noise, args.seed)

    signal_path = args.output_dir / f"ecg_{args.class_name}_{args.n}_seed{args.seed}.csv"
    latent_path = args.output_dir / f"latent_{args.class_name}_{args.n}_seed{args.seed}.csv"
    pd.DataFrame(beats, columns=[f"t{i}" for i in range(beats.shape[1])]).assign(
        class_name=args.class_name
    ).to_csv(signal_path, index=False)
    pd.DataFrame(z, columns=[f"z{i}" for i in range(z.shape[1])]).to_csv(latent_path, index=False)

    time_ms = np.arange(beats.shape[1]) / FS * 1000
    fig, ax = plt.subplots(figsize=(10, 4))
    for beat in beats[: min(8, len(beats))]:
        ax.plot(time_ms, beat, alpha=0.35)
    ax.plot(time_ms, beats.mean(axis=0), linewidth=2, label="Media")
    ax.set(title=f"ECG sintético — {args.class_name}", xlabel="Tiempo (ms)", ylabel="Amplitud")
    ax.legend()
    fig.tight_layout()
    figure_path = args.output_dir / f"ecg_{args.class_name}_{args.n}_seed{args.seed}.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    print(f"Señales: {signal_path}")
    print(f"Latentes: {latent_path}")
    print(f"Figura:   {figure_path}")

    if args.classify:
        probabilities, _, labels = engine.classify(beats)
        result = pd.DataFrame(probabilities)
        result.insert(0, "prediction", labels)
        result.to_csv(args.output_dir / "classifier_predictions.csv", index=False)
        print("Predicciones guardadas en outputs/classifier_predictions.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
