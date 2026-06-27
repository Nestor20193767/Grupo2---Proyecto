#!/usr/bin/env python3
"""Regenera figuras resumidas a partir de los CSV versionados en results/metrics."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "metrics"
OUTPUT = ROOT / "results" / "reproduced_figures"


def save_classifier_modes() -> None:
    data = pd.read_csv(METRICS / "classifier_modes.csv").set_index("mode")
    ax = data[["accuracy", "recall_macro", "f1_macro"]].plot(kind="bar", figsize=(9, 5))
    ax.set(ylabel="Métrica (%)", title="Clasificador TCN por modo")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT / "classifier_modes.png", dpi=180)
    plt.close(ax.figure)


def save_f1_by_class() -> None:
    data = pd.read_csv(METRICS / "f1_by_class.csv").set_index("class")
    ax = data.plot(kind="bar", figsize=(8, 5))
    ax.set(ylabel="F1-score (%)", title="F1 por clase: real vs real+sintético")
    ax.tick_params(axis="x", rotation=0)
    ax.figure.tight_layout()
    ax.figure.savefig(OUTPUT / "f1_by_class.png", dpi=180)
    plt.close(ax.figure)


def save_generator_summary() -> None:
    data = pd.read_csv(METRICS / "generator_summary.csv")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(data["metric"], data["value"])
    ax.set(title="Resumen cuantitativo del generador", ylabel="Valor reportado")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUTPUT / "generator_summary.png", dpi=180)
    plt.close(fig)


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save_classifier_modes()
    save_f1_by_class()
    save_generator_summary()
    print(f"Figuras reproducidas en {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
