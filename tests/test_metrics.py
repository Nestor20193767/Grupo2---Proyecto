from pathlib import Path

import pandas as pd


def test_mode_c_is_best_reported_mode():
    path = Path(__file__).resolve().parents[1] / "results" / "metrics" / "classifier_modes.csv"
    data = pd.read_csv(path).set_index("mode")
    assert data.loc["C", "recall_macro"] == data["recall_macro"].max()
    assert data.loc["C", "f1_macro"] == data["f1_macro"].max()
    assert data.loc["C", "accuracy"] == data["accuracy"].max()
