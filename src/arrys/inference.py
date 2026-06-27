"""Inferencia reproducible del decoder TCN-cVAE y clasificador TCN ONNX."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import onnxruntime as ort

from .constants import CLASSIFIER_CLASSES, GENERATOR_CLASSES, LATENT_DIM, SEQ_LEN


@dataclass(frozen=True)
class ModelPaths:
    model_dir: Path
    decoder: Path
    classifier: Path
    latent_bank: Path
    label_encoder: Path

    @classmethod
    def from_directory(cls, model_dir: str | Path) -> "ModelPaths":
        base = Path(model_dir).expanduser().resolve()
        return cls(
            model_dir=base,
            decoder=base / "tcncvae_decoder_physionet.onnx",
            classifier=base / "clf_aug_physionet.onnx",
            latent_bank=base / "latent_bank.npz",
            label_encoder=base / "label_encoder_physionet.pkl",
        )


class ArrysInference:
    """Carga los artefactos finales y genera/clasifica latidos ECG.

    Parameters
    ----------
    model_dir:
        Directorio que contiene los modelos ONNX y sus archivos ``.onnx.data``.
    providers:
        Proveedores de ONNX Runtime. Por defecto se usa CPU para máxima
        reproducibilidad.
    """

    def __init__(
        self,
        model_dir: str | Path = "models",
        providers: Sequence[str] = ("CPUExecutionProvider",),
    ) -> None:
        self.paths = ModelPaths.from_directory(model_dir)
        self.providers = list(providers)
        self._decoder: ort.InferenceSession | None = None
        self._classifier: ort.InferenceSession | None = None
        self._latent_bank: dict[str, np.ndarray] | None = None
        self._classifier_classes: tuple[str, ...] | None = None

    @property
    def decoder(self) -> ort.InferenceSession:
        if self._decoder is None:
            self._require(self.paths.decoder)
            self._decoder = ort.InferenceSession(
                str(self.paths.decoder), providers=self.providers
            )
        return self._decoder

    @property
    def classifier(self) -> ort.InferenceSession:
        if self._classifier is None:
            self._require(self.paths.classifier)
            self._classifier = ort.InferenceSession(
                str(self.paths.classifier), providers=self.providers
            )
        return self._classifier

    @property
    def classifier_classes(self) -> tuple[str, ...]:
        if self._classifier_classes is None:
            classes: tuple[str, ...] = CLASSIFIER_CLASSES
            if self.paths.label_encoder.exists():
                try:
                    encoder = joblib.load(self.paths.label_encoder)
                    loaded = tuple(str(v) for v in encoder.classes_)
                    if loaded:
                        classes = loaded
                except Exception:
                    pass
            self._classifier_classes = classes
        return self._classifier_classes

    def sample_latent(
        self,
        class_name: str,
        n: int = 1,
        noise: float = 0.8,
        seed: int | None = 42,
    ) -> np.ndarray:
        """Muestrea ``z`` a partir del banco latente, con fallback normal."""
        self._validate_class(class_name)
        if n < 1:
            raise ValueError("n debe ser mayor o igual a 1")
        if noise < 0:
            raise ValueError("noise no puede ser negativo")

        rng = np.random.default_rng(seed)
        bank = self._load_latent_bank()
        class_index = GENERATOR_CLASSES.index(class_name)

        mu = self._find_array(bank, f"mu_{class_index}", f"mu_{class_name}")
        lv = self._find_array(
            bank,
            f"lv_{class_index}",
            f"log_var_{class_index}",
            f"lv_{class_name}",
            f"log_var_{class_name}",
        )

        if mu is None:
            return (rng.standard_normal((n, LATENT_DIM)) * noise).astype(np.float32)

        mu = np.asarray(mu, dtype=np.float32).reshape(-1, LATENT_DIM)
        indices = rng.integers(0, len(mu), size=n)
        selected_mu = mu[indices]

        if lv is None:
            scale = np.ones_like(selected_mu, dtype=np.float32)
        else:
            lv = np.asarray(lv, dtype=np.float32).reshape(-1, LATENT_DIM)
            scale = np.exp(0.5 * lv[indices])

        eps = rng.standard_normal((n, LATENT_DIM)).astype(np.float32)
        return (selected_mu + noise * scale * eps).astype(np.float32)

    def generate(
        self,
        class_name: str,
        n: int = 1,
        noise: float = 0.8,
        seed: int | None = 42,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Genera latidos y devuelve ``(beats, z)`` con shape ``(n, 325)``."""
        z = self.sample_latent(class_name, n=n, noise=noise, seed=seed)
        condition = np.zeros((n, len(GENERATOR_CLASSES)), dtype=np.float32)
        condition[:, GENERATOR_CLASSES.index(class_name)] = 1.0

        input_names = {item.name for item in self.decoder.get_inputs()}
        feed = {}
        if "z" in input_names:
            feed["z"] = z
        else:
            feed[self.decoder.get_inputs()[0].name] = z
        condition_name = "condition" if "condition" in input_names else self.decoder.get_inputs()[1].name
        feed[condition_name] = condition

        output = np.asarray(self.decoder.run(None, feed)[0], dtype=np.float32)
        beats = output[:, 0, :] if output.ndim == 3 else output.reshape(n, SEQ_LEN)
        return beats, z

    def classify(self, beats: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Clasifica latidos normalizados por muestra y retorna probabilidades."""
        array = np.asarray(beats, dtype=np.float32)
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != SEQ_LEN:
            raise ValueError(f"Se esperaba shape (n, {SEQ_LEN}); se recibió {array.shape}")

        mean = array.mean(axis=1, keepdims=True)
        std = array.std(axis=1, keepdims=True) + 1e-8
        signal = ((array - mean) / std)[:, None, :].astype(np.float32)
        input_name = self.classifier.get_inputs()[0].name
        logits = np.asarray(self.classifier.run(None, {input_name: signal})[0])
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        predictions = probabilities.argmax(axis=1)

        classes = list(self.classifier_classes)
        if len(classes) != probabilities.shape[1]:
            classes = [f"Class_{i}" for i in range(probabilities.shape[1])]
        labels = [classes[index] for index in predictions]
        return probabilities, predictions, labels

    def _load_latent_bank(self) -> dict[str, np.ndarray]:
        if self._latent_bank is None:
            if self.paths.latent_bank.exists():
                with np.load(self.paths.latent_bank, allow_pickle=False) as data:
                    self._latent_bank = {key: data[key] for key in data.files}
            else:
                self._latent_bank = {}
        return self._latent_bank

    @staticmethod
    def _find_array(bank: dict[str, np.ndarray], *keys: str) -> np.ndarray | None:
        for key in keys:
            if key in bank:
                return bank[key]
        return None

    @staticmethod
    def _require(path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(
                f"No se encontró {path}. Ejecuta scripts/migrate_repository.py "
                "o copia los artefactos descritos en models/README.md."
            )

    @staticmethod
    def _validate_class(class_name: str) -> None:
        if class_name not in GENERATOR_CLASSES:
            valid = ", ".join(GENERATOR_CLASSES)
            raise ValueError(f"Clase inválida: {class_name}. Opciones: {valid}")
