"""Small Transformer encoder for direct monthly demand forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from .core import LOOKBACK

if TYPE_CHECKING:
    from tensorflow import keras


@dataclass(frozen=True)
class TransformerConfig:
    epochs: int = 15
    width: int = 32
    heads: int = 4

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("Transformer epochs must be positive")
        if self.width < 1 or self.heads < 1 or self.width % self.heads:
            raise ValueError("Transformer width must be a positive multiple of the number of heads")


def _tokens(values: np.ndarray, months: np.ndarray) -> tuple[np.ndarray, float]:
    scale = max(float(values.mean()), 1.0)
    calendar = (months - 1) * (2 * np.pi / 12)
    positions = np.linspace(-1, 0, len(values))
    return np.column_stack((values / scale, np.sin(calendar), np.cos(calendar), positions)), scale


@dataclass
class TransformerForecaster:
    network: keras.Model
    trained_through: pd.Timestamp
    lookback: int

    def predict(self, history: pd.DataFrame, requests: pd.DataFrame) -> np.ndarray:
        """Predict one quantity per item and origin using only that origin's history."""
        if requests.empty:
            return np.empty(0, dtype=float)
        groups = {str(item): group.set_index("month")["quantity"] for item, group in history.groupby("item_id")}
        windows, scales = [], []
        for row in requests[["item_id", "origin_month"]].itertuples(index=False):
            origin = pd.Timestamp(row.origin_month)
            if origin < self.trained_through:
                raise ValueError("Forecast origin cannot precede the Transformer's training cutoff")
            series = groups[str(row.item_id)]
            known = series.loc[:origin].tail(self.lookback)
            if len(known) < self.lookback or known.index[-1] != origin:
                raise ValueError(f"Transformer needs {self.lookback} known months through {origin:%Y-%m} for {row.item_id}")
            window, scale = _tokens(known.to_numpy(dtype=float), known.index.month.to_numpy())
            windows.append(window)
            scales.append(scale)

        inputs = np.asarray(windows, dtype="float32")
        ratios = np.concatenate([
            self.network(inputs[start:start + 128], training=False).numpy().ravel()
            for start in range(0, len(inputs), 128)
        ])
        return np.maximum(0, ratios * np.asarray(scales))


def fit_transformer(history: pd.DataFrame, *, through: object, horizon: int,
                    config: TransformerConfig = TransformerConfig(),
                    lookback: int = LOOKBACK) -> TransformerForecaster:
    """Train on validated monthly history with target labels observed by the cutoff."""
    if not 1 <= horizon <= 12 or lookback < 2:
        raise ValueError("Transformer needs a horizon of 1–12 months and a lookback of at least 2 months")
    cutoff = pd.Timestamp(through)
    inputs, labels = [], []
    for _, group in history.loc[history["month"] <= cutoff].groupby("item_id", sort=True):
        values = group["quantity"].to_numpy(dtype=float)
        months = group["month"].dt.month.to_numpy()
        for end in range(lookback, len(values) - horizon + 1):
            window, scale = _tokens(values[end - lookback:end], months[end - lookback:end])
            inputs.append(window)
            labels.append(values[end - 1 + horizon] / scale)
    if len(inputs) < 24:
        raise ValueError(
            "Transformer needs at least 24 labeled training windows by the first forecast date. "
            "Add earlier history or items, or reduce the number of backtest dates or the forecast horizon."
        )

    try:
        import tensorflow as tf
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError('Transformer is optional. Install it with: python -m pip install -e ".[transformer]"') from exc

    keras.utils.set_random_seed(42)
    layers = keras.layers
    sequence = keras.Input(shape=(lookback, 4))
    projected = layers.Dense(config.width)(sequence)
    attention = layers.MultiHeadAttention(num_heads=config.heads, key_dim=config.width // config.heads)
    encoded = layers.LayerNormalization()(layers.Add()([projected, attention(projected, projected)]))
    feed_forward = layers.Dense(config.width * 2, activation="relu")(encoded)
    feed_forward = layers.Dense(config.width)(feed_forward)
    encoded = layers.LayerNormalization()(layers.Add()([encoded, feed_forward]))
    last_step = layers.Cropping1D(cropping=(lookback - 1, 0))(encoded)
    output = layers.Dense(1)(layers.Flatten()(last_step))
    network = keras.Model(sequence, output)
    network.compile(optimizer="adam", loss="mse")

    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    dataset = tf.data.Dataset.from_tensor_slices((
        np.asarray(inputs, dtype="float32"), np.asarray(labels, dtype="float32"),
    ))
    dataset = dataset.shuffle(min(2048, len(inputs)), seed=42).batch(32).with_options(options)
    network.fit(dataset, epochs=config.epochs, verbose=0)
    return TransformerForecaster(network, cutoff, lookback)
