---
license: mit
library_name: keras
tags:
  - time-series
  - anomaly-detection
  - lstm
  - autoencoder
  - wti
  - crude-oil
---

# WTI LSTM Autoencoder

Unsupervised reconstruction-error model for daily WTI futures (`CL=F`). An LSTM autoencoder reconstructs 10-day windows of RobustScaler-transformed close prices. A window is flagged when its reconstruction MSE exceeds the 90th percentile of **training** MSE.

This card describes the **champion** stored in this repo. Training, weekly fine-tuning and the quality gate live in [DrAdrianDC/WTI_Anomaly_Detection](https://github.com/DrAdrianDC/WTI_Anomaly_Detection).

This is a **rarity score** over 10-day price shapes, not a crisis classifier and not a forecast.

## Files

| File | Role |
| --- | --- |
| `lstm_autoencoder.keras` | Keras 3 weights + graph |
| `scaler.pkl` | Frozen `RobustScaler` + lookback metadata |
| `metadata.json` | Threshold, MAE, `last_date`, window counts |
| `reconstruction_scores.csv` | Per-day Close, MSE, flag |
| `detected_anomalies.csv` | Flagged days only |
| `plot-anomalies.png` | Price + anomaly markers |
| `plot-reconstruction-error.png` | MSE versus threshold |

`scaler.pkl` is a joblib/pickle artifact. Load it only if you trust this repository.

## Contract

- Ticker: `CL=F` (unadjusted close, yfinance)
- Lookback: 10 trading days
- Input shape: `(batch, 10, 1)`
- Scaler: `RobustScaler`, fitted on the training split only; **never refit on fine-tune**
- Threshold: P90 of **original training** reconstruction MSE, **frozen** after that train (`0.0162`). Weekly fine-tune does not recompute it.
- Score ledger: append-only. New weights apply to **new dates only**. Historical rows are not rewritten.
- Quality gate (weekly): challenger hold-out MAE must be ≤ champion MAE × 1.10. Plots always render the full published ledger.

Replace the snapshot below with `metadata.json` after each promoted retrain.

## Champion snapshot

- `last_date`: 2026-09-04
- Epochs trained (EarlyStopping): 34 / max 100
- Train MAE / hold-out MAE: 0.072 / 0.055
- Threshold: 0.0162
- Flags: 623 / 6,528 windows (9.5%)
- Strongest event: 2020-04-20, close −37.63 USD, MSE 0.270

## Use

```python
import json
import numpy as np
import joblib
from huggingface_hub import hf_hub_download
from tensorflow import keras

repo_id = "DrAdrianDC/wti-lstm-autoencoder"

model = keras.models.load_model(hf_hub_download(repo_id, "lstm_autoencoder.keras"))
payload = joblib.load(hf_hub_download(repo_id, "scaler.pkl"))
scaler = payload["scaler"] if isinstance(payload, dict) else payload
lookback = int(payload.get("lookback", 10)) if isinstance(payload, dict) else 10

with open(hf_hub_download(repo_id, "metadata.json"), encoding="utf-8") as f:
    threshold = json.load(f)["threshold"]

# `close` is a 1-D array of unadjusted CL=F closes, oldest first.
close = np.asarray(close, dtype=np.float64).reshape(-1, 1)
scaled = scaler.transform(close).astype(np.float32)
windows = np.stack(
    [scaled[i : i + lookback] for i in range(len(scaled) - lookback + 1)],
    axis=0,
)
reconstructed = model.predict(windows, verbose=0)
mse = np.mean((reconstructed - windows) ** 2, axis=(1, 2))
flags = (mse > threshold).astype(int)
```

There is no Transformers `pipeline()` for this checkpoint. Score with Keras + the frozen scaler, then compare per-window MSE to `metadata.json["threshold"]`. The ranking of `mse` is more informative than the 0/1 flag.

## Limitations

- **P90 is a reporting quantile, not a rare-event detector.** About 9.5% of scored days sit above the threshold. The flag marks windows that were hard to reconstruct, which includes persistent volatility as well as crashes. Changing the cutoff is a policy change: set `model.frozen_threshold` and rerun `evaluate`. Do not let a weekly job move it.
- **No event labels.** There is no precision/recall. Known stress dates (GFC 2008, 2016 glut, OPEC+ March 2020, the 20 April 2020 negative print, March 2022) live in the published score ledger and are not rescored when weights update.
- **Not a forecast and not a trading signal.** The network reconstructs a window that already ended. It has no prediction horizon.
- **Univariate unadjusted close only.** No inventories, options, calendar spreads or OPEC events. A simple return z-score or Isolation Forest may also flag April 2020; this repo’s claim is a frozen feature space plus a weekly MAE gate, not superiority over classical baselines.
- **ReLU LSTMs with `clipnorm`.** Validation loss is noisy; EarlyStopping restored the best epoch (34 / 100). Changing activation, lookback or scaler type invalidates this champion.

## Intended use

Research and portfolio demonstration of an unsupervised energy time-series **monitoring** pipeline: frozen scaler, frozen decision threshold, weekly champion/challenger fine-tune, append-only scores (new dates only), and full-history plots from that ledger. Not intended for automated trading, credit decisions or alerting without a separately reviewed policy and a human in the loop.
