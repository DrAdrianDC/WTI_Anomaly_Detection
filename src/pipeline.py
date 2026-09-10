#!/usr/bin/env python3
"""Production orchestrator for the WTI LSTM autoencoder.

Execution modes
---------------
``python src/pipeline.py --mode train``
    Full-history download, scaler fit on the training split, train from
    scratch, persist ``.keras`` + ``scaler.pkl`` + ``metadata.json`` plus
    the CSV reports and portfolio plots, and optionally publish binaries
    to the Hub.

``python src/pipeline.py --mode retrain``
    Load the champion weights, fine-tune on the latest year, and promote
    only if hold-out MAE does not degrade beyond the configured margin.
    The decision threshold stays frozen. New dates are **appended** to the
    published score ledger; historical rows are never recomputed. Plots use
    the full ledger.

``python src/pipeline.py --mode evaluate``
    Append newly available dates to the published ledger with the frozen
    threshold. Does not rewrite the past and does not train.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import AppConfig, load_config  # noqa: E402
from src.data_loader import (  # noqa: E402
    DataDownloadError,
    DataValidationError,
    download_wti_prices,
    last_available_date,
)
from src.model import LSTMAutoencoder, keras_model_to_dict, set_global_seeds  # noqa: E402
from src.plots import write_result_plots  # noqa: E402
from src.preprocessor import TimeSeriesPreprocessor  # noqa: E402

logger = logging.getLogger("wti.pipeline")

HF_TOKEN_ENV = "HF_TOKEN"


class PipelineError(RuntimeError):
    """Raised when a train/retrain run cannot finish safely."""


class QualityGateRejected(RuntimeError):
    """Raised when the challenger fails the MAE degradation gate."""


def configure_logging(verbose: bool = False) -> None:
    """Configure process-wide logging for local runs and GitHub Actions."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    logging.getLogger("tensorflow").setLevel(logging.ERROR)
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or weekly-fine-tune the WTI LSTM autoencoder.",
    )
    parser.add_argument(
        "--mode",
        choices=("train", "retrain", "evaluate"),
        required=True,
        help="train: fit from scratch on full history. "
        "retrain: fine-tune the champion on the latest window. "
        "evaluate: append new dates to the published ledger; do not rewrite history.",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional path to config.yaml (defaults to the repo-level file).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training/retrain epochs from config.yaml.",
    )
    parser.add_argument(
        "--skip-upload",
        action="store_true",
        help="Do not push artifacts to the Hugging Face Hub.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args(argv)


def _iso_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _read_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PipelineError(f"Corrupt metadata file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PipelineError(f"Metadata at {path} must be a JSON object.")
    return payload


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    logger.info("Wrote metadata to %s", path)


def _decision_threshold(config: AppConfig, metadata: dict[str, Any]) -> float:
    """Return the locked decision policy.

    Weekly fine-tune may update weights. It must not move the cutoff. Prefer
    ``config.model.frozen_threshold`` (policy as code) over Hub metadata, which
    can be stale after a bad promote.
    """
    if config.model.frozen_threshold is not None:
        return float(config.model.frozen_threshold)
    raw = metadata.get("threshold")
    if raw is None:
        raise PipelineError(
            "No frozen decision threshold. Set model.frozen_threshold in "
            "config.yaml or run --mode train once."
        )
    return float(raw)


def _score_full_series(
    config: AppConfig,
    autoencoder: LSTMAutoencoder,
    preprocessor: TimeSeriesPreprocessor,
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """Per-day reconstruction MSE and frozen-threshold flags for ``frame``."""
    date_col = config.preprocessing.date_column
    target_col = config.preprocessing.target_column
    ordered = frame.sort_values(date_col).reset_index(drop=True)
    sequences = preprocessor.transform(ordered)
    errors = autoencoder.reconstruction_errors(sequences, metric="mse")
    if autoencoder.threshold_ is None:
        raise PipelineError("Decision threshold is not set; cannot flag windows.")
    flags = (errors > autoencoder.threshold_).astype(int)
    offset = preprocessor.lookback - 1
    n_windows = len(errors)
    return pd.DataFrame(
        {
            "Date": pd.to_datetime(
                ordered[date_col].iloc[offset : offset + n_windows]
            ).dt.strftime("%Y-%m-%d"),
            "Close": ordered[target_col].iloc[offset : offset + n_windows].to_numpy(),
            "reconstruction_mse": errors,
            "anomaly": flags.astype(int),
        }
    )


def _load_score_ledger(config: AppConfig) -> pd.DataFrame:
    """Load the published score history. This file is append-only after train."""
    path = config.output.scores_path
    if not path.exists():
        raise PipelineError(
            f"Score ledger missing at {path}. Run --mode train once to create it. "
            "Retrain must not rewrite history from scratch."
        )
    ledger = pd.read_csv(path)
    required = {"Date", "Close", "reconstruction_mse", "anomaly"}
    missing = required - set(ledger.columns)
    if missing:
        raise PipelineError(f"Score ledger is missing columns: {sorted(missing)}")
    ledger["Date"] = pd.to_datetime(ledger["Date"]).dt.strftime("%Y-%m-%d")
    ledger = ledger.drop_duplicates(subset=["Date"], keep="first")
    ledger = ledger.sort_values("Date").reset_index(drop=True)
    if ledger.empty:
        raise PipelineError("Score ledger is empty.")
    return ledger


def _assert_ledger_events(ledger: pd.DataFrame, event_dates: tuple[str, ...]) -> None:
    """Historical stress prints are a published record. They must stay in the ledger."""
    if not event_dates:
        return
    indexed = ledger.set_index("Date")
    missing = [day for day in event_dates if day not in indexed.index]
    dropped = [
        day
        for day in event_dates
        if day in indexed.index and int(indexed.loc[day, "anomaly"]) != 1
    ]
    if missing or dropped:
        parts = []
        if missing:
            parts.append(f"missing from ledger {missing}")
        if dropped:
            parts.append(f"unflagged in ledger {dropped}")
        raise PipelineError(
            "Score ledger failed historical event integrity ("
            + "; ".join(parts)
            + ")."
        )


def _append_forward_scores(
    config: AppConfig,
    autoencoder: LSTMAutoencoder,
    preprocessor: TimeSeriesPreprocessor,
    ledger: pd.DataFrame,
) -> pd.DataFrame:
    """Score only dates after the ledger cutoff. Do not recompute the past.

    The new champion is valid going forward. Published MSE/flags on historical
    dates stay as they were written at train (or at the day they were appended).
    """
    last_scored = pd.to_datetime(ledger["Date"]).max()
    calendar_buffer = max(40, int(preprocessor.lookback) * 4)
    start = (last_scored - pd.Timedelta(days=calendar_buffer)).strftime("%Y-%m-%d")
    logger.info(
        "Scoring forward from %s (ledger ends %s).",
        start,
        last_scored.strftime("%Y-%m-%d"),
    )
    prices = _download(config, start=start)
    scored = _score_full_series(config, autoencoder, preprocessor, prices)
    new_rows = scored.loc[pd.to_datetime(scored["Date"]) > last_scored].copy()
    if new_rows.empty:
        logger.info("No new trading dates after %s.", last_scored.strftime("%Y-%m-%d"))
        return ledger.copy()

    combined = pd.concat([ledger, new_rows], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Date"], keep="first")
    combined = combined.sort_values("Date").reset_index(drop=True)
    logger.info(
        "Appended %s new dates (%s → %s). Historical rows left unchanged.",
        len(new_rows),
        new_rows["Date"].iloc[0],
        new_rows["Date"].iloc[-1],
    )
    return combined


def _write_score_artifacts(
    config: AppConfig,
    scores: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    history: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Write CSV + plots from a score ledger. Does not rescore."""
    config.ensure_output_dir()
    ordered = scores.sort_values("Date").reset_index(drop=True)
    ordered["Date"] = pd.to_datetime(ordered["Date"]).dt.strftime("%Y-%m-%d")
    detected = ordered.loc[ordered["anomaly"] == 1].copy()

    ordered.to_csv(config.output.scores_path, index=False)
    detected.to_csv(config.output.anomalies_path, index=False)
    if history:
        pd.DataFrame(history).to_csv(config.output.history_path, index=False)

    metadata["n_windows_scored"] = int(len(ordered))
    metadata["n_anomalies"] = int(detected.shape[0])
    metadata["anomaly_rate"] = float(detected.shape[0] / len(ordered)) if len(ordered) else 0.0
    if not detected.empty:
        metadata["first_anomaly_date"] = str(detected["Date"].iloc[0])
        metadata["last_anomaly_date"] = str(detected["Date"].iloc[-1])
        peak_i = detected["reconstruction_mse"].astype(float).idxmax()
        metadata["max_anomaly_mse"] = float(detected.loc[peak_i, "reconstruction_mse"])
        metadata["max_anomaly_date"] = str(detected.loc[peak_i, "Date"])

    plots = write_result_plots(
        ordered,
        price_plot=config.output.price_plot_path,
        anomalies_plot=config.output.anomalies_plot_path,
        reconstruction_plot=config.output.reconstruction_plot_path,
        loss_plot=config.output.loss_plot_path,
        anomalies_html=config.output.anomalies_html_path,
        reconstruction_html=config.output.reconstruction_html_path,
        threshold=metadata.get("threshold"),
        history=history,
    )
    metadata["plots"] = {key: Path(value).name for key, value in plots.items()}
    logger.info(
        "Wrote %s (%s anomalies) covering %s → %s.",
        config.output.scores_path.name,
        metadata["n_anomalies"],
        ordered["Date"].iloc[0],
        ordered["Date"].iloc[-1],
    )
    return ordered


def _build_autoencoder(config: AppConfig) -> LSTMAutoencoder:
    return LSTMAutoencoder(
        lookback=config.preprocessing.lookback,
        n_features=1,
        encoder_units=config.model.encoder_units,
        decoder_units=config.model.decoder_units,
        dropout=config.model.dropout,
        activation=config.model.activation,
        loss=config.model.loss,
        optimizer=config.model.optimizer,
        clipnorm=config.model.clipnorm,
        threshold_percentile=config.model.threshold_percentile,
    )


def _download(
    config: AppConfig,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    data_cfg = config.data
    return download_wti_prices(
        ticker=data_cfg.ticker,
        start=start if start is not None else data_cfg.start_date,
        end=end if end is not None else data_cfg.end_date,
        period=data_cfg.period,
        auto_adjust=data_cfg.auto_adjust,
        max_null_ratio=data_cfg.max_null_ratio,
        retries=data_cfg.download_retries,
        retry_backoff_seconds=data_cfg.retry_backoff_seconds,
    )


def _hf_token() -> str | None:
    token = os.environ.get(HF_TOKEN_ENV, "").strip()
    return token or None


def pull_artifacts_from_hub(config: AppConfig) -> bool:
    """Download champion artifacts from Hugging Face when they are missing locally.

    Returns
    -------
    bool
        ``True`` if at least the model file is present afterwards.
    """
    model_path = config.output.model_path
    if model_path.exists() and config.output.scaler_path.exists():
        return True

    token = _hf_token()
    if token is None:
        logger.warning(
            "Local artifacts missing and %s is unset; cannot pull from the Hub.",
            HF_TOKEN_ENV,
        )
        return model_path.exists()

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise PipelineError(
            "huggingface_hub is required to restore artifacts from the Hub."
        ) from exc

    config.ensure_output_dir()
    filenames = [
        config.output.model_path.name,
        config.output.scaler_path.name,
        config.output.metadata_path.name,
    ]
    # Do not pull CSV/plots from the Hub. Those are the append-only score
    # ledger (usually committed in git). Overwriting them would let a stale
    # Hub promote erase published history.
    recovered = False
    for filename in filenames:
        try:
            hf_hub_download(
                repo_id=config.huggingface.repo_id,
                repo_type=config.huggingface.repo_type,
                filename=filename,
                local_dir=str(config.output.directory),
                token=token,
            )
            recovered = True
            logger.info("Restored %s from Hugging Face.", filename)
        except Exception as exc:
            logger.warning("Could not download %s from the Hub: %s", filename, exc)
    return recovered and model_path.exists()


def push_artifacts_to_hub(config: AppConfig) -> None:
    """Upload model, scaler and metadata to the Hugging Face Models hub.

    The upload is non-interactive: it uses ``HF_TOKEN`` and never prompts
    for credentials. Progress bars are disabled so CI logs stay quiet.
    """
    token = _hf_token()
    if token is None:
        logger.warning("Skipping Hub upload: environment variable %s is not set.", HF_TOKEN_ENV)
        return

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise PipelineError(
            "huggingface_hub is required to publish artifacts."
        ) from exc

    api = HfApi(token=token)
    repo_id = config.huggingface.repo_id
    api.create_repo(
        repo_id=repo_id,
        repo_type=config.huggingface.repo_type,
        private=config.huggingface.private,
        exist_ok=True,
        token=token,
    )

    artifacts = [
        config.output.model_path,
        config.output.scaler_path,
        config.output.metadata_path,
        config.output.scores_path,
        config.output.anomalies_path,
        config.output.anomalies_plot_path,
        config.output.price_plot_path,
        config.output.reconstruction_plot_path,
    ]
    for artifact in artifacts:
        if not artifact.exists():
            raise PipelineError(f"Cannot upload missing artifact: {artifact}")
        api.upload_file(
            path_or_fileobj=str(artifact),
            path_in_repo=artifact.name,
            repo_id=repo_id,
            repo_type=config.huggingface.repo_type,
            token=token,
        )
        logger.info("Uploaded %s → hf://%s/%s", artifact.name, repo_id, artifact.name)


def export_detection_report(
    config: AppConfig,
    autoencoder: LSTMAutoencoder,
    preprocessor: TimeSeriesPreprocessor,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    history: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Score the full series and write human-readable CSV reports.

    Used by ``train`` only: that run *creates* the published ledger.
    Retrain and evaluate append new dates and must not call this.
    """
    scores = _score_full_series(config, autoencoder, preprocessor, frame)
    return _write_score_artifacts(config, scores, metadata, history=history)


def _persist(
    config: AppConfig,
    autoencoder: LSTMAutoencoder,
    preprocessor: TimeSeriesPreprocessor,
    frame: pd.DataFrame,
    metadata: dict[str, Any],
    *,
    upload: bool,
    save_model: bool = True,
    history: dict[str, Any] | None = None,
) -> None:
    config.ensure_output_dir()
    if save_model:
        autoencoder.save(config.output.model_path)
        preprocessor.save_scaler(config.output.scaler_path)
    export_detection_report(
        config,
        autoencoder,
        preprocessor,
        frame,
        metadata,
        history=history,
    )
    _write_metadata(config.output.metadata_path, metadata)
    if upload:
        push_artifacts_to_hub(config)


def run_train(
    config: AppConfig,
    *,
    upload: bool,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Train a new autoencoder on the full configured history."""
    set_global_seeds(config.training.seed)
    frame = _download(config)

    preprocessor = TimeSeriesPreprocessor(
        lookback=config.preprocessing.lookback,
        scaler_type=config.preprocessing.scaler_type,
        target_column=config.preprocessing.target_column,
        date_column=config.preprocessing.date_column,
    )
    x_train, x_val = preprocessor.prepare_train_val(
        frame,
        train_end_date=config.data.train_end_date,
        validation_fraction=config.data.validation_fraction,
        fit_scaler=True,
    )

    autoencoder = _build_autoencoder(config)
    autoencoder.build()
    history = autoencoder.fit(
        x_train,
        x_val,
        epochs=epochs if epochs is not None else config.training.epochs,
        batch_size=config.training.batch_size,
        patience=config.training.patience,
        shuffle=config.training.shuffle,
    )

    val_mae = autoencoder.mean_absolute_error(x_val)
    train_mae = autoencoder.mean_absolute_error(x_train)
    if config.model.frozen_threshold is not None:
        threshold = float(config.model.frozen_threshold)
        autoencoder.threshold_ = threshold
        logger.info("Using frozen decision threshold %.6f (policy as code).", threshold)
    else:
        threshold = autoencoder.fit_threshold(x_train)

    metadata = {
        "mode": "train",
        "ticker": config.data.ticker,
        "last_date": last_available_date(frame).strftime("%Y-%m-%d"),
        "lookback": config.preprocessing.lookback,
        "scaler_type": config.preprocessing.scaler_type,
        "threshold": threshold,
        "threshold_frozen": True,
        "threshold_percentile": config.model.threshold_percentile,
        "train_mae": train_mae,
        "val_mae": val_mae,
        "n_train_windows": int(x_train.shape[0]),
        "n_val_windows": int(x_val.shape[0]),
        "best_val_loss": float(min(history.history.get("val_loss", [float("nan")]))),
        "epochs_trained": len(history.history.get("loss", [])),
        "trained_at": _iso_now(),
        "model": keras_model_to_dict(autoencoder.model),
    }
    _persist(
        config,
        autoencoder,
        preprocessor,
        frame,
        metadata,
        upload=upload,
        history=history.history,
    )
    logger.info(
        "Training complete. val_mae=%.6f threshold=%.6f last_date=%s",
        val_mae,
        threshold,
        metadata["last_date"],
    )
    return metadata


def _resolve_retrain_start(metadata: dict[str, Any], context_days: int) -> str | None:
    last_date = metadata.get("last_date")
    if not last_date:
        return None
    last = pd.Timestamp(str(last_date))
    start = last - pd.Timedelta(days=int(context_days))
    return start.strftime("%Y-%m-%d")


def run_retrain(
    config: AppConfig,
    *,
    upload: bool,
    epochs: int | None = None,
) -> dict[str, Any]:
    """Fine-tune the champion on the latest window with a quality gate.

    Champion/challenger protocol
    ----------------------------
    1. Restore local artifacts, or pull them from the Hub.
    2. If no champion exists, fall back to a full ``train`` run.
    3. Download prices from ``last_date - context_days`` so windows remain
       continuous, then split the recent frame chronologically.
    4. Score the *current* champion on the hold-out (baseline MAE).
    5. Fine-tune a few epochs. Score the challenger on the same hold-out.
    6. Promote only if
       ``challenger_mae <= baseline_mae * (1 + max_degradation_ratio)``.
    7. Keep the frozen decision threshold.
    8. Append scores **only for dates after the published ledger**.
       Historical rows (2008, 2020, …) are not recomputed.
    9. Plots are drawn from the full ledger (frozen past + new tail).
    """
    set_global_seeds(config.training.seed)
    config.ensure_output_dir()

    if not config.output.model_path.exists():
        logger.info("Champion model missing locally; attempting Hub restore.")
        pull_artifacts_from_hub(config)

    if not config.output.model_path.exists():
        logger.warning("No champion available. Bootstrapping with a full train run.")
        return run_train(config, upload=upload, epochs=epochs)

    if not config.output.scaler_path.exists():
        raise PipelineError(
            f"Found {config.output.model_path} but scaler is missing. "
            "Refusing to fine-tune in a different feature space."
        )

    metadata = _read_metadata(config.output.metadata_path)
    start = _resolve_retrain_start(metadata, config.retrain.context_days)
    logger.info("Retrain download window starts at %s (None = full history).", start)
    frame = _download(config, start=start)

    preprocessor = TimeSeriesPreprocessor.load_scaler(config.output.scaler_path)
    preprocessor.lookback = config.preprocessing.lookback

    x_tune, x_holdout = preprocessor.prepare_train_val(
        frame,
        train_end_date=None,
        validation_fraction=config.data.validation_fraction,
        fit_scaler=False,
    )
    if x_tune.shape[0] < config.retrain.min_sequences:
        raise PipelineError(
            f"Not enough fine-tune windows ({x_tune.shape[0]}) "
            f"for min_sequences={config.retrain.min_sequences}."
        )
    if x_holdout.shape[0] < max(8, config.retrain.min_sequences // 4):
        raise PipelineError(
            f"Hold-out is too small ({x_holdout.shape[0]} windows) "
            "to run a reliable quality gate."
        )

    champion = LSTMAutoencoder.load(
        config.output.model_path,
        threshold_percentile=config.model.threshold_percentile,
        threshold=metadata.get("threshold"),
    )
    baseline_mae = champion.mean_absolute_error(x_holdout)
    logger.info("Champion hold-out MAE (baseline) = %.6f", baseline_mae)

    history = champion.fit(
        x_tune,
        x_holdout,
        epochs=epochs if epochs is not None else config.retrain.epochs,
        batch_size=config.training.batch_size,
        patience=config.retrain.patience,
        shuffle=config.training.shuffle,
    )
    challenger_mae = champion.mean_absolute_error(x_holdout)
    allowed = baseline_mae * (1.0 + config.retrain.max_degradation_ratio)
    logger.info(
        "Challenger hold-out MAE = %.6f (allowed <= %.6f, margin=%.1f%%).",
        challenger_mae,
        allowed,
        config.retrain.max_degradation_ratio * 100.0,
    )

    if challenger_mae > allowed:
        raise QualityGateRejected(
            f"Quality gate rejected challenger: MAE {challenger_mae:.6f} "
            f"> allowed {allowed:.6f}. Champion artifacts were not overwritten."
        )

    threshold = _decision_threshold(config, metadata)
    champion.threshold_ = threshold
    logger.info("Decision threshold frozen at %.6f.", threshold)

    ledger = _load_score_ledger(config)
    _assert_ledger_events(ledger, config.retrain.acceptance_event_dates)
    combined = _append_forward_scores(config, champion, preprocessor, ledger)

    updated = {
        **metadata,
        "mode": "retrain",
        "ticker": config.data.ticker,
        "last_date": str(combined["Date"].iloc[-1]),
        "lookback": config.preprocessing.lookback,
        "scaler_type": preprocessor.scaler_type,
        "threshold": threshold,
        "threshold_frozen": True,
        "threshold_percentile": config.model.threshold_percentile,
        "val_mae": challenger_mae,
        "baseline_mae": baseline_mae,
        "n_tune_windows": int(x_tune.shape[0]),
        "n_holdout_windows": int(x_holdout.shape[0]),
        "best_val_loss": float(min(history.history.get("val_loss", [float("nan")]))),
        "epochs_trained": len(history.history.get("loss", [])),
        "trained_at": _iso_now(),
        "gate_passed": True,
        "scores_mode": "append",
        "model": keras_model_to_dict(champion.model),
    }
    config.ensure_output_dir()
    champion.save(config.output.model_path)
    preprocessor.save_scaler(config.output.scaler_path)
    _write_score_artifacts(config, combined, updated, history=history.history)
    _write_metadata(config.output.metadata_path, updated)
    if upload:
        push_artifacts_to_hub(config)
    logger.info(
        "Retrain promoted. val_mae=%.6f (was %.6f) last_date=%s",
        challenger_mae,
        baseline_mae,
        updated["last_date"],
    )
    return updated


def run_evaluate(config: AppConfig, *, upload: bool = False) -> dict[str, Any]:
    """Append newly available dates to the published ledger. Do not rescore the past."""
    if not config.output.model_path.exists() or not config.output.scaler_path.exists():
        raise PipelineError(
            "evaluate requires a saved model and scaler. "
            "Run `python src/pipeline.py --mode train` first."
        )

    preprocessor = TimeSeriesPreprocessor.load_scaler(config.output.scaler_path)
    preprocessor.lookback = config.preprocessing.lookback
    metadata = _read_metadata(config.output.metadata_path)
    threshold = _decision_threshold(config, metadata)
    autoencoder = LSTMAutoencoder.load(
        config.output.model_path,
        threshold_percentile=config.model.threshold_percentile,
        threshold=threshold,
    )

    ledger = _load_score_ledger(config)
    _assert_ledger_events(ledger, config.retrain.acceptance_event_dates)
    combined = _append_forward_scores(config, autoencoder, preprocessor, ledger)

    metadata["mode"] = "evaluate"
    metadata["threshold"] = threshold
    metadata["threshold_frozen"] = True
    metadata["scores_mode"] = "append"
    metadata["last_date"] = str(combined["Date"].iloc[-1])
    metadata["evaluated_at"] = _iso_now()
    history = None
    if config.output.history_path.exists():
        history = pd.read_csv(config.output.history_path).to_dict(orient="list")
    _write_score_artifacts(config, combined, metadata, history=history)
    _write_metadata(config.output.metadata_path, metadata)
    if upload:
        push_artifacts_to_hub(config)
    logger.info(
        "Evaluation complete. n_anomalies=%s last_date=%s",
        metadata.get("n_anomalies"),
        metadata["last_date"],
    )
    return metadata


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(verbose=args.verbose)

    try:
        config = load_config(args.config)
    except Exception as exc:
        logger.error("Failed to load configuration: %s", exc)
        return 2

    logger.info("Starting pipeline mode=%s config=%s", args.mode, config.source_path)

    try:
        if args.mode == "train":
            run_train(config, upload=not args.skip_upload, epochs=args.epochs)
        elif args.mode == "retrain":
            run_retrain(config, upload=not args.skip_upload, epochs=args.epochs)
        else:
            run_evaluate(config, upload=not args.skip_upload)
    except QualityGateRejected as exc:
        # A rejected challenger is an expected weekly outcome, not a CI outage.
        logger.warning("%s", exc)
        return 0
    except (DataValidationError, DataDownloadError, PipelineError) as exc:
        logger.error("Pipeline failed: %s", exc)
        return 1
    except Exception as exc:
        logger.exception("Unexpected pipeline failure: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
