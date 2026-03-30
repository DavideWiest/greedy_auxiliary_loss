from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import pandas as pd
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from greedy_auxiliary_loss.config import RunConfig
from greedy_auxiliary_loss.data import build_datamodule
from greedy_auxiliary_loss.module import ClassificationModule
from greedy_auxiliary_loss.utils.repro import seed_everything
from greedy_auxiliary_loss.utils.results import append_csv_row, write_json


def _safe_metric(metrics: list[dict[str, Any]], key: str) -> float:
    if not metrics or key not in metrics[0]:
        return float("nan")
    value = metrics[0][key]
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def run_experiment(config: RunConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    torch.set_float32_matmul_precision("high")

    run_dir = Path(config.output_dir) / "runs" / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", config.to_dict())

    datamodule = build_datamodule(config.dataset)
    datamodule.prepare_data()
    datamodule.setup("fit")
    model = ClassificationModule(config=config, dataset_metadata=datamodule.metadata)

    checkpoint = ModelCheckpoint(
        dirpath=Path(config.output_dir) / "checkpoints" / config.run_name,
        filename="best",
        monitor="val/acc",
        mode="max",
        save_top_k=1,
    )
    logger = CSVLogger(save_dir=config.output_dir, name="lightning_logs", version=config.run_name)

    trainer = pl.Trainer(
        max_epochs=config.optimizer.max_epochs,
        accelerator=config.trainer.accelerator,
        devices=config.trainer.devices,
        precision=config.trainer.precision,
        logger=logger,
        callbacks=[checkpoint],
        log_every_n_steps=config.trainer.log_every_n_steps,
        deterministic=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        limit_train_batches=config.trainer.limit_train_batches,
        limit_val_batches=config.trainer.limit_val_batches,
        limit_test_batches=config.trainer.limit_test_batches,
    )

    start_time = time.time()
    trainer.fit(model, datamodule=datamodule)
    best_path = checkpoint.best_model_path or None
    val_metrics = trainer.validate(model, datamodule=datamodule, ckpt_path=best_path, verbose=False)
    test_metrics = trainer.test(model, datamodule=datamodule, ckpt_path=best_path, verbose=False)
    duration_seconds = time.time() - start_time

    history_path = Path(logger.log_dir) / "metrics.csv"
    metrics_history = pd.read_csv(history_path) if history_path.exists() else pd.DataFrame()
    if not metrics_history.empty:
        metrics_history.to_csv(run_dir / "history.csv", index=False)

    result = {
        "run_name": config.run_name,
        "seed": config.seed,
        "dataset": config.dataset.name,
        "model": config.model.name,
        "beta": config.auxiliary.beta,
        "aux_enabled": config.auxiliary.enabled,
        "strategy": config.auxiliary.strategy,
        "lookahead": config.auxiliary.lookahead,
        "sigma": config.auxiliary.sigma,
        "detach_target": config.auxiliary.detach_target,
        "normalize_gradients": config.auxiliary.normalize_gradients,
        "max_epochs": config.optimizer.max_epochs,
        "train_subset": config.dataset.train_subset,
        "val_subset": config.dataset.val_subset,
        "test_subset": config.dataset.test_subset,
        "val_acc": _safe_metric(val_metrics, "val/acc"),
        "val_total_loss": _safe_metric(val_metrics, "val/total_loss"),
        "test_acc": _safe_metric(test_metrics, "test/acc"),
        "test_total_loss": _safe_metric(test_metrics, "test/total_loss"),
        "duration_seconds": duration_seconds,
        "best_checkpoint": best_path,
        "log_dir": logger.log_dir,
    }
    append_csv_row(Path(config.output_dir) / "all_runs.csv", result)
    write_json(run_dir / "result.json", result)

    del trainer, model, datamodule
    gc.collect()
    return result
