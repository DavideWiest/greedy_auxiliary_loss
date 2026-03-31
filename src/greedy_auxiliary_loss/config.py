from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class DatasetConfig:
    name: str
    data_dir: str = "data"
    batch_size: int = 128
    num_workers: int = 0
    train_subset: float = 1.0
    val_subset: float = 1.0
    test_subset: float = 1.0
    image_size: int | None = None
    normalization: str = "dataset"


@dataclass
class ModelConfig:
    name: str
    hidden_dim: int = 256
    num_layers: int = 4
    dropout: float = 0.1
    patch_size: int = 4
    num_heads: int = 4
    mlp_ratio: float = 4.0
    pretrained: bool = False


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    lr: float = 3e-4
    weight_decay: float = 1e-4
    momentum: float = 0.9
    max_epochs: int = 10
    gradient_clip_val: float = 1.0
    scheduler: str = "none"
    min_lr: float = 1e-6
    warmup_epochs: int = 0
    label_smoothing: float = 0.0


@dataclass
class AuxiliaryLossConfig:
    enabled: bool = False
    beta: float = 0.0
    beta_schedule: str = "constant"
    beta_mode: str = "convex"
    loss_ema_decay: float = 0.95
    aux_scale_max: float = 100.0
    strategy: str = "fixed"
    lookahead: int = 1
    sigma: float = 1.0
    include_output: bool = True
    detach_target: bool = True
    normalize_gradients: bool = False
    aux_dim: int = 0
    loss_type: str = "cosine"
    projector_seed: int = 17
    skip_last_aux_layers: int = 0
    direct_hidden_target: bool = False


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32-true"
    log_every_n_steps: int = 10
    limit_train_batches: float = 1.0
    limit_val_batches: float = 1.0
    limit_test_batches: float = 1.0


@dataclass
class RunConfig:
    run_name: str
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    auxiliary: AuxiliaryLossConfig = field(default_factory=AuxiliaryLossConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)
    output_dir: str = "results"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
