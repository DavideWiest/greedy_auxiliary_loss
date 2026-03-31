from __future__ import annotations

import argparse

from greedy_auxiliary_loss.config import (
    AuxiliaryLossConfig,
    DatasetConfig,
    ModelConfig,
    OptimizerConfig,
    RunConfig,
    TrainerConfig,
)
from greedy_auxiliary_loss.runner import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single greedy auxiliary-loss experiment.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset", choices=["mnist", "cifar100", "ag_news", "dbpedia_14"], required=True)
    parser.add_argument("--model", choices=["mlp", "vit", "resnet18", "text_transformer"], required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--train-subset", type=float, default=1.0)
    parser.add_argument("--val-subset", type=float, default=1.0)
    parser.add_argument("--test-subset", type=float, default=1.0)
    parser.add_argument("--image-size", type=int, default=0)
    parser.add_argument("--normalization", choices=["dataset", "imagenet"], default="dataset")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--patch-size", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=float, default=4.0)
    parser.add_argument("--pretrained", action="store_true", default=False)
    parser.add_argument("--optimizer-name", choices=["adamw", "sgd"], default="adamw")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default="none")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--warmup-epochs", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--beta-schedule", choices=["constant", "linear_decay", "linear_warmup"], default="constant")
    parser.add_argument("--strategy", choices=["fixed", "gaussian", "uniform", "output", "exponential"], default="fixed")
    parser.add_argument("--lookahead", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--include-output", action="store_true", default=False)
    parser.add_argument("--detach-target", action="store_true", default=False)
    parser.add_argument("--normalize-gradients", action="store_true", default=False)
    parser.add_argument("--aux-dim", type=int, default=0)
    parser.add_argument("--skip-last-aux-layers", type=int, default=0)
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--limit-train-batches", type=float, default=1.0)
    parser.add_argument("--limit-val-batches", type=float, default=1.0)
    parser.add_argument("--limit-test-batches", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        run_name=args.run_name,
        seed=args.seed,
        dataset=DatasetConfig(
            name=args.dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            train_subset=args.train_subset,
            val_subset=args.val_subset,
            test_subset=args.test_subset,
            image_size=args.image_size or None,
            normalization=args.normalization,
        ),
        model=ModelConfig(
            name=args.model,
            hidden_dim=args.hidden_dim,
            num_layers=args.num_layers,
            dropout=args.dropout,
            patch_size=args.patch_size,
            num_heads=args.num_heads,
            mlp_ratio=args.mlp_ratio,
            pretrained=args.pretrained,
        ),
        optimizer=OptimizerConfig(
            name=args.optimizer_name,
            lr=args.lr,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
            max_epochs=args.max_epochs,
            gradient_clip_val=args.gradient_clip_val,
            scheduler=args.scheduler,
            min_lr=args.min_lr,
            warmup_epochs=args.warmup_epochs,
            label_smoothing=args.label_smoothing,
        ),
        auxiliary=AuxiliaryLossConfig(
            enabled=args.beta > 0.0,
            beta=args.beta,
            beta_schedule=args.beta_schedule,
            strategy=args.strategy,
            lookahead=args.lookahead,
            sigma=args.sigma,
            include_output=args.include_output,
            detach_target=args.detach_target,
            normalize_gradients=args.normalize_gradients,
            aux_dim=args.aux_dim,
            skip_last_aux_layers=args.skip_last_aux_layers,
        ),
        trainer=TrainerConfig(
            limit_train_batches=args.limit_train_batches,
            limit_val_batches=args.limit_val_batches,
            limit_test_batches=args.limit_test_batches,
        ),
        output_dir=args.output_dir,
    )
    result = run_experiment(config)
    print(result)


if __name__ == "__main__":
    main()
