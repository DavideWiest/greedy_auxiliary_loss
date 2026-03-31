from __future__ import annotations

from typing import Any

import lightning.pytorch as pl
import torch
import torch.nn.functional as F

from greedy_auxiliary_loss.config import RunConfig
from greedy_auxiliary_loss.losses import LayerwiseAuxiliaryObjective
from greedy_auxiliary_loss.models import LayerwiseMLP, LayerwiseResNet18, LayerwiseTextTransformer, LayerwiseViT


def build_model(config: RunConfig, dataset_metadata: dict[str, int | tuple[int, int, int]]) -> torch.nn.Module:
    num_classes = int(dataset_metadata["num_classes"])
    if config.model.name == "mlp":
        return LayerwiseMLP(
            input_shape=dataset_metadata["image_shape"],
            num_classes=num_classes,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            dropout=config.model.dropout,
        )
    if config.model.name == "text_transformer":
        return LayerwiseTextTransformer(
            vocab_size=int(dataset_metadata["vocab_size"]),
            max_length=int(dataset_metadata["max_length"]),
            num_classes=num_classes,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
        )
    if config.model.name == "vit":
        return LayerwiseViT(
            input_shape=dataset_metadata["image_shape"],
            num_classes=num_classes,
            hidden_dim=config.model.hidden_dim,
            num_layers=config.model.num_layers,
            patch_size=config.model.patch_size,
            num_heads=config.model.num_heads,
            mlp_ratio=config.model.mlp_ratio,
            dropout=config.model.dropout,
        )
    if config.model.name == "resnet18":
        return LayerwiseResNet18(
            num_classes=num_classes,
            pretrained=config.model.pretrained,
            dropout=config.model.dropout,
        )
    raise ValueError(f"Unsupported model: {config.model.name}")


class ClassificationModule(pl.LightningModule):
    def __init__(self, config: RunConfig, dataset_metadata: dict[str, int | tuple[int, int, int]]) -> None:
        super().__init__()
        self.save_hyperparameters(config.to_dict())
        self.config = config
        self.model = build_model(config, dataset_metadata=dataset_metadata)
        num_classes = int(dataset_metadata["num_classes"])
        self.automatic_optimization = False
        self._scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
        if config.auxiliary.enabled and config.auxiliary.beta > 0.0:
            self.auxiliary_objective = LayerwiseAuxiliaryObjective(
                layer_dims=self.model.hidden_dims,
                num_classes=num_classes,
                strategy=config.auxiliary.strategy,
                lookahead=config.auxiliary.lookahead,
                sigma=config.auxiliary.sigma,
                include_output=config.auxiliary.include_output,
                detach_target=config.auxiliary.detach_target,
                aux_dim=config.auxiliary.aux_dim,
                loss_type=config.auxiliary.loss_type,
                projector_seed=config.auxiliary.projector_seed,
                skip_last_aux_layers=config.auxiliary.skip_last_aux_layers,
            )
        else:
            self.auxiliary_objective = None

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer_name = self.config.optimizer.name.lower()
        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.config.optimizer.lr,
                weight_decay=self.config.optimizer.weight_decay,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.config.optimizer.lr,
                momentum=self.config.optimizer.momentum,
                weight_decay=self.config.optimizer.weight_decay,
                nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer.name}")

        scheduler_name = self.config.optimizer.scheduler.lower()
        if scheduler_name == "none":
            self._scheduler = None
        elif scheduler_name == "cosine":
            main_epochs = max(1, self.config.optimizer.max_epochs - self.config.optimizer.warmup_epochs)
            cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=main_epochs,
                eta_min=self.config.optimizer.min_lr,
            )
            if self.config.optimizer.warmup_epochs > 0:
                warmup = torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=0.1,
                    end_factor=1.0,
                    total_iters=self.config.optimizer.warmup_epochs,
                )
                self._scheduler = torch.optim.lr_scheduler.SequentialLR(
                    optimizer,
                    schedulers=[warmup, cosine],
                    milestones=[self.config.optimizer.warmup_epochs],
                )
            else:
                self._scheduler = cosine
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.optimizer.scheduler}")
        return optimizer

    def _current_beta(self) -> float:
        beta = self.config.auxiliary.beta
        schedule = self.config.auxiliary.beta_schedule.lower()
        if schedule == "constant" or beta <= 0.0 or self.trainer is None:
            return beta
        total_steps = max(1, int(getattr(self.trainer, "estimated_stepping_batches", 1)))
        progress = min(1.0, self.global_step / max(1, total_steps - 1))
        if schedule == "linear_decay":
            return beta * (1.0 - progress)
        if schedule == "linear_warmup":
            return beta * progress
        raise ValueError(f"Unsupported beta schedule: {self.config.auxiliary.beta_schedule}")

    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> dict[str, Any]:
        if isinstance(batch, dict):
            targets = batch["labels"]
            model_inputs = {key: value for key, value in batch.items() if key != "labels"}
            logits, hidden_states = self.model(**model_inputs)
        else:
            inputs, targets = batch
            logits, hidden_states = self.model(inputs)
        primary_loss = F.cross_entropy(logits, targets, label_smoothing=self.config.optimizer.label_smoothing)
        aux_loss = logits.new_tensor(0.0)
        if self.auxiliary_objective is not None:
            aux_loss, _ = self.auxiliary_objective(hidden_states, logits)
        beta = self._current_beta()
        total_loss = (1.0 - beta) * primary_loss + beta * aux_loss
        predictions = logits.argmax(dim=-1)
        accuracy = (predictions == targets).float().mean()
        self.log(f"{stage}/primary_loss", primary_loss, on_step=False, on_epoch=True, prog_bar=stage != "train")
        self.log(f"{stage}/aux_loss", aux_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}/total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=stage != "train")
        self.log(f"{stage}/acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)
        return {
            "primary_loss": primary_loss,
            "aux_loss": aux_loss,
            "total_loss": total_loss,
            "beta": beta,
        }

    def _apply_normalized_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        primary_loss: torch.Tensor,
        aux_loss: torch.Tensor,
        beta: float,
    ) -> tuple[float, float]:
        params = [parameter for parameter in self.parameters() if parameter.requires_grad]
        primary_grads = torch.autograd.grad(primary_loss, params, retain_graph=True, allow_unused=True)
        aux_grads = torch.autograd.grad(aux_loss, params, retain_graph=False, allow_unused=True)

        def grad_norm(grads: tuple[torch.Tensor | None, ...]) -> torch.Tensor:
            squared = [torch.sum(grad.detach() ** 2) for grad in grads if grad is not None]
            if not squared:
                return primary_loss.new_tensor(0.0)
            return torch.sqrt(torch.stack(squared).sum())

        primary_norm = grad_norm(primary_grads)
        aux_norm = grad_norm(aux_grads)
        reference_norm = (1.0 - beta) * primary_norm + beta * aux_norm
        optimizer.zero_grad(set_to_none=True)
        for parameter, primary_grad, aux_grad in zip(params, primary_grads, aux_grads):
            if primary_grad is None and aux_grad is None:
                continue
            combined = 0.0
            if primary_grad is not None:
                combined = combined + (1.0 - beta) * primary_grad / primary_norm.clamp_min(1e-8)
            if aux_grad is not None:
                combined = combined + beta * aux_grad / aux_norm.clamp_min(1e-8)
            if isinstance(combined, float):
                continue
            parameter.grad = reference_norm * combined
        return float(primary_norm.detach().cpu()), float(aux_norm.detach().cpu())

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        step_outputs = self._step(batch, stage="train")
        optimizer = self.optimizers()
        optimizer.zero_grad(set_to_none=True)
        should_normalize = (
            self.auxiliary_objective is not None
            and self.config.auxiliary.normalize_gradients
            and 0.0 < step_outputs["beta"] < 1.0
        )
        if should_normalize:
            primary_norm, aux_norm = self._apply_normalized_gradients(
                optimizer=optimizer,
                primary_loss=step_outputs["primary_loss"],
                aux_loss=step_outputs["aux_loss"],
                beta=step_outputs["beta"],
            )
            self.log("train/primary_grad_norm", primary_norm, on_step=False, on_epoch=True)
            self.log("train/aux_grad_norm", aux_norm, on_step=False, on_epoch=True)
        else:
            self.manual_backward(step_outputs["total_loss"])
        self.log("train/beta", step_outputs["beta"], on_step=False, on_epoch=True)
        if self.config.optimizer.gradient_clip_val > 0:
            self.clip_gradients(
                optimizer,
                gradient_clip_val=self.config.optimizer.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )
        optimizer.step()
        return step_outputs["total_loss"].detach()

    def on_train_epoch_end(self) -> None:
        if self._scheduler is None:
            return
        self._scheduler.step()
        optimizer = self.optimizers()
        self.log("train/lr", optimizer.param_groups[0]["lr"], on_step=False, on_epoch=True)

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, stage="val")

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        self._step(batch, stage="test")
