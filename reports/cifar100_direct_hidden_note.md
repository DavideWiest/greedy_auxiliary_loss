# Direct Hidden-Target CIFAR-100 Study

This study reruns CIFAR-100 with the direct hidden-target auxiliary variant requested after the earlier logits/projection experiments. The auxiliary objective uses only hidden states, compares matching coordinates directly with MSE, keeps `detach_target=True`, and interprets `beta` as the auxiliary share of the update through gradient-share normalization rather than as a raw loss coefficient.

The recipe is intentionally architecture-aligned with that objective: a same-width 4-layer ViT (`hidden_dim=128`, `patch_size=4`) trained on full CIFAR-100 for 1 epoch on CPU. The strategy scan used `beta=0.1`, and the sweep then reused the validation-best strategy.

The main result is that the new `beta` definition is no longer flat. Accuracy changes materially well before `beta=1.0`. The best positive setting in this study was `beta=0.1` with validation accuracy `0.1068` and test accuracy `0.1102`, compared with the baseline at `0.1265`. That means the direct hidden-target variant is now tunable rather than inert, but under this CPU-feasible ViT recipe it still does not beat the baseline.
