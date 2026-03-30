# Greedy Auxiliary Loss

This repository tests a layerwise auxiliary objective in which each hidden layer predicts a detached summary of later representations. The target is built from fixed projections of downstream hidden states, and the training loss mixes the primary task objective with the auxiliary term using a coefficient `beta`.

I first selected the variant on MNIST, where a Gaussian future-target kernel centered two layers ahead worked best and gradient-normalized mixing failed badly. I then ran a full `beta` sweep on MNIST and a CPU-feasible CIFAR-100 ViT pilot, and the positive CIFAR result triggered two larger text follow-ups on AG News and DBPedia 14. The best observed test-accuracy gains were `+0.0019` on MNIST, `+0.0046` on CIFAR-100, `+0.0118` on AG News, and `+0.0073` on DBPedia 14.

![MNIST and CIFAR-100 beta sweep](reports/figures/beta_sweep.png)
![Absolute gain from the best auxiliary setting](reports/figures/dataset_summary.png)

The headline result is that the method helped on all four pilot datasets when `beta` stayed in a moderate range. On the harder datasets, the most reliable settings were `beta=0.1` to `beta=0.2`; very large `beta` values eventually overwhelmed the primary objective and collapsed task performance. A fuller write-up, including the stage-1 ablation and the exact experimental budgets, is in [reports/experiment_report.md](reports/experiment_report.md).
