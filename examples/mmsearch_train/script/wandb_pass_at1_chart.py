"""Log a single W&B chart: val/unknown/pass@1 vs step.

X-axis uses training step with stride STEP_SIZE (e.g. 0, 10, 20, ...).

Run: python examples/mmsearch_train/script/wandb_pass_at1_chart.py
Requires: wandb login
"""

from __future__ import annotations

import wandb

STEP_SIZE = 10

# Replace with your y values; x will be 0, STEP_SIZE, 2*STEP_SIZE, ...
YS: list[float] = [0.25, 0.32, 0.33, 0.28, 0.27, 0, 0, 0, 0, 0]

wandb.init(project="mmsearch-debug", name="pass-at-1-chart")
for i, y in enumerate(YS):
    wandb.log({"val/unknown/pass@1": y}, step=i * STEP_SIZE)
wandb.finish()
