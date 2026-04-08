"""Log a single W&B chart: val/unknown/pass@1 vs step (10 steps).

Run: python examples/mmsearch_train/script/wandb_pass_at1_chart.py
Requires: wandb login
"""

from __future__ import annotations

import wandb

# Replace with your y values (length must be 10).
YS: list[float] = [0.25, 0.32, 0.33, 0.28, 0.27]

NUM_STEPS = 10
assert len(YS) == NUM_STEPS, f"YS must have length {NUM_STEPS}"

wandb.init(project="mmsearch-debug", name="pass-at-1-chart")
for step in range(NUM_STEPS):
    wandb.log({"val/unknown/pass@1": YS[step]}, step=step)
wandb.finish()
