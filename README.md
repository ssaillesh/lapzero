# LapZero

LapZero trains an AI race driver in the F1TENTH Gym simulator. It first copies human laps (behaviour cloning), then improves with reinforcement learning (PPO). A release gate decides whether a newly trained policy replaces the current champion.

## Release gate

**Robustness-aware promotion rule:** a challenger policy replaces the champion if it achieves **>= 98% completion** across the domain-randomized friction spectrum (**mu in [0.60, 1.05]**) **and** maintains its **nominal-grip lap time within <= 2.0%** of the champion baseline.

How it is measured (`scripts/eval_sweep.py`, decided by `lapzero/eval/gate.py`):

- Both policies are swept on the same track with the same settings: friction mu 0.50 to 1.05 in 0.05 steps, 10 one-lap episodes per level from jittered start poses (seeds 0 to 9, the same at every level).
- Completion is pooled over every episode at mu 0.60 to 1.05: 10 levels, 100 episodes, so at most 2 may fail. The levels below 0.60 are reported but do not count.
- Nominal grip is mu 1.05 (F1TENTH's default 1.0489, rounded). The challenger's mean lap time there may be at most 2.0% above the champion's. A policy with no completed lap at nominal grip fails this check.
- The gate refuses to compare policies whose `contract_version` major numbers differ.

Changing these rules requires approval.
