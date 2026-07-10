# lerobot_policy_smolvla_drift

SmolVLA-Drift as a [LeRobot](https://github.com/huggingface/lerobot) plugin policy. It swaps
SmolVLA's flow-matching objective for a one-step **Drifting** (DBPO) objective with **1-NFE**
inference — same VLM + action expert, byte-identical weights. On LIBERO-Spatial: **90.2% success,
1 NFE, ~4.4× faster** per chunk than 10-step flow matching.

**Project website:** [zuoxingdong.github.io/drift-vla](https://zuoxingdong.github.io/drift-vla/)

## Install

```bash
pip install lerobot_policy_smolvla_drift   # pulls lerobot[smolvla,dataset]>=0.6.0,<0.7 · Python >=3.12
```

For the LIBERO eval below you also need the benchmark extra: `pip install 'lerobot[libero]'`
(then pin `pip install 'mujoco==3.3.2'` — newer MuJoCo changes rendered colors).

## Train & evaluate on LIBERO

The winning recipe is the default (G=8, temperatures (0.02, 0.05, 0.2), per-action-dim), so no
drift flags are needed:

```bash
# train (fresh action expert + pretrained VLM; effective batch ~64)
lerobot-train --policy.type=smolvla_drift --policy.load_vlm_weights=true \
  --policy.n_action_steps=1 \
  --dataset.repo_id=lerobot/libero --batch_size=64 --seed=1000 --steps=30000
```

`--policy.n_action_steps=1` (closed-loop replanning, saved into the checkpoint) is load-bearing:
the 90.2% was measured closed-loop; executing the full 50-step chunk open-loop scores far lower.

```bash
# evaluate (needs a graphics-capable node for the MuJoCo/LIBERO renderer)
# 90.2% protocol: 20 episodes × each of the 10 libero_spatial tasks, seed 1000.
for task_id in 0 1 2 3 4 5 6 7 8 9; do
  lerobot-eval --policy.path=<checkpoint-path-or-hub-id> \
    --policy.n_action_steps=1 \
    --env.type=libero --env.task=libero_spatial --env.task_ids="[$task_id]" \
    --eval.n_episodes=20 --eval.batch_size=2 --eval.use_async_envs=true \
    --seed=1000 --output_dir=eval/task_${task_id}
done   # then aggregate the ten eval_info.json files
```

## Use from Python

```python
import lerobot_policy_smolvla_drift            # registers "smolvla_drift" — import before loading
from lerobot.policies.factory import get_policy_class

policy = get_policy_class("smolvla_drift").from_pretrained("<checkpoint-path-or-hub-id>")
action = policy.select_action(batch)           # (B, action_dim); 1-NFE drift inference
```

The `import lerobot_policy_smolvla_drift` line is required in scripts: LeRobot auto-discovers the
plugin inside the `lerobot-*` CLIs, but not in your own Python process.

## Provenance & license

Faithfully vendored from LeRobot's SmolVLA (fork `zuoxingdong/smolvla-drift` @ `15778da0`) — only
class renames, the registration string, import fixes, drift-recipe defaults, and one docstring.
Apache-2.0 (`LICENSE`); the original HuggingFace copyright headers are retained in each source file.
