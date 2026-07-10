# lerobot_policy_smolvla_drift

A [LeRobot](https://github.com/huggingface/lerobot) plugin policy for **SmolVLA-Drift**.

SmolVLA decodes an action chunk by integrating a flow-matching ODE, which takes 10 forward
passes of the action expert per chunk. SmolVLA-Drift trains the same network with a one-step
**Drifting** (DBPO) objective instead: a single forward pass maps noise directly to the chunk.

- **90.2%** success on LIBERO-Spatial (fresh action expert, 30k steps)
- **1 NFE** per chunk — **~4.4× faster** decode than 10-step flow matching
- Same VLM + action expert as SmolVLA, byte-identical weight layout

Project website: <https://zuoxingdong.github.io/drift-vla/>

## Install

Python >= 3.12. Pulls `lerobot[smolvla,dataset]>=0.6.0,<0.7`.

From GitHub:

```bash
pip install "git+https://github.com/zuoxingdong/lerobot_policy_smolvla_drift.git"
```

From a local clone (editable):

```bash
git clone https://github.com/zuoxingdong/lerobot_policy_smolvla_drift.git
cd lerobot_policy_smolvla_drift
pip install -e .
```

The LIBERO evaluation below additionally needs:

```bash
pip install "lerobot[libero]"
pip install "mujoco==3.3.2"   # newer MuJoCo changes rendered colors
```

## Train on LIBERO

The winning recipe (G=8, temperatures (0.02, 0.05, 0.2), per-action-dim) is the config default,
so no drift flags are needed:

```bash
lerobot-train \
  --policy.type=smolvla_drift \
  --policy.load_vlm_weights=true \
  --policy.n_action_steps=1 \
  --dataset.repo_id=lerobot/libero \
  --batch_size=64 \
  --seed=1000 \
  --steps=30000
```

`--policy.n_action_steps=1` (closed-loop replanning, saved into the checkpoint) is load-bearing:
the 90.2% was measured closed-loop. Executing the full 50-step chunk open-loop scores far lower.

## Evaluate on LIBERO-Spatial

Needs a graphics-capable node for the MuJoCo/LIBERO renderer. The 90.2% protocol is 20 episodes
on each of the 10 `libero_spatial` tasks, seed 1000:

```bash
for task_id in 0 1 2 3 4 5 6 7 8 9; do
  lerobot-eval \
    --policy.path=<checkpoint-path-or-hub-id> \
    --policy.n_action_steps=1 \
    --env.type=libero \
    --env.task=libero_spatial \
    --env.task_ids="[$task_id]" \
    --eval.n_episodes=20 \
    --eval.batch_size=2 \
    --eval.use_async_envs=true \
    --seed=1000 \
    --output_dir=eval/task_${task_id}
done
# then aggregate the ten eval_info.json files
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

Faithfully vendored from LeRobot's SmolVLA (fork
[`zuoxingdong/smolvla-drift`](https://github.com/zuoxingdong/smolvla-drift) @
[`15778da0`](https://github.com/zuoxingdong/smolvla-drift/commit/15778da0)) — only class renames,
the registration string, import fixes, drift-recipe defaults, and one docstring. Apache-2.0
(`LICENSE`); the original HuggingFace copyright headers are retained in each source file.
