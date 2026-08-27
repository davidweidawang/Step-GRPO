#!/bin/bash
# Step-GRPO: Internalizing Dynamic Early Exit for Efficient Reasoning.
# Hyper-parameters follow the paper: G=5, delta=0.95, alpha=0.9, beta=0.5,
# lr=1e-6, KL=0.01, rollout batch 512, global batch 128, 100 steps, 16k tokens.
#
# Ablations:
#   w/o Step Reward:     worker.reward.reward_function_kwargs='{"enable_step_reward": false}'
#   w/o Dynamic Rollout: worker.rollout.deer_enable=false
#   w/ All-Sample Mean:  worker.reward.reward_function_kwargs='{"use_correct_mean": false}'
# Baseline GRPO:         worker.rollout.deer_enable=false + reward_function=./examples/reward_function/math.py:compute_score

set -x

export WANDB_MODE=offline

# Ray sockets need a SHORT, LOCAL path (AF_UNIX path <= 107 bytes; NFS is unreliable
# for sockets). ~/.tmp filled up, so use local /tmp instead.
export RAY_TMPDIR=/tmp/ray_wwd
mkdir -p "${RAY_TMPDIR}"

# HF cache defaults to a read-only, root-owned dir on this node; redirect the
# datasets cache to a writable location so load_dataset can build the parquet cache.
export HF_HOME=/mnt/shared-storage-user/ai4cmp/step-grpo/.hf_cache
export HF_DATASETS_CACHE=/mnt/shared-storage-user/ai4cmp/step-grpo/.hf_cache/datasets
mkdir -p "${HF_DATASETS_CACHE}"

# HF cache-style dir: point to the actual snapshot that contains config.json.
MODEL_PATH=/mnt/shared-storage-user/ai4cmp/models/Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5
TRAIN_FILE=/mnt/shared-storage-user/ai4cmp/step-grpo/data/dapo-data/data/train.parquet  # DAPO-Math-17k
VAL_FILE=/mnt/shared-storage-user/ai4cmp/step-grpo/data/dapo-data/data/test.parquet

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.max_response_length=16384 \
    data.rollout_batch_size=512 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=128 \
    worker.rollout.n=5 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.deer_enable=true \
    `# NOTE: keep tensor_parallel_size=1 for the DEER rollout whenever the model` \
    `# fits on a single GPU (one vLLM engine per GPU, data-parallel across GPUs).` \
    `# For larger models that require TP > 1, also set worker.rollout.enforce_eager=true` \
    `# to avoid TP-group divergence hangs in vLLM's SPMD (external_launcher) mode.` \
    worker.rollout.deer_threshold=0.95 \
    `# DEER's Qwen3 over-confidence fix (default true): only trust the confidence` \
    `# when the tentative answer itself generated </think>, i.e. the model closed` \
    `# the think block right after the induced answer. Filters "confidently wrong"` \
    `# exits whose answer was cut off by the probe token budget.` \
    worker.rollout.deer_qwen3_strict=true \
    worker.rollout.val_override_config.temperature=0.6 \
    worker.rollout.val_override_config.top_p=0.9 \
    worker.reward.reward_function=./examples/reward_function/step_grpo.py:compute_score \
    trainer.max_steps=100 \
    trainer.experiment_name=qwen3_8b_math_step_grpo
