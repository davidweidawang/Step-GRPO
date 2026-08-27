#!/bin/bash
# Smoke test for the Step-GRPO Dynamic Truncated Rollout: tiny batch / few
# steps / short responses just to verify the DEER rollout runs end-to-end and
# produces sensible early-exit statistics. NOT for producing paper results.
#
# Watch rank-0 logs for:
#   [Step-GRPO rollout] batch=..., exits={'early_exit':N,'natural':N,'budget':N,'eos':N}
#   reward metrics: accuracy / steps / step_mean

set -x

# Ray sockets need a SHORT, LOCAL path (AF_UNIX path <= 107 bytes; NFS is unreliable
# for sockets). ~/.tmp filled up, so use local /tmp instead.
export RAY_TMPDIR=/tmp/ray_wwd
mkdir -p "${RAY_TMPDIR}"

# HF cache defaults to a read-only, root-owned dir on this node; redirect the
# datasets cache to a writable location so load_dataset can build the parquet cache.
export HF_HOME=/mnt/shared-storage-user/ai4cmp/step-grpo/.hf_cache
export HF_DATASETS_CACHE=/mnt/shared-storage-user/ai4cmp/step-grpo/.hf_cache/datasets
mkdir -p "${HF_DATASETS_CACHE}"

# Reduce CUDA fragmentation for the colocated train + vLLM setup (single GPU is tight).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# HF cache-style dir: point to the actual snapshot that contains config.json.
MODEL_PATH=/mnt/shared-storage-user/ai4cmp/models/Qwen3-8B/snapshots/9c925d64d72725edaf899c6cb9c377fd0709d9c5
TRAIN_FILE=/mnt/shared-storage-user/ai4cmp/step-grpo/data/dapo-data/data/train.parquet
VAL_FILE=/mnt/shared-storage-user/ai4cmp/step-grpo/data/dapo-data/data/test.parquet

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.rollout_batch_size=8 \
    data.val_batch_size=8 \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.actor.global_batch_size=8 \
    worker.rollout.n=5 \
    worker.rollout.deer_enable=true \
    worker.rollout.deer_threshold=0.95 \
    worker.rollout.deer_think_ratio=0.9 \
    worker.rollout.gpu_memory_utilization=0.4 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=./examples/reward_function/step_grpo.py:compute_score \
    trainer.total_epochs=1 \
    trainer.max_steps=2 \
    trainer.val_before_train=false \
    trainer.val_freq=-1 \
    trainer.save_freq=-1 \
    trainer.n_gpus_per_node=1 \
    trainer.experiment_name=qwen3_8b_step_grpo_smoke
