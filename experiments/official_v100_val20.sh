#!/usr/bin/env bash
set -euo pipefail

cd /home/wzy/workspace/r2searcher/R2-Searcher

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export RAY_DEDUP_LOGS=0
export VLLM_ATTENTION_BACKEND=XFORMERS
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

export RAY_TMPDIR=/mnt/data/R2Searcher/ray_v100_smoke
export TMPDIR="$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR" logs log/val

args=(
  data.train_files="$PWD/data/train.parquet"
  data.val_files="$PWD/data/smoke_20.parquet"
  data.train_batch_size=4
  data.val_batch_size=1

  data.max_prompt_length=2048
  data.max_start_length=1024
  data.max_response_length=384
  data.max_obs_length=384

  actor_rollout_ref.model.path="$PWD/models/Qwen2.5-3B-Instruct"
  actor_rollout_ref.model.use_remove_padding=false

  '+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:fp16,reduce_dtype:fp32,buffer_dtype:fp32}'
  '+actor_rollout_ref.ref.fsdp_config.mixed_precision={param_dtype:fp16,reduce_dtype:fp32,buffer_dtype:fp32}'
  '+actor_rollout_ref.ref.fsdp_config.model_dtype=fp16'

  actor_rollout_ref.actor.ppo_mini_batch_size=2
  actor_rollout_ref.actor.ppo_micro_batch_size=2
  actor_rollout_ref.rollout.log_prob_micro_batch_size=2
  actor_rollout_ref.ref.log_prob_micro_batch_size=2

  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.dtype=float16
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30
  actor_rollout_ref.rollout.enforce_eager=true
  actor_rollout_ref.rollout.max_num_batched_tokens=4096
  actor_rollout_ref.rollout.max_num_seqs=8

  '+actor_rollout_ref.rollout.disable_log_stats=true'
  '+actor_rollout_ref.rollout.enable_chunked_prefill=false'

  actor_rollout_ref.rollout.tree_search=true
  actor_rollout_ref.rollout.ts_m=2
  actor_rollout_ref.rollout.ts_n=2
  actor_rollout_ref.rollout.ts_l=1
  actor_rollout_ref.rollout.ts_k=3
  actor_rollout_ref.rollout.reward_mode=base

  algorithm.adv_estimator=tree
  reward_model.enable=false

  trainer.n_gpus_per_node=2
  trainer.nnodes=1
  trainer.logger='[console]'
  trainer.project_name=r2_v100
  trainer.experiment_name=official_val20
  trainer.default_hdfs_dir=null
  trainer.default_local_dir="$PWD/results/official_val20"

  '+trainer.val_only=true'
  '+trainer.val_before_train=true'

  max_turns=3
)

# 先验证 Hydra 配置，不启动模型
python -m verl.trainer.main_ppo_format_ts \
  "${args[@]}" --cfg job --resolve \
  > logs/official_val20_config.yaml

echo "HYDRA CONFIG PASS"

# 启动官方验证
python -u -m verl.trainer.main_ppo_format_ts \
  "${args[@]}" \
  2>&1 | tee logs/official_val20.log
