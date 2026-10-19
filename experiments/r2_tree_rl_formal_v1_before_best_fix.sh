#!/usr/bin/env bash
set -Eeuo pipefail

cd /home/wzy/workspace/r2searcher/R2-Searcher

# ---------- Preflight ----------
for f in \
  data/train.parquet \
  data/valid_500.parquet \
  models/Qwen2.5-3B-R2-SFT-E5/config.json
do
  test -s "$f" || {
    echo "ERROR: missing or empty file: $f"
    exit 2
  }
done

# 检索服务必须仍然可连接；不向服务发送测试请求
if ! timeout 2 bash -c \
  'cat < /dev/null > /dev/tcp/127.0.0.1/8000' 2>/dev/null
then
  echo "ERROR: retriever port 8000 is not reachable."
  echo "Start the existing retriever service before training."
  exit 2
fi

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export RAY_gcs_rpc_server_reconnect_timeout_s=100

ulimit -n 65535 || true

RUN_ID="r2_tree_rl_formal_v1_$(date +%Y%m%d_%H%M%S)"
export RAY_TMPDIR="/tmp/r2r_${RUN_ID##*_}"
export TMPDIR="$RAY_TMPDIR"

mkdir -p "$RAY_TMPDIR" logs outputs log/val
LOG="logs/${RUN_ID}.log"
PREFLIGHT_LOG="logs/${RUN_ID}_preflight.txt"

echo "RUN_ID=$RUN_ID"
echo "LOG=$LOG"

{
  echo "RUN_ID=$RUN_ID"
  echo "DATE=$(date -Is)"
  echo "GIT_COMMIT=$(git rev-parse HEAD)"
  echo "=== GIT STATUS ==="
  git status --short
  echo "=== DATA AND MODEL HASHES ==="
  sha256sum \
    data/train.parquet \
    data/valid_500.parquet \
    models/Qwen2.5-3B-R2-SFT-E5/config.json
  echo "=== GPU ==="
  nvidia-smi
  echo "=== HOST MEMORY ==="
  free -h
  echo "=== DISK ==="
  df -h .
} 2>&1 | tee "$PREFLIGHT_LOG"

# ---------- RL training ----------
set +e
python -m verl.trainer.main_ppo_format_ts \
  data.train_files="$PWD/data/train.parquet" \
  data.val_files="$PWD/data/valid_500.parquet" \
  data.train_data_num=null \
  data.val_data_num=null \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=2048 \
  data.max_start_length=1024 \
  data.max_response_length=256 \
  data.max_obs_length=256 \
  data.shuffle_train_dataloader=true \
  algorithm.adv_estimator=tree \
  algorithm.no_think_rl=false \
  algorithm.use_kl_in_reward=false \
  actor_rollout_ref.model.path="$PWD/models/Qwen2.5-3B-R2-SFT-E5" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.policy_loss=grpo \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.08 \
  actor_rollout_ref.actor.ppo_mini_batch_size=4 \
  actor_rollout_ref.actor.ppo_micro_batch_size=1 \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.state_masking=true \
  actor_rollout_ref.actor.fsdp_config.param_offload=false \
  actor_rollout_ref.actor.fsdp_config.grad_offload=false \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
  '+actor_rollout_ref.actor.fsdp_config.mixed_precision={param_dtype:fp16,reduce_dtype:fp32,buffer_dtype:fp32}' \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.dtype=float16 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  '+actor_rollout_ref.rollout.disable_log_stats=true' \
  '+actor_rollout_ref.rollout.enable_chunked_prefill=false' \
  actor_rollout_ref.rollout.n_agent=1 \
  actor_rollout_ref.rollout.temperature=1 \
  actor_rollout_ref.rollout.tree_search=true \
  actor_rollout_ref.rollout.ts_m=2 \
  actor_rollout_ref.rollout.ts_n=1 \
  actor_rollout_ref.rollout.ts_l=1 \
  actor_rollout_ref.rollout.ts_k=2 \
  actor_rollout_ref.rollout.reward_mode=base \
  actor_rollout_ref.rollout.expand_mode=random \
  actor_rollout_ref.rollout.log_prob_micro_batch_size=1 \
  actor_rollout_ref.ref.log_prob_micro_batch_size=1 \
  actor_rollout_ref.ref.fsdp_config.param_offload=true \
  '+actor_rollout_ref.ref.fsdp_config.mixed_precision={param_dtype:fp16,reduce_dtype:fp32,buffer_dtype:fp32}' \
  '+actor_rollout_ref.ref.fsdp_config.model_dtype=fp16' \
  reward_model.enable=false \
  reward_model.structure_format_score=0.2 \
  reward_model.final_format_score=0.1 \
  reward_model.retrieval_score=0 \
  trainer.n_gpus_per_node=2 \
  trainer.nnodes=1 \
  trainer.logger='[console]' \
  trainer.project_name=r2_v100 \
  trainer.experiment_name="$RUN_ID" \
  '+trainer.val_only=false' \
  '+trainer.val_before_train=false' \
  trainer.total_epochs=2 \
  trainer.total_training_steps=640 \
  trainer.test_freq=60 \
  trainer.save_freq=-1 \
  '+trainer.smoke_snapshot_every=10' \
  '+trainer.full_resume_enabled=true' \
  '+trainer.full_resume_every=30' \
  '+trainer.full_resume_data_seed=20261012' \
  '+trainer.full_resume_from=null' \
  '+trainer.smoke_skip_saving=true' \
  '+trainer.smoke_best_snapshot=true' \
  '+trainer.smoke_best_metric=auto' \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$PWD/outputs/$RUN_ID" \
  max_turns=3 \
  retriever.url=http://127.0.0.1:8000/retrieve \
  retriever.topk=3 \
  2>&1 | tee "$LOG"

RC=${PIPESTATUS[0]}
set -e

echo "TRAIN_PROCESS_EXIT_CODE=$RC" | tee -a "$LOG"
echo "RUN_ID=$RUN_ID" | tee -a "$LOG"
echo "LOG=$LOG" | tee -a "$LOG"

echo "=== KEY EVENTS ==="
grep -E \
  'epoch |FULL_RESUME_|actor/|val/|Final validation metrics|Traceback|CUDA out of memory|OutOfMemoryError|Error executing job' \
  "$LOG" | tail -100 || true

exit "$RC"
