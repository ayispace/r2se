#!/usr/bin/env bash
# One-step functional test: REAL train-time tree expansion -> rewards -> R2PO update.
# Not a paper score replication. Never touches CUDA device 4.
set -euo pipefail
cd /home/wzy/workspace/r2searcher/R2-Searcher

for f in data/train.parquet data/smoke_1.parquet models/Qwen2.5-3B-Instruct/config.json; do
  test -f "$f" || { echo "MISSING: $f"; exit 2; }
done

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export VLLM_ATTENTION_BACKEND=XFORMERS
export TOKENIZERS_PARALLELISM=false
export TREE_SEARCH_DEBUG=true
export PYTHONUNBUFFERED=1
RUN_ID="tree_rl_step1_$(date +%Y%m%d_%H%M%S)"
export RAY_TMPDIR="/tmp/${RUN_ID}_ray"
mkdir -p "$RAY_TMPDIR" logs "outputs/$RUN_ID" log/val
LOG="logs/${RUN_ID}.log"
echo "RUN_ID=$RUN_ID"
echo "LOG=$LOG"
df -h .
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv

set +e
python -m verl.trainer.main_ppo_format_ts \
  data.train_files="$PWD/data/train.parquet" \
  data.val_files="$PWD/data/smoke_1.parquet" \
  data.train_data_num=1 \
  data.val_data_num=1 \
  data.train_batch_size=1 \
  data.val_batch_size=1 \
  data.max_prompt_length=2048 \
  data.max_start_length=1024 \
  data.max_response_length=256 \
  data.max_obs_length=256 \
  data.shuffle_train_dataloader=false \
  algorithm.adv_estimator=tree \
  actor_rollout_ref.model.path="$PWD/models/Qwen2.5-3B-Instruct" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  actor_rollout_ref.model.use_remove_padding=false \
  actor_rollout_ref.actor.policy_loss=grpo \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.0 \
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
  actor_rollout_ref.rollout.gpu_memory_utilization=0.30 \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
  actor_rollout_ref.rollout.max_num_seqs=8 \
  '+actor_rollout_ref.rollout.disable_log_stats=true' \
  '+actor_rollout_ref.rollout.enable_chunked_prefill=false' \
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
  trainer.total_epochs=1 \
  trainer.total_training_steps=1 \
  +trainer.smoke_skip_saving=true trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.default_hdfs_dir=null \
  trainer.default_local_dir="$PWD/outputs/$RUN_ID" \
  max_turns=3 \
  retriever.url=http://127.0.0.1:8000/retrieve \
  retriever.topk=3 \
  2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
set -e
printf 'TRAIN_PROCESS_EXIT_CODE=%s\n' "$RC"
printf 'LOG=%s\n' "$LOG"
echo '=== KEY EVENTS ==='
grep -E 'epoch |get expansion for root|update_actor|actor/|Final validation metrics|Traceback|CUDA out of memory|Error executing job' "$LOG" | tail -40 || true
exit "$RC"
