export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='./data'

export VLLM_ATTENTION_BACKEND=XFORMERS
export RAY_gsc_rpc_server_reconnect_timeout_s=100
export BASE_DIR=$(pwd)

WAND_PROJECT=""
export RAY_TMPDIR="/tmp/s_ray"
mkdir -p "$RAY_TMPDIR"
export TMPDIR="$RAY_TMPDIR"
N_NODES=1

n_gpus_per_node=4
train_batch_size=$[$n_gpus_per_node * 24]
val_batch_size=$[$n_gpus_per_node * 256]
actor_ppo_mini_batch_size=$[$n_gpus_per_node * 12]
actor_ppo_micro_batch_size=$[$n_gpus_per_node * 3]
log_prob_micro_batch_size=$[$n_gpus_per_node * 3]

tree_search_m=2
tree_search_n=3
tree_search_l=1
tree_search_k=3
export BASE_MODEL='Path of sft model'
export EXPERIMENT_NAME=R2Searcher-3B


ulimit -n 65535

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo_format_ts \
    data.train_files=$DATA_DIR/train.parquet \
    data.val_files=$DATA_DIR/valid_500.parquet \
    data.train_data_num=null \
    data.val_data_num=null \
    data.train_batch_size=$train_batch_size \
    data.val_batch_size=$val_batch_size \
    data.max_prompt_length=4096 \
    data.max_response_length=500 \
    data.max_start_length=2048 \
    data.max_obs_length=768 \
    data.shuffle_train_dataloader=True \
    algorithm.adv_estimator=tree \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.actor.policy_loss=grpo \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.08 \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.actor.ppo_mini_batch_size=$actor_ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_micro_batch_size=$actor_ppo_micro_batch_size \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.grad_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tree_search=true \
    actor_rollout_ref.rollout.ts_m=$tree_search_m \
    actor_rollout_ref.rollout.ts_n=$tree_search_n \
    actor_rollout_ref.rollout.ts_l=$tree_search_l \
    actor_rollout_ref.rollout.ts_k=$tree_search_k \
    actor_rollout_ref.rollout.reward_mode=base \
    actor_rollout_ref.rollout.expand_mode=random \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    +actor_rollout_ref.rollout.disable_log_stats=true \
    +actor_rollout_ref.rollout.enable_chunked_prefill=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size=$log_prob_micro_batch_size \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    algorithm.no_think_rl=false \
    algorithm.use_kl_in_reward=false \
    actor_rollout_ref.rollout.n_agent=1 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.actor.state_masking=true \
    trainer.logger="['swanlab', 'console']" \
    +trainer.val_only=false \
    +trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=60 \
    trainer.project_name=$WAND_PROJECT \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.total_epochs=2 \
    trainer.total_training_steps=640 \
    trainer.default_hdfs_dir=null \
    trainer.default_local_dir=${BASE_DIR}/verl_checkpoints/$EXPERIMENT_NAME \
    reward_model.structure_format_score=0.2 \
    reward_model.final_format_score=0.1 \
    reward_model.retrieval_score=0 \
    max_turns=3 \
    retriever.url="http://127.0.0.1:8000/retrieve" \
    retriever.topk=3 \
    2>&1 | tee verl_log/$EXPERIMENT_NAME.log
