set -x

# Colocated GRPO training+generation for Qwen3-30B-A3B on SearchR1 data.
# follow the instructions in examples/search/README.md for setting up the dataset
# and for starting the local search server
# export WANDB_API_KEY=<your_key_here>
# bash examples/training_backends/megatron/run_search_megatron.sh

# path for dataset (.parquet files) containing the prompts and metadata for each question
MODEL_NAME="Qwen/Qwen3-30B-A3B"

DATA_DIR="$HOME/data/searchR1" # save to shared storage across nodes or use local storage on each node
NUM_NODES=4
NUM_GPUS_PER_NODE=8

MEGATRON_TP=2
MEGATRON_PP=1
MEGATRON_CP=1
MEGATRON_EP=8
MEGATRON_ETP=1

MICRO_TRAIN_BATCH_SIZE_PER_GPU=1
MICRO_FORWARD_BATCH_SIZE_PER_GPU=2

NUM_INFERENCE_ENGINES=4
INFERENCE_ENGINE_TP=8

export SKYRL_PYTHONPATH_EXPORT=1

uv run --isolated --frozen --extra mcore --extra vllm --env-file .env -m skyrl_train.entrypoints.main_base \
  data.train_data="['${DATA_DIR}/train.parquet']" \
  data.val_data="['${DATA_DIR}/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.policy.optimizer_config.max_grad_norm=0.5 \
  trainer.policy.optimizer_config.num_warmup_steps=94 \
  trainer.algorithm.use_kl_loss=true \
  trainer.algorithm.kl_loss_coef=0.001 \
  trainer.policy.model.path=$MODEL_NAME \
  trainer.placement.colocate_all=true \
  trainer.strategy=megatron \
  trainer.placement.policy_num_nodes=$NUM_NODES \
  trainer.placement.ref_num_nodes=$NUM_NODES \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS_PER_NODE \
  trainer.policy.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.policy.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.policy.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.policy.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.policy.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  trainer.ref.megatron_config.tensor_model_parallel_size=$MEGATRON_TP \
  trainer.ref.megatron_config.context_parallel_size=$MEGATRON_CP \
  trainer.ref.megatron_config.pipeline_model_parallel_size=$MEGATRON_PP \
  trainer.ref.megatron_config.expert_model_parallel_size=$MEGATRON_EP \
  trainer.ref.megatron_config.expert_tensor_parallel_size=$MEGATRON_ETP \
  generator.num_inference_engines=$NUM_INFERENCE_ENGINES \
  generator.inference_engine_tensor_parallel_size=$INFERENCE_ENGINE_TP \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.gpu_memory_utilization=0.6 \
  trainer.epochs=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=512 \
  trainer.policy_mini_batch_size=256 \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_FORWARD_BATCH_SIZE_PER_GPU \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BATCH_SIZE_PER_GPU \
  trainer.max_prompt_length=2048 \
  generator.max_input_length=4096 \
  generator.sampling_params.max_generate_length=500 \
  generator.async_engine=true \
  generator.batched=false \
  generator.use_conversation_multi_turn=false \
  generator.n_samples_per_prompt=5 \
  generator.max_turns=4 \
  generator.sampling_params.temperature=1.0 \
  generator.sampling_params.top_p=1.0 \
  generator.sampling_params.stop='["</search>", "</answer>"]' \
  environment.env_class="search" \
  environment.skyrl_gym.max_env_workers=16 \
  environment.skyrl_gym.search.log_requests=false \
  environment.skyrl_gym.search.search_url="http://172.25.102.175:8000/retrieve" \
  environment.skyrl_gym.search.topk=3 \
  trainer.logger="wandb" \
  trainer.project_name="skyrl-search" \
  trainer.run_name="skyrl-search_4turns_maxgeneratelen_500_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_qwen30b" \
  trainer.ckpt_interval=20 \
  trainer.hf_save_interval=100 \
  trainer.max_ckpts_to_keep=5 \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/skyrl-search_4turns_maxgeneratelen_500_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_qwen30b" \
  trainer.eval_batch_size=256 \
  trainer.eval_before_train=false \
  generator.eval_sampling_params.temperature=0 \
  generator.eval_sampling_params.stop='["</search>", "</answer>"]' \
  trainer.export_path="$HOME/skyrl-search_4turns_maxgeneratelen_500_megatron_tp${MEGATRON_TP}_pp${MEGATRON_PP}_cp${MEGATRON_CP}_qwen30b/exports" \
  trainer.eval_interval=50 \
  $@
  