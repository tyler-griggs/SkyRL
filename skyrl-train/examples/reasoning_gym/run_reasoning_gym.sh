export INFERENCE_BACKEND="vllm"
export DATA_DIR="$HOME/data/reasoning_gym"
export LOGGER="console"


uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=1 \
  trainer.placement.ref_num_gpus_per_node=1 \
  generator.num_inference_engines=1 \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=1 \
  trainer.eval_batch_size=32 \
  trainer.eval_before_train=true \
  trainer.eval_interval=1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=32 \
  trainer.policy_mini_batch_size=8 \
  trainer.micro_forward_batch_size_per_gpu=4 \
  trainer.micro_train_batch_size_per_gpu=4 \
  trainer.ckpt_interval=1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=256 \
  trainer.policy.optimizer_config.lr=5.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=2 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="gsm8k" \
  trainer.run_name="gsm8k_test_small" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_0.5B_ckpt" \
  $@
