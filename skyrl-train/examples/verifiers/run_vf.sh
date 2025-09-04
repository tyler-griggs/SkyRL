set -x

DATA_DIR="$HOME/data/verifiers/wordle"
NUM_GPUS=1
LOGGER="console"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"
# INFERENCE_BACKEND="sglang"

# uv run --extra $INFERENCE_BACKEND --extra verifiers -m skyrl_train.entrypoints.main_verifiers \
# uv run --isolated --with wordle==0.1.4 --extra-index-url https://hub.primeintellect.ai/will/simple/ --extra $INFERENCE_BACKEND --with "-e /home/ubuntu/tgriggs/SkyRL/skyrl-train/verifiers" -m skyrl_train.entrypoints.main_verifiers \
uv run --isolated --with-editable 'verifiers@file:///home/ubuntu/tgriggs/SkyRL/skyrl-train/verifiers' --with wordle==0.1.4 --extra-index-url https://hub.primeintellect.ai/will/simple/ --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_verifiers \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path="Qwen/Qwen2.5-0.5B-Instruct" \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=20 \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=-1 \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=2 \
  trainer.policy_mini_batch_size=2 \
  trainer.micro_forward_batch_size_per_gpu=1 \
  trainer.micro_train_batch_size_per_gpu=1 \
  trainer.ckpt_interval=-1 \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=1024 \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.enable_http_endpoint=true \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=2 \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="vf" \
  trainer.run_name="vf_wordle_test" \
  trainer.resume_mode=null \
  trainer.ckpt_path="$HOME/ckpts/gsm8k_1.5B_ckpt" \
  $@