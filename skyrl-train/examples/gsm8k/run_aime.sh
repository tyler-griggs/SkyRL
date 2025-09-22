set -x

DATA_DIR="$HOME/data/aime"
# TODO: put train.parquet and validation.parquet into $DATA_DIR

NUM_GPUS=8
LOGGER="wandb"  # change to "console" to print to stdout

INFERENCE_BACKEND="vllm"

GROUP_SIZE=12
VALIDATION_INTERVAL=20
EPOCHS=2
MODEL="Qwen/Qwen3-8B"
MODEL_SAVE_INTERVAL=20

TRAIN_BATCH_SIZE=32
MICRO_TRAIN_BSZ_PERGPU=8
MAX_GEN_LENGTH=8192

uv run --isolated --extra $INFERENCE_BACKEND -m skyrl_train.entrypoints.main_base \
  data.train_data="['$DATA_DIR/train.parquet']" \
  data.val_data="['$DATA_DIR/validation.parquet', '$DATA_DIR/test.parquet']" \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.policy.model.path=$MODEL \
  trainer.placement.colocate_all=true \
  trainer.strategy=fsdp2 \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  trainer.epochs=$EPOCHS \
  trainer.eval_batch_size=1024 \
  trainer.eval_before_train=false \
  trainer.eval_interval=$VALIDATION_INTERVAL \
  trainer.update_epochs_per_batch=1 \
  trainer.train_batch_size=$TRAIN_BATCH_SIZE \
  trainer.policy_mini_batch_size=$TRAIN_BATCH_SIZE \
  trainer.micro_forward_batch_size_per_gpu=$MICRO_TRAIN_BSZ_PERGPU \
  trainer.micro_train_batch_size_per_gpu=$MICRO_TRAIN_BSZ_PERGPU \
  trainer.policy.sequence_parallel_size=2 \
  trainer.ckpt_interval=-1 \
  trainer.hf_save_interval=$MODEL_SAVE_INTERVAL \
  trainer.max_prompt_length=512 \
  generator.sampling_params.max_generate_length=$MAX_GEN_LENGTH \
  trainer.policy.optimizer_config.lr=1.0e-6 \
  trainer.algorithm.use_kl_loss=true \
  generator.backend=$INFERENCE_BACKEND \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.batched=true \
  environment.env_class=gsm8k \
  generator.n_samples_per_prompt=$GROUP_SIZE \
  generator.gpu_memory_utilization=0.8 \
  trainer.logger="$LOGGER" \
  trainer.project_name="aime" \
  trainer.run_name="aime_test" \
  trainer.resume_mode=null \
  trainer.export_path="$HOME/exports" \
  trainer.ckpt_path="$HOME/ckpts/aime_0.5B_ckpt" \
  $@