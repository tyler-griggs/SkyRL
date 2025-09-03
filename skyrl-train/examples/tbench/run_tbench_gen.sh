set -x

# WORK IN PROGRESS
# Colocated GRPO training+generation for Qwen2.5-1.5B-Instruct on TBench tasks.

# uv run examples/gsm8k/gsm8k_dataset.py --output_dir $HOME/data/gsm8k
# export WANDB_API_KEY=<your_key_here>
# bash examples/tbench/run_tbench.sh

NUM_GPUS=1
LOGGER="wandb"  # change to "console" to print to stdout

uv run --isolated --extra vllm --extra sandboxes -m examples.tbench.main_tbench_generate \
  trainer.policy.model.path="Qwen/Qwen2.5-1.5B-Instruct" \
  generator.num_inference_engines=$NUM_GPUS \
  generator.inference_engine_tensor_parallel_size=1 \
  generator.enable_http_endpoint=true \
  generator.http_endpoint_host="127.0.0.1" \
  generator.http_endpoint_port=8000 \
  generator.sampling_params.max_generate_length=4096 \
  generator.backend=vllm \
  generator.run_engines_locally=true \
  generator.weight_sync_backend=nccl \
  generator.async_engine=true \
  generator.gpu_memory_utilization=0.8 \
  trainer.algorithm.advantage_estimator="grpo" \
  trainer.placement.colocate_all=true \
  trainer.placement.policy_num_gpus_per_node=$NUM_GPUS \
  trainer.placement.ref_num_gpus_per_node=$NUM_GPUS \
  trainer.logger="$LOGGER" \
  $@