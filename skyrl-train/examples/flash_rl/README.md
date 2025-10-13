## Installation

The `flashrl` extra requires a custom vllm wheel that is not available on PyPI. You must install it manually.

## Install the custom vllm wheel

```bash
uv pip install https://github.com/NovaSky-AI/SkyRL/releases/download/skyrl_train-v0.1.0/vllm-0.1.dev7509+gcc487699a.d20250821-cp312-cp312-linux_x86_64.whl
```

## Running the Examples

After installation, you can run the Flash RL examples:

```bash
bash run_dapo_gsm8k_flashrl_0.5b_int8.sh
```

See the other `.sh` files in this directory for more examples.

