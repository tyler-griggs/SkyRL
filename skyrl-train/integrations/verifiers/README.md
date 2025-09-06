## Verifiers + SkyRL Guide

This directory holds the workflow to train on Verifiers environments with SkyRL.

To start training, follow three simple steps:
1) Install the environment from Environments Hub
2) Prepare the environment's training and validation datasets
3) Launch training!

Start by entering the `skyrl-train` directory:
```bash
cd SkyRL/skyrl-train
```

### 1) Install the environment
Specify your desired `ENV_ID` from the [Environments Hub](https://app.primeintellect.ai/dashboard/environments) and run the following to install the environment and add it to the `uv` project:
```bash
bash integrations/verifiers/install_environment.sh <ENV_ID>
```
For example:
```bash
bash integrations/verifiers/install_environment.sh will/wordle
```

### 2) Prepare the dataset
Next, load the environment's dataset and convert to SkyRL format:
```bash
bash integrations/verifiers/prepare_dataset.sh <ENV_ID>
```
For example:
```bash
bash integrations/verifiers/prepare_dataset.sh will/wordle
```
This will:
- Resolve and install the environment specified by `ENV_ID`
- Generate Parquet files under `DATA_DIR` (default: `$HOME/data/$ENV_ID`):
  - `train.parquet`
  - `validation.parquet`, if included in the environment

Notes:
- For issues in loading the dataset, see the Troubleshooting section below.
- Internally, the script runs `verifiers_dataset.py`, which accepts optional dataset limits:
  - `--num_train` and `--num_eval` (set to `-1` for no limit). You can run it directly if you need custom sizing.

### 3) Launch training
Open `run_verifiers.sh`, which specifies the training configuration parameters and is the primary interface for launching training runs.

Set `ENV_ID="<ENV_ID>"`, such as `ENV_ID="will/wordle"`, then launch your training run:

```bash
bash integrations/verifiers/run_verifiers.sh
```

To change basic training settings, edit the variables at the top of `run_verifiers.sh`:
- `DATA_DIR`: path where the Parquet files live -- defaults to the same as `prepare_dataset.sh`
- `NUM_GPUS`: number of GPUs to use for training and generation.
- `LOGGER`: export training statistics to `console` or `wandb`

You can also modify other config overrides when running the script and they will be forwarded to the trainer, such as the model choice (`trainer.policy.model.path"`, GRPO group size (`generator.n_samples_per_prompt`), or training batch size (`trainer.train_batch_size`). See all training configuration parameters in `ppo_base_config.yaml`.


## Troubleshooting

For issues with SkyRL or the integration with Verifiers, please [create an Issue](https://github.com/NovaSky-AI/SkyRL/issues/new). 


### Datasets
Verifiers environments can handle dataset splits in different ways. Some environments require passing a `dataset_split` argument to `load_environment()` (e.g., to specify `train` vs `test`), others implement both `vf_env.load_dataset()` and `vf_env.load_eval_datset()`. The implementation in `verifiers_dataset.py` assumes the latter approach to get datasets, which may be incorrect for some environments. Please modify `verifiers_dataset.py` as needed to extract and prepare the correct datasets.


## TODOs and Limitations
* Make it easier to use different Verifiers environments for training and validation.
* Make it smoother to specify which dataset splits to use.
* Consider plumbing Verifiers-specific config to the VerifiersGenerator. For example: `zero_truncated_completions` and `mask_truncated_completions`.