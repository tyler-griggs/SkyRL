"""
uv run --isolated --extra dev --extra vllm --with verifiers pytest tests/gpu/gpu_ci/test_verifiers_generator.py
"""

import pytest
import ray
import hydra
from omegaconf import DictConfig
from transformers import AutoTokenizer
import socket

from skyrl_train.inference_engines.ray_wrapped_inference_engine import create_ray_wrapped_inference_engines
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.utils import get_sampling_params_for_backend
from integrations.verifiers.generator.verifiers_generator import VerifiersGenerator
from skyrl_train.utils.utils import initialize_ray
from skyrl_train.entrypoints.main_base import config_dir

# Mark all tests in this file as "integrations"
pytestmark = pytest.mark.integrations


def _get_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _get_base_cfg() -> DictConfig:
    with hydra.initialize_config_dir(config_dir=config_dir):
        cfg = hydra.compose(config_name="ppo_base_config")
        cfg.generator.backend = "vllm"
        return cfg


@pytest.fixture(scope="module")
def verifiers_runtime():
    model = "Qwen/Qwen2.5-1.5B-Instruct"
    http_port = _get_free_port()

    initialize_ray(_get_base_cfg())

    tokenizer = AutoTokenizer.from_pretrained(model)
    inference_engines = create_ray_wrapped_inference_engines(
        num_inference_engines=1,
        tensor_parallel_size=1,
        model_dtype="bfloat16",
        pretrain=model,
        seed=42,
        vllm_v1_disable_multiproc=True,
        enable_prefix_caching=True,
        enforce_eager=True,
        max_model_len=3072,
        shared_pg=None,
        gpu_memory_utilization=0.8,
        inference_engine_enable_sleep=True,
        async_engine=True,
        max_num_batched_tokens=8192,
        max_num_seqs=1024,
        tokenizer=tokenizer,
    )

    cfg = _get_base_cfg()
    cfg.trainer.policy.model.path = model
    cfg.generator = DictConfig(
        {
            "sampling_params": {"max_generate_length": 256, "logprobs": None},
            "max_input_length": 2048,
            "backend": "vllm",
            "enable_http_endpoint": True,
            "http_endpoint_host": "127.0.0.1",
            "http_endpoint_port": http_port,
        }
    )

    client = InferenceEngineClient(inference_engines, tokenizer, cfg)

    try:
        yield {"client": client, "tokenizer": tokenizer, "http_port": http_port, "model": model}
    finally:
        ray.shutdown()


async def _run_verifiers_end_to_end(
    *,
    existing_client: InferenceEngineClient,
    model: str = "Qwen/Qwen2.5-1.5B-Instruct",
    num_prompts: int = 2,
    http_host: str = "127.0.0.1",
    http_port: int | None = None,
    max_input_length: int = 2048,
    max_generate_length: int = 256,
    sampling_overrides: dict | None = None,
    prompt_text: str | None = None,
    existing_tokenizer=None,
):
    client = existing_client
    tokenizer = existing_tokenizer
    if http_port is None:
        http_port = _get_free_port()

    await client.wake_up()

    generator_cfg = DictConfig(
        {
            "sampling_params": {"max_generate_length": max_generate_length, "logprobs": None},
            "max_input_length": max_input_length,
            "backend": "vllm",
            "enable_http_endpoint": True,
            "http_endpoint_host": http_host,
            "http_endpoint_port": http_port,
        }
    )

    generator = VerifiersGenerator(
        generator_cfg=generator_cfg,
        inference_engine_client=client,
        tokenizer=tokenizer,
        model_name=model,
    )

    prompts = [
        [
            {
                "role": "user",
                "content": prompt_text
                or "You are playing Wordle. Think step-by-step and propose the next guess based on previous feedback.",
            }
        ]
        for _ in range(num_prompts)
    ]
    env_extras = [
        {"verifiers": {"environment": "wordle", "answer": "", "info": {}, "task": "default"}}
        for _ in range(num_prompts)
    ]

    base_sampling = DictConfig(
        {
            "temperature": 0.7,
            "top_p": 0.95,
            "top_k": -1,
            "max_generate_length": max_generate_length,
            "min_p": 0.0,
            "logprobs": None,
            "stop": None,
        }
    )
    if sampling_overrides:
        for k, v in sampling_overrides.items():
            base_sampling[k] = v

    sampling_params = get_sampling_params_for_backend("vllm", base_sampling)

    input_batch = {
        "prompts": prompts,
        "env_extras": env_extras,
        "sampling_params": sampling_params,
    }

    output = await generator.generate(input_batch)
    return output


@pytest.mark.asyncio
async def test_verifiers_e2e_wordle_http(verifiers_runtime):
    rt = verifiers_runtime
    out = await _run_verifiers_end_to_end(
        existing_client=rt["client"],
        model=rt["model"],
        num_prompts=2,
        max_input_length=2048,
        max_generate_length=256,
        http_host="127.0.0.1",
        http_port=rt["http_port"],
        existing_tokenizer=rt["tokenizer"],
    )

    for key in [
        "prompt_token_ids",
        "response_ids",
        "rewards",
        "loss_masks",
        "rollout_logprobs",
        "rollout_metrics",
    ]:
        assert key in out

    assert len(out["response_ids"]) == 2
    assert len(out["prompt_token_ids"]) == 2
    for resp, mask, logp in zip(
        out["response_ids"], out["loss_masks"], out["rollout_logprobs"] or [[]] * len(out["response_ids"])
    ):
        assert isinstance(resp, list) and all(isinstance(t, int) for t in resp)
        assert len(resp) == len(mask)
        if out["rollout_logprobs"] is not None:
            assert len(logp) == len(resp)


@pytest.mark.asyncio
async def test_verifiers_e2e_sampling_toggles(verifiers_runtime):
    rt = verifiers_runtime
    out = await _run_verifiers_end_to_end(
        existing_client=rt["client"],
        model=rt["model"],
        num_prompts=2,
        max_input_length=2048,
        max_generate_length=128,
        sampling_overrides={
            "skip_special_tokens": True,
            "include_stop_str_in_output": False,
            "top_k": 20,
            "min_p": 0.05,
            "repetition_penalty": 1.05,
            "min_tokens": 1,
        },
        http_host="127.0.0.1",
        http_port=rt["http_port"],
        existing_tokenizer=rt["tokenizer"],
    )

    assert len(out["response_ids"]) == 2
    for resp, mask in zip(out["response_ids"], out["loss_masks"]):
        assert len(resp) == len(mask)


@pytest.mark.asyncio
async def test_verifiers_length_constraints(verifiers_runtime):
    rt = verifiers_runtime
    max_input_length = 512
    max_generate_length = 64
    long_prompt = "Wordle analysis: " + ("A" * 400)

    out = await _run_verifiers_end_to_end(
        existing_client=rt["client"],
        model=rt["model"],
        num_prompts=1,
        max_input_length=max_input_length,
        max_generate_length=max_generate_length,
        prompt_text=long_prompt,
        http_host="127.0.0.1",
        http_port=rt["http_port"],
        existing_tokenizer=rt["tokenizer"],
    )

    limit = max_input_length + max_generate_length
    resp = out["response_ids"][0]
    assert len(resp) <= limit
    assert len(resp) == len(out["loss_masks"][0])
