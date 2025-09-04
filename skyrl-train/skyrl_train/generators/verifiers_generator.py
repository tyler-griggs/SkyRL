
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from omegaconf import DictConfig
from openai import AsyncOpenAI
import httpx
# from skyrl_train.inference_engines.launch_inference_engine_http_server import generate_with_http_server

from verifiers import load_environment
from verifiers.types import GenerateOutputs, ProcessedOutputs, SamplingArgs, GenerateInputs

# TODO: First e2e demo
# 1) DONE Confirm generation
# 2) DONE Pipe dataset through trainer to generator (or consider not...)
# 3) DONE Send vf env outputs back
# 4) Test eval()
# 5) See curve actually go up... 
# 6) Next: test other envs. Correctness and performance checks?
# 7) Solve general installation process, make it clean

# Issues along the way:
# 1) Formatting dataset is annoying. We should be able to take a HF dataset as input, and if we do want a special dataset format it should be better defined
    # Consider how the dataset should be propagated through the trainer. Do we even want it to be propagated through the trainer?
    # First PR: remove our dataset preprocessing and required format. You just need a Dataset object and the Generator can define what fields are necessary.
# 2) Need HTTP support, and need it to be smooth.
    # a) NEED logprobs from HTTP server
# 3) How to install env? Do you need to run [uv run prime env install ...]. Do you need to drop isolated? Or have some way of picking up new envs from prime intellect hub?
# 4) Examples. There should be a README for every example discussing what's going on and how to run it and set up the dataset.
# 5) Evals: min_tokens doesn't seem to work for chat completions? Seems like they want sampling args as per OAI?

# Option 1: add dataset preprocessing where we intialize the env and then get a dataset. We pass this to the trainer, etc.
# Option 2: 

# TODOs on dataset format:
# Keep using GenerateInputs only with fully-formed prompts.
# Don’t hardcode env id; use env_class or configuration.
# Add return_tokens_as_token_ids=True (or your server’s equivalent) in extra_body when you need token ids.
# Default answer and task in the generator to avoid KeyErrors.
# Ensure sampling args for logprobs are set consistently in both generate(...) and test_generate(...).
# In short: align env selection with data, make prompt shapes consistent with message_type, ensure token/logprob flags match your vLLM server’s expectations, and make the generator resilient to missing fields.

class VerifiersGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
        model_name: str,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.generator_cfg = generator_cfg
        self.inference_engine_client = inference_engine_client
        self.tokenizer = tokenizer
        self.model_name = model_name
        
        assert generator_cfg.enable_http_endpoint, "HTTP endpoint must be enabled for VerifiersGenerator"
        
        # Store the base URL for direct HTTP requests
        self.base_url = f"http://{generator_cfg.http_endpoint_host}:{generator_cfg.http_endpoint_port}/v1"


    def _setup_client(self) -> AsyncOpenAI:
      # We use a longer request timeout than default, but if more than 20min, we probably need faster inference deployment
      timeout = httpx.Timeout(timeout=60, connect=5.0)
      # We use as many concurrent connections as possible, but lower than available ports
      limits = httpx.Limits(
          max_connections=28000,  # OAI default: 1000
          max_keepalive_connections=28000,  # OAI default: 100
      )
      http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
      print(f"Using AsyncOpenAI with base_url: {self.base_url}")
      return AsyncOpenAI(
          base_url=self.base_url,
          api_key="dummy",
          max_retries=10,  # OAI default: 2 (does exponential backoff and reasonable timeout in between retries)
          http_client=http_client,
      )

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        # TODO(tgriggs): Get the config from the verifiers config? For now hard code it.
        # vf_env = load_environment(config.environment.id, **config.environment.args)
            
        print(f"Got input_batch: {input_batch}")
        verifiers_dicts = [sample["verifiers"] for sample in input_batch["env_extras"]]
        print(f"Got verifiers_dicts: {verifiers_dicts}")

        # Defaults are based on Verifiers' defaults.
        generate_inputs = GenerateInputs(
            prompt=input_batch["prompts"],
            answer=[dict.get("answer", "") for dict in verifiers_dicts],
            info=[dict.get("info", {}) for dict in verifiers_dicts],
            task=[dict.get("task", "default") for dict in verifiers_dicts],
        )
        
        # NOTE: Currently assumes all training samples correspond to the same Verifiers environment.
        # If multiple environments are needed, use Verifiers' EnvGroup abstraction.
        print("Loading environment...")
        environment_id = verifiers_dicts[0]["environment"]
        vf_env = load_environment(environment_id)
        
        print(f"Got generate_inputs: {generate_inputs}")
        
        
        sampling_params = input_batch.get("sampling_params", None)
        if sampling_params is None:
            sampling_params = {}
        sampling_params["logprobs"] = True
        sampling_params["top_logprobs"] = 1
        sampling_params["extra_body"] = {
            "return_tokens_as_token_ids": True,
        }
        
        # Clean the sampling params
        extra_body_keys = ["min_tokens", "skip_special_tokens", "include_stop_str_in_output", "top_k", "min_p", "repetition_penalty"]
        for key in extra_body_keys:
            if key in sampling_params:
                sampling_params["extra_body"][key] = sampling_params[key]
                del sampling_params[key]

        # Call a_generate with the dataset directly
        client = self._setup_client()
        generate_outputs: GenerateOutputs = await vf_env.a_generate(
            inputs=generate_inputs,
            client=client,
            model=self.model_name,
            sampling_args=sampling_params,
        )
        
        print(f"Got outputs: {generate_outputs}")
        print(f"Example\nPrompt: {generate_outputs.prompt[0]}\nResponse: {generate_outputs.completion[0]}\nAnswer: {generate_outputs.answer[0]}\nReward: {generate_outputs.reward[0]}")
        print(f"Highest reward: {max(generate_outputs.reward)}")
        
        processed_outputs: ProcessedOutputs = vf_env.process_env_results_vllm(
            prompts=generate_outputs.prompt,
            completions=generate_outputs.completion,
            states=generate_outputs.state,
            rewards=generate_outputs.reward,
            processing_class=self.tokenizer,
            max_seq_len=self.generator_cfg.get("max_seq_len", -1),
            mask_env_responses=self.generator_cfg.get("mask_env_responses", True),
        )
        
        print(f"Processed outputs: {processed_outputs}")

        return GeneratorOutput(
            prompt_token_ids=processed_outputs.prompt_ids,
            response_ids=processed_outputs.completion_ids,
            rewards=processed_outputs.rewards,
            loss_masks=processed_outputs.completion_mask,
            stop_reasons=None,
            rollout_logprobs=processed_outputs.completion_logprobs,
            rollout_metrics=None,
        )