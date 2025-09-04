
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
# 2) Pipe dataset through trainer to generator (or consider not...)
# 3) Send vf env outputs back
# 4) Test eval()
# 5) Next: test other envs. Correctness and performance checks?
# 6) Solve general installation process


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
        # TODO(tgriggs): To start, just load_environment(), pass in client, and call a_generate().
        # Where do we get the dataset? Do we need to do anything special to it?
        
        # TODO(tgriggs): Get the config from the verifiers config? For now hard code it.
        # vf_env = load_environment(config.environment.id, **config.environment.args)
            
        print(f"Got input_batch: {input_batch}")
        verifiers_dicts = [sample["verifiers"] for sample in input_batch["env_extras"]]
        print(f"Got verifiers_dicts: {verifiers_dicts}")

        # Defaults based on Verifiers defaults.
        generate_inputs = GenerateInputs(
            prompt=input_batch["prompts"],
            answer=[dict.get("answer", "") for dict in verifiers_dicts],
            info=[dict.get("info", {}) for dict in verifiers_dicts],
            task=[dict.get("task", "default") for dict in verifiers_dicts],
        )
        
        print(f"Got generate_inputs: {generate_inputs}")
        
        client = self._setup_client()
        
        sampling_params = input_batch.get("sampling_params", None)
        if sampling_params is None:
            sampling_params = {}
        sampling_params["logprobs"] = True
        sampling_params["top_logprobs"] = 1
        sampling_params["extra_body"] = {
            "return_tokens_as_token_ids": True,
        }
        
        extra_body_keys = ["min_tokens", "skip_special_tokens", "include_stop_str_in_output", "top_k", "min_p", "repetition_penalty"]
        
        # Clean sampling params
        for key in extra_body_keys:
            if key in sampling_params:
                sampling_params["extra_body"][key] = sampling_params[key]
                del sampling_params[key]
        
        print("Loading environment...")
        vf_env = load_environment("wordle")

        # Call a_generate with the dataset directly
        generate_outputs: GenerateOutputs = await vf_env.a_generate(
            inputs=generate_inputs,
            client=client,
            model=self.model_name,
            sampling_args=sampling_params,
        )
        
        print(f"Got outputs: {generate_outputs}")
        print(f"Example\nPrompt: {generate_outputs.prompt[0]}\nResponse: {generate_outputs.completion[0]}\nAnswer: {generate_outputs.answer[0]}\nReward: {generate_outputs.reward[0]}")
        print(f"Highest reward: {max(generate_outputs.reward)}")
        
        
        # class GenerateOutputs(BaseModel):
        #     """Pydantic model for generation outputs."""
        #     prompt: list[Messages]
        #     completion: list[Messages]
        #     answer: list[str]
        #     state: list[State]
        #     info: list[Info]
        #     task: list[str]
        #     reward: list[float]
        #     metrics: dict[str, list[float]] = Field(default_factory=dict)
        
        # class GeneratorOutput(TypedDict):
        #     prompt_token_ids: List[List[int]]
        #     response_ids: List[List[int]]
        #     rewards: Union[List[float], List[List[float]]]
        #     loss_masks: List[List[int]]
        #     stop_reasons: Optional[List[str]]
        #     rollout_metrics: Optional[Dict[str, Any]]
        #     rollout_logprobs: Optional[List[List[float]]]
        
        # TODO: convert generate_outputs to GeneratorOutput
        processed_outputs: ProcessedOutputs = vf_env.process_env_results_vllm(
            prompts=generate_outputs.prompt,
            completions=generate_outputs.completion,
            states=generate_outputs.state,
            rewards=generate_outputs.reward,
            processing_class=self.tokenizer,
            max_seq_len=self.generator_cfg.get("max_seq_len", -1),
            mask_env_responses=self.generator_cfg.get("mask_env_responses", True),
        )

        return GeneratorOutput(
            prompt_token_ids=processed_outputs.prompt_ids,
            response_ids=processed_outputs.completion_ids,
            rewards=processed_outputs.rewards,
            loss_masks=processed_outputs.completion_mask,
            stop_reasons=None,
            rollout_logprobs=processed_outputs.completion_logprobs,
            rollout_metrics=None,
        )



    async def test_generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        # TODO(tgriggs): To start, just load_environment(), pass in client, and call a_generate().
        # Where do we get the dataset? Do we need to do anything special to it?
        
        # TODO(tgriggs): Get the config from the verifiers config? For now hard code it.
        # vf_env = load_environment(config.environment.id, **config.environment.args)
        
        print("Loading environment")
        
        vf_env = load_environment("wordle")
        
        batch_size = 8
        train_batch = vf_env.get_dataset(n=batch_size)
        
        print(f"Got train_batch: {train_batch}")
        
        # TODO(tgriggs): Build a client
        client = self._setup_client()
        
        print(f"Setup client")
        
        # engine_input = {
        #     "prompts": [[{"role": "user", "content": "Hello, how are you?"}]],
        #     "trajectory_ids": None,
        #     "sampling_params": None,
        # }
        
        # output = await generate_with_http_server(
        #     base_url=self.base_url,
        #     model_name=self.model_name,
        #     input_batch=engine_input,
        # )
        
        # print(f"Got output for generate_with_http_server: {output}")
        
        # TODO(tgriggs): Make a test call to the client.
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )
        print(f"Got response: {response}")
        
        sampling_args : SamplingArgs = {
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_tokens": 1024,
            "min_p": 0.0,
        }

        # Call a_generate with the dataset directly
        generate_outputs: GenerateOutputs = await vf_env.a_generate(
            inputs=train_batch,
            client=client,                 # to be wired later
            model=self.model_name,         # store model_name in __init__
            # sampling_args=getattr(self.generator_cfg, "sampling_args", None),
            # max_concurrent=getattr(self.generator_cfg, "max_concurrent", -1),
        )
        
        print(f"Got outputs: {generate_outputs}")
        print(f"Example\nPrompt: {generate_outputs.prompt[0]}\nResponse: {generate_outputs.completion[0]}\nAnswer: {generate_outputs.answer[0]}\nReward: {generate_outputs.reward[0]}")
        print(f"Highest reward: {max(generate_outputs.reward)}")