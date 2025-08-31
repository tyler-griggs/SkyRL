
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from omegaconf import DictConfig
from openai import AsyncOpenAI
import httpx
from skyrl_train.inference_engines.launch_inference_engine_http_server import generate_with_http_server

from verifiers import load_environment
from verifiers.types import GenerateOutputs, ProcessedOutputs

# Issues along the way:
# 1) Formatting dataset is annoying. We should be able to take a HF dataset as input, and if we do want a special dataset format it should be better defined
    # Consider how the dataset should be propagated through the trainer. Do we even want it to be propagated through the trainer?
    # --> this is a good undergrad task
# 2) Need HTTP support, and need it to be smooth.
# 3) How to install env? Do you need to run [uv run prime env install ...]. Do you need to drop isolated? Or have some way of picking up new envs from prime intellect hub?


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
        
        self.use_http_server_inference_engine_client = generator_cfg.get(
            "use_http_server_inference_engine_client", False
        )
        self.http_server_inference_engine_client_host = generator_cfg.get(
            "http_server_inference_engine_client_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_cfg.get(
            "http_server_inference_engine_client_port", 8000
        )
        
        if self.use_http_server_inference_engine_client:
            # Store the base URL for direct HTTP requests
            self.base_url = f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        else:
            self.base_url = None

    def _setup_client(self) -> AsyncOpenAI:
      # We use a longer request timeout than default, but if more than 20min, we probably need faster inference deployment
      timeout = httpx.Timeout(timeout=60, connect=5.0)
      # We use as many concurrent connections as possible, but lower than available ports
      limits = httpx.Limits(
          max_connections=28000,  # OAI default: 1000
          max_keepalive_connections=28000,  # OAI default: 100
      )
      http_client = httpx.AsyncClient(limits=limits, timeout=timeout)
      base_url = f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}/v1"
      print(f"Using AsyncOpenAI with base_url: {base_url}")
      return AsyncOpenAI(
          base_url=base_url,
          api_key="dummy",
          max_retries=10,  # OAI default: 2 (does exponential backoff and reasonable timeout in between retries)
          http_client=http_client,
      )

    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
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
        
        engine_input = {
            "prompts": [[{"role": "user", "content": "Hello, how are you?"}]],
            "trajectory_ids": None,
            "sampling_params": None,
        }
        
        
        output = await generate_with_http_server(
            base_url=self.base_url,
            model_name=self.model_name,
            input_batch=engine_input,
        )
        
        print(f"Got output for generate_with_http_server: {output}")
        
        # TODO(tgriggs): Make a test call to the client.
        response = await client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": "Hello, how are you?"}],
        )
        print(f"Got response: {response}")

        # Call a_generate with the dataset directly
        generate_outputs: GenerateOutputs = await vf_env.a_generate(
            inputs=train_batch,
            client=client,                 # to be wired later
            model=self.model_name,         # store model_name in __init__
            # sampling_args=getattr(self.generator_cfg, "sampling_args", None),
            # max_concurrent=getattr(self.generator_cfg, "max_concurrent", -1),
        )
        
        print(f"Got outputs: {generate_outputs}")