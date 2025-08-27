import asyncio
from typing import List, Tuple
from skyrl_train.generators.base import GeneratorInterface, GeneratorInput, GeneratorOutput
from skyrl_train.inference_engines.inference_engine_client import InferenceEngineClient
from skyrl_train.inference_engines.base import ConversationType
from omegaconf import DictConfig
from pathlib import Path
from sandbox.models.trial.config import TrialConfig, AgentConfig, LocalTaskConfig
from sandbox.models.task.id import LocalTaskId
from sandbox.models.agent.name import AgentName
from sandbox.trial.trial import Trial
import numpy as np

class TBenchGenerator(GeneratorInterface):
    def __init__(
        self,
        generator_cfg: DictConfig,
        inference_engine_client: InferenceEngineClient,
        tokenizer,
    ):
        """
        Args:
            generator_cfg: DictConfig object containing the generator configuration
            inference_engine_client: InferenceEngineClient object for interacting with the inference engines
            tokenizer: tokenizer object for encoding and decoding text
        """
        self.http_server_inference_engine_client_host = generator_cfg.get(
            "http_server_inference_engine_client_host", "127.0.0.1"
        )
        self.http_server_inference_engine_client_port = generator_cfg.get(
            "http_server_inference_engine_client_port", 8000
        )
        self.base_url = f"http://{self.http_server_inference_engine_client_host}:{self.http_server_inference_engine_client_port}"
        # [Marianna] set trial dir as environment var for testing (permission denied)
        self.generator_cfg = generator_cfg
        self.tokenizer = tokenizer
        
        # TODO(tgriggs): Should turns be in the agent configuration? Currently it's just used to estimate overall trajectory length limits.
        self.max_turns = generator_cfg.max_turns
        self.model_name = generator_cfg.model_name
        
        
    # TODO(tgriggs): Create a tbench subconfig in training data to pass along metadata (e.g,. task)
    async def generate(self, input_batch: GeneratorInput) -> GeneratorOutput:
        prompts = input_batch["prompts"]
            
        tasks = []

        print(f"About to start agent {len(prompts)} tbench_agent_loop instances")

        for i in range(len(prompts)):
            tasks.append(
                self.tbench_agent_loop(
                    prompt="Hello, how are you?",
                    max_tokens=1024,
                    max_input_length=1024,
                )
            )
 
        all_outputs = await asyncio.gather(*tasks)

        responses = [output[0] for output in all_outputs]
        rewards = [output[1] for output in all_outputs]
        stop_reasons = [output[2] for output in all_outputs]
        loss_masks = [output[3] for output in all_outputs]
        prompt_token_ids = [output[4] for output in all_outputs]
        rollout_metrics = self._rollout_metrics(responses, rewards)

        generator_output: GeneratorOutput = {
            "prompt_token_ids": prompt_token_ids,
            "response_ids": responses,
            "rewards": rewards,
            "loss_masks": loss_masks,
            "stop_reasons": stop_reasons,
            "rollout_metrics": rollout_metrics,
            "rollout_logprobs": None,
        }

        return generator_output

    # TODO(tgriggs): Debug this error:
    # ValueError: The decoder prompt (length 1557) is longer than the maximum model length of 1536. Make sure that `max_model_len` is no smaller than the number of text tokens.
    # The max model length is set when constructing the inference engine:
    #   max_model_len=cfg.generator.max_input_length + cfg.generator.sampling_params.max_generate_length,
    #  We need to ensure this aligns with tbench's constraints. 

    # TODO(tgriggs): Where are the sampling params used by the agent defined? More generally, how should we pass tbench config? What needs to be passed?
    async def tbench_agent_loop(
        self,
        prompt: ConversationType,
        max_tokens: int,
        max_input_length: int,
    ) -> Tuple[List[int], float, str, List[int], List[int]]:
        """
        Run a single tbench agent.
        """  
        # TODO(tgriggs): Get these inputs from the config, use tbench config
        # trials_dir = self.generator_cfg.get("trial_runs_dir")
        trials_dir = "/home/ubuntu/tgriggs/trials"
        agent_name = "terminus"
        sandboxes_dir = "/home/ubuntu/tgriggs/SkyRL/skyrl-train/sandboxes"

        if agent_name == "terminus":
            trial_config = TrialConfig(
                task=LocalTaskConfig(id=LocalTaskId(path=f"{sandboxes_dir}/examples/tasks/hello-world")),
                trials_dir=Path(trials_dir),
                agent=AgentConfig(
                    name=AgentName.TERMINUS_2.value,
                    model_name=f"{self.model_name}",
                    kwargs={"api_base": f"{self.base_url}/v1", "key": "fake_key"},
                    max_episodes=1,
                )
            )
        elif agent_name == "oracle":
            trial_config = TrialConfig(
                task=LocalTaskConfig(id=LocalTaskId(path=f"{sandboxes_dir}/examples/tasks/hello-world")),
                trials_dir=Path(trials_dir),
                agent=AgentConfig(
                    name=AgentName.ORACLE,
                    model_name=self.model_name,
                    max_episodes=1,
                )
            )
        else:
            raise ValueError(f"Invalid agent name: {agent_name}")
        
        trial = Trial(trial_config)
        print(f"About to start agent {agent_name}")
        # Run the trial
        while True:
            results = await trial.run()
            reward = results.verifier_result.rewards
            chat_history = results.agent_result.all_messages
            if len(chat_history) > 0:
                break
            else:
                print(f"[WARNING] Agent {agent_name} did not return a response")
            
        print(f"Agent {agent_name} finished running")
        # Use the first message as the prompt
        prompt = [chat_history[0]]
        initial_input_ids = self.tokenizer.apply_chat_template(
            prompt,
            add_generation_prompt=True,  # Always add generation prompt for multi-turn
            tokenize=True,
        )
        initial_prompt_length = len(initial_input_ids)
        
        # Process response messages (everything after the first message)
        response_messages = chat_history[1:]
        
        response_ids = []
        loss_mask = []

        # TODO(tgriggs): This is a hack because qwen2 (or 3?) loss masking does not work 
        for message in response_messages:
            # Apply chat template and tokenize each message
            msg_encoding = self.tokenizer.apply_chat_template(
                [message],
                add_generation_prompt=False,
                tokenize=True
            )
            
            # Extend response_ids with the tokens
            response_ids.extend(msg_encoding)
            
            # Extend loss_mask: 0s for user, 1s for assistant
            if message["role"] == "user":
                loss_mask.extend([0] * len(msg_encoding))
            else:  # assistant
                loss_mask.extend([1] * len(msg_encoding))
        # Extract prompt ids
        prompt_ids = initial_input_ids
        
        # TODO(tgriggs): Consider removing these? Or move to a utils file?
        # Calculate maximum response tokens allowed
        if hasattr(self, 'max_turns') and self.max_turns > 1:
            max_response_tokens = max_tokens + max_input_length - initial_prompt_length
        else:
            max_response_tokens = max_tokens
        
        # Determine stop reason
        stop_reason = "complete"  # Default for trial completion
        if len(response_ids) > max_response_tokens:
            stop_reason = "length"
        
        # Truncate to maximum allowed length
        response_ids = response_ids[:max_response_tokens]
        loss_mask = loss_mask[:max_response_tokens]
        return response_ids, reward, stop_reason, loss_mask, prompt_ids
    
    # TODO(tgriggs): Move this to a utils file
    def _rollout_metrics(self, responses: List[List[int]], rewards: List[float]):
        num_tokens_arr = np.array([len(response) for response in responses])
        non_zero_rewards_arr = np.array([reward > 0.0 for reward in rewards])
        zero_rewards_arr = np.array([reward == 0.0 for reward in rewards])
        # average tokens for non zero rewards
        avg_tokens_non_zero_rewards = (
            np.mean(num_tokens_arr[non_zero_rewards_arr]) if non_zero_rewards_arr.sum() > 0 else np.zeros(1)
        )
        # average tokens for zero rewards
        avg_tokens_zero_rewards = (
            np.mean(num_tokens_arr[zero_rewards_arr]) if zero_rewards_arr.sum() > 0 else np.zeros(1)
        )

        return {
            "generate/min_num_tokens": np.min(num_tokens_arr).item(),
            "generate/max_num_tokens": np.max(num_tokens_arr).item(),
            "generate/avg_num_tokens": np.mean(num_tokens_arr).item(),
            "generate/std_num_tokens": np.std(num_tokens_arr).item(),
            "generate/avg_tokens_non_zero_rewards": avg_tokens_non_zero_rewards.item(),
            "generate/avg_tokens_zero_rewards": avg_tokens_zero_rewards.item(),
        }