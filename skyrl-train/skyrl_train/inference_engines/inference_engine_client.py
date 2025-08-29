from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)
from transformers import PreTrainedTokenizerBase
import asyncio
from typing import List, Any, Optional, Dict, Hashable


class InferenceEngineClient(InferenceEngineInterface):
    """
    Client to talk to a set of InferenceEngines.

    Note that InferenceEngineClient sub-classes InferenceEngineInterface so it can be used as if talking to a single engine.
    """

    # TODO(tgriggs): Take generator config as input.
    def __init__(self, engines: List[InferenceEngineInterface], tokenizer: PreTrainedTokenizerBase, backend: Optional[str] = None, max_model_len: Optional[int] = None):
        self.engines = engines
        self.tokenizer = tokenizer
        self.backend = backend
        self.max_model_len = max_model_len
        print(f"InferenceEngineClient initialized with {len(engines)} engines.")

    async def _run_on_all_engines(self, method_name: str, *args, **kwargs):
        """
        Call a method on all engines concurrently and gather the results.
        """
        assert len(self.engines) > 0, "No engines to call method on"

        awaitables = [getattr(engine, method_name)(*args, **kwargs) for engine in self.engines]
        return await asyncio.gather(*awaitables)

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        prompts = input_batch.get("prompts")
        prompt_token_ids = input_batch.get("prompt_token_ids")
        trajectory_ids = input_batch.get("trajectory_ids")
        sampling_params = input_batch.get("sampling_params")

        if (prompts is None and prompt_token_ids is None) or (prompts is not None and prompt_token_ids is not None):
            raise ValueError("Either `prompts` or `prompt_token_ids` must be provided, but not both.")

        if prompt_token_ids is None:
            prompt_token_ids = self.tokenizer.apply_chat_template(
                prompts,
                add_generation_prompt=True,
                add_special_tokens=False,
                return_dict=True,
                tokenize=True,
            )["input_ids"]

        # Clamp sampling params based on max context length the model supports and current prompt lengths.
        sampling_params = self._clamp_sampling_params(prompt_token_ids, sampling_params)

        # TODO(tgriggs): If there are no traj ids, we'd still like to load balance instead of landing on a single engine.
        if trajectory_ids is not None:
            # Route based on trajectory_ids
            return await self._generate_with_trajectory_routing(prompt_token_ids, trajectory_ids, sampling_params)
        else:
            # Split evenly across engines
            return await self._generate_batched(prompt_token_ids, sampling_params)

    def _clamp_sampling_params(self, prompt_token_ids: List[List[int]], sampling_params: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Clamp backend-specific max generation length based on max_model_len and prompt lengths.
        
        TODO(tgriggs): Mention assumptions here.
        """
        if sampling_params is None or len(prompt_token_ids) > 1:
            return sampling_params
        
        # Only one prompt, extract it.
        prompt_tokens = prompt_token_ids[0]

        # Compute remaining tokens in context.
        remaining_tokens_in_context = max(0, min(self.max_model_len - len(prompt_tokens)))

        # Copy and clamp the sampling params.
        clamped: Dict[str, Any] = dict(sampling_params)

        if self.backend == "vllm":
            max_tokens = clamped.get("max_tokens")
            effective_max_tokens = remaining_tokens_in_context if max_tokens is None else min(max_tokens, remaining_tokens_in_context)
            clamped["max_tokens"] = max(0, effective_max_tokens)
            
            # Ensure min_tokens does not exceed max_tokens
            if "min_tokens" in clamped and clamped["min_tokens"] is not None:
                if clamped["min_tokens"] > clamped["max_tokens"]:
                    clamped["min_tokens"] = clamped["max_tokens"]
        elif self.backend == "sglang":
            max_new_tokens = clamped.get("max_new_tokens")
            effective_max_new_tokens = remaining_tokens_in_context if max_new_tokens is None else min(max_new_tokens, remaining_tokens_in_context)
            clamped["max_new_tokens"] = max(0, effective_max_new_tokens)
        else:
            return sampling_params

        return clamped

    async def _generate_with_trajectory_routing(
        self, 
        prompt_token_ids : List[List[int]], 
        trajectory_ids : List[Hashable], 
        sampling_params : Optional[Dict[str, Any]]
    ) -> InferenceEngineOutput:
        """
        Route prompts to engines based on trajectory_ids and return results in the original order of the prompts.
        """
        # Group prompts by engine
        engine_groups: dict[int, dict[str, list]] = {}
        for i, (token_ids, traj_id) in enumerate(zip(prompt_token_ids, trajectory_ids)):
            engine_idx = abs(hash(str(traj_id))) % len(self.engines)
            group = engine_groups.setdefault(engine_idx, {"token_ids": [], "indices": []})
            group["token_ids"].append(token_ids)
            group["indices"].append(i)

        # Build two parallel lists: one of tasks, one of the index‐lists
        tasks: list[asyncio.Task] = []
        indices_list: list[list[int]] = []
        for engine_idx, group in engine_groups.items():
            inp = InferenceEngineInput(
                prompt_token_ids=group["token_ids"],
                sampling_params=sampling_params,
            )
            coro = self.engines[engine_idx].generate(inp)
            tasks.append(asyncio.create_task(coro))
            indices_list.append(group["indices"])

        results = await asyncio.gather(*tasks)

        # Reconstruct output in original order
        n = len(prompt_token_ids)
        responses: list[str] = [""] * n
        stop_reasons: list[str] = [""] * n
        response_logprobs: List[Optional[List[float]]] = [None for _ in range(n)]
        response_ids: List[List[int]] = [[] for _ in range(n)]
        # a bit hacky for now
        add_resp_logprobs = False

        for indices, result in zip(indices_list, results):
            for local_idx, original_idx in enumerate(indices):
                responses[original_idx] = result["responses"][local_idx]
                stop_reasons[original_idx] = result["stop_reasons"][local_idx]
                response_ids[original_idx] = result["response_ids"][local_idx]
                if result.get("response_logprobs", None):
                    add_resp_logprobs = True
                    response_logprobs[original_idx] = result["response_logprobs"][local_idx]

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs if add_resp_logprobs else None,
        )

    async def _generate_batched(
        self, 
        prompt_token_ids : List[List[int]], 
        sampling_params : Optional[Dict[str, Any]]
    ) -> InferenceEngineOutput:
        """
        Split prompts evenly across engines and return results in the original order of the prompts.
        """
        num_inference_engines = len(self.engines)
        dp_item_size = (len(prompt_token_ids) + num_inference_engines - 1) // num_inference_engines

        tasks = []
        for dp_rank in range(num_inference_engines):
            start_idx = dp_rank * dp_item_size
            end_idx = (dp_rank + 1) * dp_item_size
            dp_items = prompt_token_ids[start_idx:end_idx]

            if len(dp_items) <= 0:
                continue

            engine_input = InferenceEngineInput(
                prompt_token_ids=dp_items,
                sampling_params=sampling_params,
            )
            tasks.append(self.engines[dp_rank].generate(engine_input))

        all_outputs = await asyncio.gather(*tasks)

        # Flatten results
        responses = []
        stop_reasons = []
        response_ids = []
        response_logprobs = []
        for output in all_outputs:
            responses.extend(output["responses"])
            stop_reasons.extend(output["stop_reasons"])
            response_ids.extend(output["response_ids"])
            if output.get("response_logprobs", None):
                response_logprobs.extend(output["response_logprobs"])

        return InferenceEngineOutput(
            responses=responses,
            stop_reasons=stop_reasons,
            response_ids=response_ids,
            response_logprobs=response_logprobs if len(response_logprobs) else None,
        )

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self._run_on_all_engines("wake_up", *args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self._run_on_all_engines("sleep", *args, **kwargs)

    async def init_weight_update_communicator(
        self,
        master_addr,
        master_port,
        rank_offset,
        world_size,
        group_name,
        backend,
        override_existing: bool = False,
    ):
        tasks = []
        rank_offset_count = rank_offset

        for engine in self.engines:
            assert engine.tp_size is not None, "Engine must have a tp_size"
            tasks.append(
                engine.init_weight_update_communicator(
                    master_addr=master_addr,
                    master_port=master_port,
                    rank_offset=rank_offset_count,
                    world_size=world_size,
                    group_name=group_name,
                    backend=backend,
                    override_existing=override_existing,
                )
            )
            rank_offset_count += engine.tp_size
        await asyncio.gather(*tasks)

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self._run_on_all_engines("update_named_weights", request=request)

    async def reset_prefix_cache(self):
        return await self._run_on_all_engines("reset_prefix_cache")

    async def teardown(self):
        return await self._run_on_all_engines("teardown")
