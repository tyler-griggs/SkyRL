import ray
from packaging import version
from ray.actor import ActorHandle
from typing import Any, List, Dict
from ray.util.placement_group import PlacementGroupSchedulingStrategy, placement_group

from skyrl_train.inference_engines.base import (
    InferenceEngineInterface,
    InferenceEngineInput,
    InferenceEngineOutput,
    NamedWeightsUpdateRequest,
)


class RayWrappedInferenceEngine(InferenceEngineInterface):
    """
    A thin wrapper around a Ray ActorHandle to another InferenceEngineInterface.
    This class implements the InferenceEngineInterface by delegating calls to the remote actor.
    """

    def __init__(self, inference_engine_actor: ActorHandle):
        self.inference_engine_actor = inference_engine_actor

    def tp_size(self):
        return ray.get(self.inference_engine_actor.tp_size.remote())

    def dp_size(self):
        return ray.get(self.inference_engine_actor.dp_size.remote())

    async def generate(self, input_batch: InferenceEngineInput) -> InferenceEngineOutput:
        return await self.inference_engine_actor.generate.remote(input_batch=input_batch)

    async def wake_up(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.wake_up.remote(*args, **kwargs)

    async def sleep(self, *args: Any, **kwargs: Any):
        return await self.inference_engine_actor.sleep.remote(*args, **kwargs)

    async def init_weight_update_communicator(
        self, master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing: bool = False
    ):
        return await self.inference_engine_actor.init_weight_update_communicator.remote(
            master_addr, master_port, rank_offset, world_size, group_name, backend, override_existing
        )

    async def update_named_weights(self, request: NamedWeightsUpdateRequest):
        return await self.inference_engine_actor.update_named_weights.remote(request)

    async def teardown(self):
        return await self.inference_engine_actor.teardown.remote()

    async def reset_prefix_cache(self):
        return await self.inference_engine_actor.reset_prefix_cache.remote()

    async def chat_completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.chat_completion.remote(request_payload)

    async def completion(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        return await self.inference_engine_actor.completion.remote(request_payload)


def create_ray_wrapped_inference_engines(
    num_inference_engines: int,
    tensor_parallel_size: int,
    model_dtype: str,
    pretrain: str,
    seed: int,
    vllm_v1_disable_multiproc: bool,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    expert_parallel_size: int = 1,
    data_parallel_size: int = 1,
    shared_pg=None,
    gpu_memory_utilization=None,
    inference_engine_enable_sleep=False,
    async_engine=False,
    max_num_batched_tokens=8192,
    max_num_seqs=1024,
    tokenizer=None,
    backend="vllm",
    engine_init_kwargs: Dict[str, Any] = {},
) -> List[InferenceEngineInterface]:
    """
    Create a list of RayWrappedInferenceEngine instances wrapping Ray actor handles to InferenceEngineInterface instances.
    """
    from skyrl_train.utils import ray_noset_visible_devices, get_all_env_variables, get_ray_pg_ready_with_timeout
    from skyrl_train.utils.constants import SKYRL_RAY_PG_TIMEOUT_IN_S

    # assert data_parallel_size == 1, "Data parallel size > 1 is not yet supported"

    if backend == "vllm":
        import vllm
        from skyrl_train.inference_engines.vllm.vllm_engine import VLLMRayActor, AsyncVLLMRayActor

        # if a dev version is being used, skip the version check
        if "dev" not in vllm.__version__:
            assert version.parse(vllm.__version__) >= version.parse("0.8.3"), "SkyRL-Train only supports vLLM >= 0.8.3"
    elif backend == "sglang":
        # We import SGLang later to avoid importing vllm. See `get_sglang_engine` for more.
        pass
    else:
        raise ValueError(f"Unsupported backend: {backend}")

    inference_engine_actors = []
    noset_visible_devices = ray_noset_visible_devices(ray.get(get_all_env_variables.remote()))
    # NOTE: we use the ray backend for tensor parallel size > 1 to explicitly manage resource allocation
    # TODO: we should be able to support mp backend by allocating resources at engine level
    distributed_executor_backend = "uni" if (tensor_parallel_size == 1 and data_parallel_size == 1) else "ray"
    # Use mp for data parallelism so vLLM does not create placement groups on its own.
    data_parallel_backend = "mp"
    use_hybrid_engine = shared_pg is not None
    
    # NOTE: currently unused in vllm with hardcoded values
    num_gpus_per_actor = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:
        # Every worker will use 0.2 GPU, so that we can schedule
        # inference and training workers on the same GPUs.
        num_gpus_per_actor = 0.2

    per_engine_gpu_count = tensor_parallel_size * data_parallel_size
    if backend == "vllm":
        total_bundle_count = num_inference_engines * max(1, data_parallel_size) * max(1, tensor_parallel_size)
    else:
        total_bundle_count = num_inference_engines * max(1, per_engine_gpu_count)

    if not use_hybrid_engine:
        # Create a big placement group to ensure that all inference engines are packed
        bundles = [{"GPU": 1, "CPU": 1} for _ in range(total_bundle_count)]
        shared_pg = placement_group(bundles, strategy="PACK")
        get_ray_pg_ready_with_timeout(shared_pg, timeout=SKYRL_RAY_PG_TIMEOUT_IN_S)
    else:
        print(f"We are using the hybrid engine: {use_hybrid_engine}")

    if backend == "vllm":
        actor_cls = AsyncVLLMRayActor if async_engine else VLLMRayActor
        dp_world_size = max(1, data_parallel_size)
        per_actor_tp_count = max(1, tensor_parallel_size)
        bundle_offset = 0
        vllm_actor_groups: List[List[ActorHandle]] = []

        for engine_idx in range(num_inference_engines):
            group: List[ActorHandle] = []
            for dp_rank in range(dp_world_size):
                bundle_indices = list(range(bundle_offset, bundle_offset + per_actor_tp_count))
                bundle_offset += per_actor_tp_count

                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=shared_pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=bundle_indices[0],
                )

                engine = actor_cls.options(
                    num_cpus=1.0,
                    num_gpus=0.0,
                    scheduling_strategy=scheduling_strategy,
                ).remote(
                    model=pretrain,
                    enforce_eager=enforce_eager,
                    worker_extension_cls="skyrl_train.inference_engines.vllm.vllm_engine.WorkerWrap",
                    tensor_parallel_size=tensor_parallel_size,
                    enable_expert_parallel=expert_parallel_size > 1,
                    data_parallel_size=dp_world_size,
                    seed=seed + engine_idx * dp_world_size + dp_rank,
                    distributed_executor_backend=distributed_executor_backend,
                    data_parallel_backend=data_parallel_backend,
                    max_model_len=max_model_len,
                    enable_prefix_caching=enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    vllm_v1_disable_multiproc=vllm_v1_disable_multiproc,
                    gpu_memory_utilization=gpu_memory_utilization,
                    bundle_indices=bundle_indices,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    enable_sleep_mode=inference_engine_enable_sleep,
                    noset_visible_devices=noset_visible_devices,
                    max_num_batched_tokens=max_num_batched_tokens,
                    max_num_seqs=max_num_seqs,
                    # only need the logprobs for the chosen token if any
                    max_logprobs=1,
                    lazy_init=True,
                    **engine_init_kwargs,
                )

                inference_engine_actors.append(engine)
                group.append(engine)

            vllm_actor_groups.append(group)

        init_refs = []
        for group in vllm_actor_groups:
            if not group:
                continue
            if dp_world_size > 1:
                master_ip = ray.get(group[0].get_node_ip.remote())
                master_port = ray.get(group[0].get_free_port.remote())
                for rank, engine in enumerate(group):
                    init_refs.append(
                        engine.init_engine.remote(
                            data_parallel_rank=rank,
                            data_parallel_size=dp_world_size,
                            data_parallel_address=master_ip,
                            data_parallel_rpc_port=master_port,
                        )
                    )
            else:
                for engine in group:
                    init_refs.append(engine.init_engine.remote())

        if init_refs:
            ray.get(init_refs)

    elif backend == "sglang":
        for i in range(num_inference_engines):
            bundle_indices = list(range(i * per_engine_gpu_count, (i + 1) * per_engine_gpu_count))

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=shared_pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=i * per_engine_gpu_count,
            )

            # NOTE: there is no async / sync engine distinction in SGLang

            # NOTE(Charlie): We need `torch.cuda.is_available()` to be True to import SGLang. Otherwise, it requires
            # importing vllm. See https://github.com/sgl-project/sglang/blob/v0.4.8.post1/python/sglang/srt/layers/quantization/utils.py#L11-L17
            # Similar comment: https://github.com/volcengine/verl/blob/9cc307767b0c787e8f5ef581dac929f7bde044ef/verl/workers/fsdp_workers.py#L520-L527
            @ray.remote
            def get_sglang_engine():
                # A workaround to avoid importing vllm is to give this task a GPU.
                import os

                before_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
                os.environ["CUDA_VISIBLE_DEVICES"] = "0"
                from skyrl_train.inference_engines.sglang.sglang_engine import SGLangRayActor

                os.environ["CUDA_VISIBLE_DEVICES"] = before_cuda_visible_devices

                actor_class = SGLangRayActor
                engine = actor_class.options(
                    num_cpus=num_gpus_per_actor,
                    num_gpus=num_gpus_per_actor,
                    scheduling_strategy=scheduling_strategy,
                ).remote(
                    model_path=pretrain,
                    tp_size=tensor_parallel_size,
                    mem_fraction_static=gpu_memory_utilization,
                    random_seed=seed + i,
                    context_length=max_model_len,
                    disable_radix_cache=not enable_prefix_caching,
                    dtype=model_dtype,
                    trust_remote_code=True,
                    max_prefill_tokens=max_num_batched_tokens,
                    max_running_requests=max_num_seqs,
                    # Borrowed from veRL's SGLang rollout
                    mm_attention_backend="fa3",
                    attention_backend="fa3",
                    enable_memory_saver=inference_engine_enable_sleep,
                    # Will be popped before instantiating sgl.Engine
                    distributed_executor_backend=distributed_executor_backend,
                    noset_visible_devices=noset_visible_devices,
                    bundle_indices=bundle_indices,
                    num_gpus=0.2 if use_hybrid_engine else 1,
                    tokenizer=tokenizer,
                    **engine_init_kwargs,
                )
                return engine

            engine = ray.get(get_sglang_engine.remote())

            inference_engine_actors.append(engine)

    engines = [RayWrappedInferenceEngine(actor_handle) for actor_handle in inference_engine_actors]

    if inference_engine_enable_sleep:
        sleep_refs = [engine.inference_engine_actor.sleep.remote() for engine in engines]
        ray.get(sleep_refs)

    return engines
