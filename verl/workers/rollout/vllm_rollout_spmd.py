# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import zlib
from contextlib import contextmanager
from datetime import timedelta
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from transformers import PreTrainedTokenizer, ProcessorMixin
from vllm import LLM, RequestOutput, SamplingParams
from vllm.lora.request import LoRARequest

from ...protocol import DataProto
from ...utils import torch_functional as VF
from ...utils.dataset import process_image, process_video
from ...utils.torch_dtypes import PrecisionType
from ...utils.vllm_utils import VLLMHijack
from . import deer_utils
from .base import BaseRollout
from .config import RolloutConfig


def _repeat_interleave(value: Union[torch.Tensor, np.ndarray], repeats: int) -> Union[torch.Tensor, np.ndarray]:
    # repeat the elements, supports both tensor and numpy array
    if isinstance(value, torch.Tensor):
        return value.repeat_interleave(repeats, dim=0)
    else:
        return np.repeat(value, repeats, axis=0)


def _get_logit_bias(processor: Optional[ProcessorMixin]) -> Optional[dict[int, float]]:
    # enforce vllm to not output image token
    # TODO: add video token
    if processor is not None and hasattr(processor, "image_token"):
        image_token_id = processor.tokenizer.convert_tokens_to_ids(processor.image_token)
        return {image_token_id: -100}
    else:
        return None


def _process_multi_modal_data(
    multi_modal_data: dict[str, Any],
    min_pixels: int,
    max_pixels: int,
    video_fps: float,
    return_video_metadata: bool = False,
) -> dict[str, Any]:
    # may convert image path to image object
    images, videos = [], []
    if "images" in multi_modal_data:
        for image in multi_modal_data["images"]:
            images.append(process_image(image, min_pixels, max_pixels))

    if "videos" in multi_modal_data:
        for video in multi_modal_data["videos"]:
            videos.append(
                process_video(
                    video,
                    min_pixels,
                    max_pixels,
                    video_fps,
                    return_metadata=return_video_metadata,
                )
            )

    if len(images) != 0:
        return {"image": images}

    if len(videos) != 0:
        return {"video": videos}

    return None


class vLLMRollout(BaseRollout):
    def __init__(
        self,
        model_path: str,
        config: RolloutConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        **kwargs,
    ):
        """A vLLM rollout. It requires the module is supported by the vllm.

        Args:
            module: module here follows huggingface APIs
            config: DictConfig
            tokenizer: the task/model tokenizer
        """
        super().__init__()
        self.rank = int(os.getenv("RANK", "0"))
        self.config = config
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        self.return_video_metadata = processor is not None and "Qwen3VLProcessor" in processor.__class__.__name__
        self.use_tqdm = (self.rank == 0) and (not config.disable_tqdm)
        if config.tensor_parallel_size > torch.distributed.get_world_size():
            raise ValueError("Tensor parallelism size should be less than world size.")

        world_size = torch.distributed.get_world_size()
        if world_size % config.tensor_parallel_size != 0:
            raise ValueError("world_size must be divisible by tensor_parallel_size.")

        # Tell vLLM about the data-parallel replicas (world = dp x tp). With
        # external_launcher, vLLM derives each engine's dp rank from
        # RANK // tp_size (matching verl's rank layout) and coordinates the
        # replicas natively: has_unfinished_requests() is all-reduced across
        # the DP group, replicas without local work run dummy batches, and
        # per-step batch sizes are padded/aligned across DP. This engine-step
        # level lockstep is required for the DEER rollout, whose data-dependent
        # multi-round generation makes replicas issue different numbers of
        # engine steps and deadlock otherwise (slow replicas' GPUs spin at 100%
        # inside a collective that early-finished replicas never join).
        self.data_parallel_size = world_size // config.tensor_parallel_size
        engine_kwargs_dp = {}
        if self.data_parallel_size > 1:
            engine_kwargs_dp["data_parallel_size"] = self.data_parallel_size

        # With TP > 1 the two (or more) ranks of a TP group run independent
        # in-process engines that must stay bitwise-deterministic w.r.t. each
        # other; we have observed rare scheduler/CUDA-graph divergences that
        # desynchronize the TP collectives and hang all GPUs at 100% during the
        # multi-round DEER rollout. TP = 1 (one engine per GPU, dp = world)
        # avoids the entire class and is faster for models that fit on one GPU.
        # If the model is too large for a single GPU, prefer enforce_eager=true,
        # which removes CUDA-graph replay from the failure surface (known
        # community mitigation for SPMD rollout hangs).
        if config.deer_enable and config.tensor_parallel_size > 1 and self.rank == 0:
            print(
                "[Step-GRPO rollout] WARNING: deer_enable=true with "
                f"tensor_parallel_size={config.tensor_parallel_size} > 1 can deadlock due to "
                "TP-group engine divergence in vLLM's SPMD (external_launcher) mode. "
                "Recommended: tensor_parallel_size=1 when the model fits on one GPU; "
                "otherwise set worker.rollout.enforce_eager=true."
            )

        if config.max_num_batched_tokens < config.prompt_length + config.response_length:
            raise ValueError("max_num_batched_tokens should be greater than prompt_length + response_length.")

        lora_kwargs = kwargs.pop("lora_kwargs", {})
        self.lora_kwargs = lora_kwargs

        engine_kwargs = {}
        if processor is not None:  # only VLMs have processor
            engine_kwargs["disable_mm_preprocessor_cache"] = True
            if config.limit_images:
                engine_kwargs["limit_mm_per_prompt"] = {"image": config.limit_images}

        VLLMHijack.hijack()

        self.inference_engine = LLM(
            model=model_path,
            skip_tokenizer_init=False,
            trust_remote_code=config.trust_remote_code,
            load_format="dummy" if not self.lora_kwargs else "safetensors",
            dtype=PrecisionType.to_str(PrecisionType.to_dtype(config.dtype)),
            seed=config.seed,
            max_model_len=config.max_model_len or config.prompt_length + config.response_length,
            distributed_executor_backend="external_launcher",
            tensor_parallel_size=config.tensor_parallel_size,
            gpu_memory_utilization=config.gpu_memory_utilization,
            max_num_batched_tokens=config.max_num_batched_tokens,
            disable_log_stats=config.disable_log_stats,
            enforce_eager=config.enforce_eager,
            disable_custom_all_reduce=True,
            enable_chunked_prefill=config.enable_chunked_prefill,
            enable_sleep_mode=True,
            **engine_kwargs_dp,
            **lora_kwargs,
            **engine_kwargs,
        )

        # Offload vllm model to reduce peak memory usage
        self.inference_engine.sleep(level=1)

        sampling_kwargs = {
            "max_tokens": config.response_length,
            "detokenize": False,
            "logit_bias": _get_logit_bias(processor),
        }
        default_sampling_params = SamplingParams()
        for key in config.to_dict().keys():
            if hasattr(default_sampling_params, key):
                sampling_kwargs[key] = getattr(config, key)

        print(f"Sampling params: {sampling_kwargs}.")
        self.sampling_params = SamplingParams(**sampling_kwargs)

        # Dedicated CPU (gloo) group for the DEER cross-replica control-flow
        # all-reduce, created lazily on first use (see _get_deer_sync_group).
        self._deer_sync_group = None

    @contextmanager
    def update_sampling_params(self, **kwargs):
        # update sampling params
        old_sampling_params_args = {}
        if kwargs:
            for key, value in kwargs.items():
                if hasattr(self.sampling_params, key):
                    old_value = getattr(self.sampling_params, key)
                    old_sampling_params_args[key] = old_value
                    setattr(self.sampling_params, key, value)

        yield
        # roll back to previous sampling params
        for key, value in old_sampling_params_args.items():
            setattr(self.sampling_params, key, value)

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # Dynamic Truncated Rollout (Step-GRPO): only for training rollouts
        # (n > 1). Validation (n == 1) always uses plain generation so that
        # evaluation reflects the model's internalized behavior.
        desired_n = int(prompts.meta_info.get("n", self.sampling_params.n))
        is_multi_modal = prompts.non_tensor_batch.get("multi_modal_data") is not None
        if self.config.deer_enable and desired_n > 1 and not is_multi_modal and not self.lora_kwargs:
            return self._generate_sequences_deer(prompts)

        # left-padded attention_mask
        input_ids: torch.Tensor = prompts.batch["input_ids"]  # (bs, prompt_length)
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]
        batch_size = input_ids.size(0)

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        batch_multi_modal_data = non_tensor_batch.pop("multi_modal_data", None)
        if batch_size != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        if batch_multi_modal_data is not None:
            vllm_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(batch_raw_prompt_ids, batch_multi_modal_data):
                vllm_inputs.append(
                    {
                        "prompt_token_ids": list(raw_prompt_ids),
                        "multi_modal_data": _process_multi_modal_data(
                            multi_modal_data,
                            prompts.meta_info["min_pixels"],
                            prompts.meta_info["max_pixels"],
                            prompts.meta_info["video_fps"],
                            return_video_metadata=self.return_video_metadata,
                        ),
                    }
                )
        else:
            vllm_inputs = [{"prompt_token_ids": list(raw_prompt_ids)} for raw_prompt_ids in batch_raw_prompt_ids]

        lora_requests = None
        if self.lora_kwargs:
            lora_int_ids = list(self.inference_engine.llm_engine.list_loras())
            if len(lora_int_ids) > 0:
                lora_int_id = lora_int_ids[0]
                lora_requests = [
                    LoRARequest(lora_name=f"{lora_int_id}", lora_int_id=lora_int_id, lora_path="/simon-stub-path")
                ] * batch_size

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**prompts.meta_info):
            completions: list[RequestOutput] = self.inference_engine.generate(
                prompts=vllm_inputs,
                sampling_params=self.sampling_params,
                lora_request=lora_requests,
                use_tqdm=self.use_tqdm,
            )
            response_ids = [output.token_ids for completion in completions for output in completion.outputs]
            response_ids = VF.pad_2d_list_to_length(
                response_ids, self.pad_token_id, max_length=self.config.response_length
            ).to(input_ids.device)

            if self.sampling_params.n > 1:
                batch_size = batch_size * self.sampling_params.n
                input_ids = _repeat_interleave(input_ids, self.sampling_params.n)
                attention_mask = _repeat_interleave(attention_mask, self.sampling_params.n)
                position_ids = _repeat_interleave(position_ids, self.sampling_params.n)
                if batch_multi_modal_data is not None:
                    batch_multi_modal_data = _repeat_interleave(batch_multi_modal_data, self.sampling_params.n)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1 | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3 | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        if batch_multi_modal_data is not None:
            non_tensor_batch = {"multi_modal_data": batch_multi_modal_data}
        else:
            non_tensor_batch = {}

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch, meta_info=prompts.meta_info)

    def _get_deer_sync_group(self):
        """Lazily create a world-wide CPU (gloo) group for the DEER control-flow
        all-reduce.

        The tiny "is anyone still active?" flag must be reduced across every
        rank, but it must NOT ride on the default NCCL group: DEER replicas can
        spend very different wall-clock time per round (a 16k-token generation
        vs. a 1-token dummy), so a fast replica may wait many minutes at the
        barrier for a slow one. On the default NCCL group that trips the CUDA
        collective watchdog (default 600s) and kills training. A gloo group with
        a long timeout tolerates the skew and keeps the flag off the GPU
        watchdog entirely. All ranks run this rollout, so new_group (collective
        over the default group) is entered by everyone.
        """
        if self._deer_sync_group is None:
            self._deer_sync_group = torch.distributed.new_group(
                backend="gloo", timeout=timedelta(hours=2)
            )

        return self._deer_sync_group

    @staticmethod
    def _deer_seed(prompt_hash: int, idx: int, iteration: int, phase_code: int) -> int:
        """Deterministic per-request sampling seed.

        The two ranks inside a tensor-parallel group hold identical prompts and
        identical (bitwise) logits, but vLLM's default (seedless) sampling draws
        the per-request generator from process-local state, so the two ranks can
        occasionally sample *different* tokens. In the plain rollout that is
        harmless, but DEER's control flow branches on the generated tokens: a
        single divergence makes the two ranks build different requests next round
        and mismatch the TP collective inside generate(), hanging both GPUs at
        100% util. Seeding every request with a value derived only from
        replica-identical inputs (crc32 is process-independent, unlike Python's
        randomized hash()) forces identical sampling and keeps the ranks in
        lockstep. The seed still varies per sample / round / phase / step (the
        prompt hash differs across steps), so rollout diversity is preserved.
        """
        return (prompt_hash * 1_000_003 + idx * 10_007 + iteration * 101 + phase_code) & 0x7FFFFFFF

    def _append_text_ids(self, ids: list[int], text: str) -> None:
        """Append `text` to a token id list, re-encoding a short tail to keep BPE boundaries intact."""
        if not text:
            return

        overlap = min(8, len(ids))
        if overlap == 0:
            ids.extend(self.tokenizer.encode(text, add_special_tokens=False))
            return

        tail_text = self.tokenizer.decode(ids[-overlap:], skip_special_tokens=False)
        new_tail = self.tokenizer.encode(tail_text + text, add_special_tokens=False)
        ids[len(ids) - overlap :] = new_tail

    @torch.no_grad()
    def _generate_sequences_deer(self, prompts: DataProto) -> DataProto:
        """Dynamic Truncated Rollout (Step-GRPO, Section 3.2), aligned with the
        DEER reference implementation (vllm-deer-qwen3.py).

        Every sample in the rollout group follows the DEER loop:
        1) generate until a transition trigger word (semantic step boundary);
           the trigger is NOT appended yet -- the probe context must end at the
           completed reasoning step, not at a dangling "Wait",
        2) append the answer-inducing prompt (inside the think block, stopping
           at "</think>") and generate a tentative answer,
        3) compute its confidence c(a) (Eq. 2 / DEER avg2 policy),
        4) if c(a) > delta (or the thinking budget is nearly exhausted),
           DISCARD the tentative answer, close the think block with
           "\\n</think>\\n\\n" and generate the final answer with the full
           remaining budget; otherwise append the trigger word back and resume
           thinking.

        Trajectories whose confidence never crosses delta finish naturally,
        so the group mixes truncated and natural completions.
        """
        config = self.config
        input_ids: torch.Tensor = prompts.batch["input_ids"]
        attention_mask: torch.Tensor = prompts.batch["attention_mask"]
        position_ids: torch.Tensor = prompts.batch["position_ids"]
        eos_token_id: int = prompts.meta_info["eos_token_id"]

        non_tensor_batch = prompts.non_tensor_batch
        batch_raw_prompt_ids = non_tensor_batch.pop("raw_prompt_ids")
        non_tensor_batch.pop("multi_modal_data", None)
        if input_ids.size(0) != len(batch_raw_prompt_ids):
            raise RuntimeError("vllm sharding manager is not work properly.")

        # Expand the batch by the group size G (rollout.n): sample i * n + j
        # belongs to prompt i, matching DataProto.repeat(n, interleave=True).
        num_samples = int(prompts.meta_info.get("n", self.sampling_params.n))
        input_ids = _repeat_interleave(input_ids, num_samples)
        attention_mask = _repeat_interleave(attention_mask, num_samples)
        position_ids = _repeat_interleave(position_ids, num_samples)
        batch_raw_prompt_ids = [ids for ids in batch_raw_prompt_ids for _ in range(num_samples)]
        batch_size = input_ids.size(0)

        temperature = float(prompts.meta_info.get("temperature", self.sampling_params.temperature))
        top_p = float(prompts.meta_info.get("top_p", self.sampling_params.top_p))

        response_length = config.response_length
        think_limit = int(response_length * config.deer_think_ratio)
        model_len = (config.max_model_len or (config.prompt_length + config.response_length)) - 64
        triggers = list(config.deer_trigger_words)
        answer_prompt_ids = self.tokenizer.encode(config.deer_answer_prompt, add_special_tokens=False)

        # Note: vLLM stop strings are substring matches, so "Wait" also stops
        # inside e.g. "Waiting". This matches the original DEER behavior; the
        # reward side counts standalone trigger words only.
        think_stops = triggers + [deer_utils.THINK_END]
        free_run_stops = [deer_utils.THINK_END]

        states = []
        for i in range(batch_size):
            states.append(
                {
                    "ids": list(batch_raw_prompt_ids[i]),
                    "prompt_len": len(batch_raw_prompt_ids[i]),
                    "phase": "think",  # think -> probe -> answer -> done
                    "judge_steps": 0,
                    "free_run": False,
                    "think_closed": False,
                    "exit_reason": None,
                    # Trigger word that paused thinking for the current probe;
                    # appended back to the path only if the probe declines.
                    "pending_trigger": None,
                    # DEER repetition early exit (seq_rep_n) bookkeeping.
                    "last_chunk": None,
                    "rep_count": 0,
                    # Replica-identical, process-independent hash of the prompt;
                    # used to derive deterministic per-request sampling seeds so
                    # the two TP ranks never diverge (see _deer_seed).
                    "prompt_hash": zlib.crc32(np.asarray(batch_raw_prompt_ids[i], dtype=np.int64).tobytes()),
                }
            )

        exit_counter = {"early_exit": 0, "natural": 0, "budget": 0, "eos": 0, "rep": 0}
        probe_confidences: list[float] = []  # every Eq. 2 confidence, for diagnostics
        max_iterations = 3 * config.deer_max_judge_steps + 16

        # Per-request seeds exist solely to keep the ranks of a tensor-parallel
        # group in bitwise lockstep (their engines must sample identically or
        # the TP collectives inside generate() diverge and hang). With TP = 1
        # every GPU runs an independent engine, no such constraint exists, and
        # we keep vLLM's default (unseeded) sampling.
        needs_seed = config.tensor_parallel_size > 1

        # vLLM's `external_launcher` backend runs every rank in the torch world
        # in SPMD lockstep: each rank must issue the *same number* of
        # `generate()` collectives. The dynamic loop below terminates per sample,
        # so different data-parallel replicas would otherwise finish after a
        # different number of rounds and deadlock (finished replicas block on the
        # next collective while slower ones keep generating). We therefore drive
        # the loop by a world-wide "is anyone still active?" all-reduce and pad
        # replicas that have finished with a dummy request so the per-iteration
        # generate() count stays identical across the whole world.
        #
        # The flag is reduced on a dedicated long-timeout gloo group (not the
        # default NCCL group) because per-round wall-clock time varies wildly
        # across replicas; otherwise a fast replica waiting at the barrier trips
        # the NCCL collective watchdog (default 600s) and crashes training.
        sync_across_ranks = torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1
        sync_group = self._get_deer_sync_group() if sync_across_ranks else None

        iteration = 0
        while True:
            local_active = any(state["phase"] != "done" for state in states)
            if sync_across_ranks:
                active_flag = torch.tensor([1 if local_active else 0], dtype=torch.int32)
                torch.distributed.all_reduce(active_flag, op=torch.distributed.ReduceOp.MAX, group=sync_group)
                global_active = bool(active_flag.item())
            else:
                global_active = local_active

            if not global_active:
                break

            iteration += 1
            if iteration > max_iterations:
                for state in states:
                    if state["phase"] != "done":
                        state["phase"] = "done"
                        if state["exit_reason"] is None:
                            state["exit_reason"] = "budget"
                            exit_counter["budget"] += 1
                break

            requests, request_params, request_meta = [], [], []
            for idx, state in enumerate(states):
                if state["phase"] == "done":
                    continue

                response_tokens = len(state["ids"]) - state["prompt_len"]
                context_remaining = model_len - len(state["ids"])

                if state["phase"] == "think":
                    max_new = min(think_limit - response_tokens, context_remaining)
                    if max_new <= 0:  # thinking budget exhausted
                        state["phase"] = "answer"
                        if state["exit_reason"] is None:
                            state["exit_reason"] = "budget"
                            exit_counter["budget"] += 1
                    else:
                        requests.append({"prompt_token_ids": list(state["ids"])})
                        request_params.append(
                            SamplingParams(
                                max_tokens=max_new,
                                temperature=temperature,
                                top_p=top_p,
                                stop=free_run_stops if state["free_run"] else think_stops,
                                detokenize=True,
                                seed=self._deer_seed(state["prompt_hash"], idx, iteration, 0) if needs_seed else None,
                            )
                        )
                        request_meta.append((idx, "think"))
                        continue

                if state["phase"] == "probe":
                    probe_len = len(state["ids"]) + len(answer_prompt_ids) + config.deer_prob_check_max_tokens
                    if probe_len > model_len:  # no room for answer induction
                        state["pending_trigger"] = None
                        state["phase"] = "answer"
                        if state["exit_reason"] is None:
                            state["exit_reason"] = "budget"
                            exit_counter["budget"] += 1
                    else:
                        requests.append({"prompt_token_ids": list(state["ids"]) + answer_prompt_ids})
                        request_params.append(
                            SamplingParams(
                                max_tokens=config.deer_prob_check_max_tokens,
                                temperature=0.0,
                                logprobs=1,
                                detokenize=True,
                                # DEER: the probe runs inside the think block and
                                # stops when the model closes it. Reaching this
                                # stop is also what deer_qwen3_strict checks.
                                stop=[deer_utils.THINK_END],
                                seed=self._deer_seed(state["prompt_hash"], idx, iteration, 2) if needs_seed else None,
                            )
                        )
                        request_meta.append((idx, "probe"))
                        continue

                if state["phase"] == "answer":
                    if not state["think_closed"]:
                        self._append_text_ids(state["ids"], "\n" + deer_utils.THINK_END + "\n\n")
                        state["think_closed"] = True
                        response_tokens = len(state["ids"]) - state["prompt_len"]
                        context_remaining = model_len - len(state["ids"])

                    max_new = min(response_length - response_tokens, context_remaining)
                    if max_new <= 0:
                        state["phase"] = "done"
                        if state["exit_reason"] is None:
                            state["exit_reason"] = "budget"
                            exit_counter["budget"] += 1
                    else:
                        requests.append({"prompt_token_ids": list(state["ids"])})
                        request_params.append(
                            SamplingParams(
                                max_tokens=max_new,
                                temperature=temperature,
                                top_p=top_p,
                                detokenize=True,
                                seed=self._deer_seed(state["prompt_hash"], idx, iteration, 1) if needs_seed else None,
                            )
                        )
                        request_meta.append((idx, "answer"))
                        continue

            # Keep every rank in lockstep: even when this replica has no real
            # work this round (all local samples done, or every active sample
            # merely switched phase without emitting a request), it must still
            # issue exactly one generate() so the external_launcher collective
            # matches the busier replicas. The dummy output has no request_meta
            # entry, so it is discarded by the zip below.
            if not requests:
                dummy_prompt = list(states[0]["ids"][:1]) or [self.pad_token_id]
                requests.append({"prompt_token_ids": dummy_prompt})
                request_params.append(SamplingParams(max_tokens=1, temperature=0.0, detokenize=False))

            completions = self.inference_engine.generate(
                prompts=requests, sampling_params=request_params, use_tqdm=False
            )
            for completion, (idx, phase) in zip(completions, request_meta):
                state = states[idx]
                output = completion.outputs[0]

                if phase == "think":
                    state["ids"].extend(output.token_ids)

                    # DEER repetition early exit (seq_rep_n): three consecutive
                    # repeated thinking chunks force the answer phase.
                    repeated = False
                    if config.deer_rep_enable:
                        chunk = output.text or ""
                        if state["last_chunk"] is not None and deer_utils.is_repetition(
                            state["last_chunk"], chunk
                        ):
                            state["rep_count"] += 1
                            repeated = state["rep_count"] >= 3

                    if repeated:
                        state["phase"] = "answer"
                        state["exit_reason"] = "rep"
                        exit_counter["rep"] += 1
                    elif output.finish_reason == "length":
                        # Thinking budget exhausted: forced answer (DEER's too_long)
                        state["phase"] = "answer"
                        state["exit_reason"] = "budget"
                        exit_counter["budget"] += 1
                    elif output.stop_reason in triggers:
                        # Semantic step boundary: pause to evaluate the
                        # necessity of further reasoning. The trigger word is
                        # NOT appended yet (DEER): the probe context must end
                        # at the completed step, and the trigger is restored
                        # only if the probe declines to exit.
                        state["pending_trigger"] = str(output.stop_reason)
                        state["last_chunk"] = output.text or ""
                        state["phase"] = "probe"
                    elif output.stop_reason == deer_utils.THINK_END:
                        # Model closed the think block itself (natural end).
                        self._append_text_ids(state["ids"], deer_utils.THINK_END + "\n\n")
                        state["think_closed"] = True
                        state["phase"] = "answer"
                        state["exit_reason"] = state["exit_reason"] or "natural"
                    else:  # eos: the whole completion ended inside thinking
                        state["phase"] = "done"
                        state["exit_reason"] = "eos"
                        exit_counter["eos"] += 1

                elif phase == "probe":
                    confidence = deer_utils.answer_confidence(output.logprobs, config.deer_qwen3_strict)
                    probe_confidences.append(confidence)
                    # DEER also exits when the thinking budget is nearly
                    # exhausted (within 50 tokens), regardless of confidence.
                    response_tokens = len(state["ids"]) - state["prompt_len"]
                    think_limit_reached = response_tokens >= think_limit - 50
                    confident = deer_utils.should_early_exit(confidence, config.deer_threshold)
                    if confident or think_limit_reached:
                        # The tentative answer is DISCARDED (DEER): close the
                        # think block and regenerate the final answer with the
                        # full remaining budget in the answer phase.
                        if confident:
                            state["exit_reason"] = "early_exit"
                            exit_counter["early_exit"] += 1
                        else:
                            state["exit_reason"] = "budget"
                            exit_counter["budget"] += 1
                        state["pending_trigger"] = None
                        state["phase"] = "answer"
                    else:
                        # Not confident: discard the tentative answer, restore
                        # the trigger word and resume thinking after it.
                        if state["pending_trigger"]:
                            self._append_text_ids(state["ids"], state["pending_trigger"])
                            state["pending_trigger"] = None
                        state["judge_steps"] += 1
                        if state["judge_steps"] >= config.deer_max_judge_steps:
                            state["free_run"] = True
                        state["phase"] = "think"

                elif phase == "answer":
                    state["ids"].extend(output.token_ids)
                    state["phase"] = "done"
                    if state["exit_reason"] is None:
                        state["exit_reason"] = "natural"
                    if state["exit_reason"] == "natural":
                        exit_counter["natural"] += 1

        if self.rank == 0:
            if probe_confidences:
                confs = sorted(probe_confidences)
                n_conf = len(confs)
                conf_stats = (
                    f"probes={n_conf}, conf p10/p50/p90/max="
                    f"{confs[int(0.10 * (n_conf - 1))]:.3f}/{confs[int(0.50 * (n_conf - 1))]:.3f}/"
                    f"{confs[int(0.90 * (n_conf - 1))]:.3f}/{confs[-1]:.3f}, "
                    f"over_delta={sum(c > config.deer_threshold for c in confs)}"
                )
            else:
                conf_stats = "probes=0"
            print(f"[Step-GRPO rollout] batch={batch_size}, exits={exit_counter}, {conf_stats}")

        # Assemble tensors in the same layout as the plain rollout path.
        eos_token_ids = eos_token_id if isinstance(eos_token_id, (list, tuple)) else [eos_token_id]
        response_ids_list = []
        for state in states:
            response_ids = state["ids"][state["prompt_len"] :]
            if len(response_ids) < response_length and (not response_ids or response_ids[-1] not in eos_token_ids):
                response_ids.append(eos_token_ids[0])
            response_ids_list.append(response_ids[:response_length])

        response_ids = VF.pad_2d_list_to_length(
            response_ids_list, self.pad_token_id, max_length=response_length
        ).to(input_ids.device)

        sequence_ids = torch.cat([input_ids, response_ids], dim=-1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.view(1, -1).expand(batch_size, -1)
        if position_ids.ndim == 3:  # qwen2vl mrope: (batch_size, 4, seq_length)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        response_position_ids = position_ids[..., -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_mask = VF.get_response_mask(
            response_ids=response_ids, eos_token_id=eos_token_id, dtype=attention_mask.dtype
        )
        attention_mask = torch.cat((attention_mask, response_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": input_ids,
                "responses": response_ids,
                "input_ids": sequence_ids,
                "attention_mask": attention_mask,
                "response_mask": response_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch, non_tensor_batch={}, meta_info=prompts.meta_info)
