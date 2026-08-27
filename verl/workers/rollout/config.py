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
"""
Rollout config
"""

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RolloutConfig:
    name: str = "vllm"
    n: int = 1
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    seed: int = 1
    limit_images: int = 0
    dtype: str = "bf16"
    gpu_memory_utilization: float = 0.6
    ignore_eos: bool = False
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False  # only for v0 engine
    tensor_parallel_size: int = 2
    max_model_len: Optional[int] = None
    max_num_batched_tokens: int = 32768
    disable_log_stats: bool = True
    disable_tqdm: bool = False
    val_override_config: dict[str, Any] = field(default_factory=dict)
    # Dynamic Truncated Rollout (Step-GRPO, Section 3.2). Only active in
    # training rollouts (n > 1); validation always uses plain generation.
    deer_enable: bool = False
    deer_threshold: float = 0.95  # confidence threshold delta
    deer_think_ratio: float = 0.9  # thinking budget = ratio * response_length
    deer_max_judge_steps: int = 10  # max answer-induction attempts per sample
    deer_prob_check_max_tokens: int = 20  # max tokens of the tentative answer
    # Answer-inducing prompt p_ind (DEER reference implementation). The probe
    # stays INSIDE the think block and stops at "</think>"; the tentative
    # answer is only used to compute the confidence and is always discarded.
    deer_answer_prompt: str = "\n**Final Answer**\n\\boxed"
    # Thinking transition point (DEER's `continue_str`). DEER uses a single
    # trigger word ("Wait", or "Alternatively" via its `points` flag).
    deer_trigger_words: list[str] = field(default_factory=lambda: ["Wait"])
    # DEER's repetition early exit (seq_rep_n): if the thinking chunk repeats
    # the previous one 3 times in a row, force the answer phase.
    deer_rep_enable: bool = True
    # DEER's Qwen3 over-confidence fix: only accept the confidence when the
    # tentative answer itself generated "</think>" (see deer_utils.answer_confidence).
    deer_qwen3_strict: bool = True
    # below are auto keys
    prompt_length: int = field(default=-1, init=False)
    response_length: int = field(default=-1, init=False)
    trust_remote_code: bool = field(default=False, init=False)

    def to_dict(self):
        return asdict(self)
