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
"""Pure decision logic for the Dynamic Truncated Rollout (Step-GRPO, Section 3.2).

This module is engine-agnostic (no vLLM / torch dependency) so that the
early-exit logic can be unit-tested in isolation. It follows DEER
(arXiv:2504.15895) for the mechanism and the Step-GRPO paper for the
confidence definition (Eq. 2).

Keep TRIGGER_WORDS in sync with examples/reward_function/step_grpo.py.
"""

import math
import re
from typing import Any, Optional

# Transition trigger words W_trig marking semantic step boundaries.
# DEER uses a single trigger word ("Wait"); the semantic step count (Eq. 3)
# uses the same word so rollout truncation and reward stay consistent.
TRIGGER_WORDS = ("Wait",)

THINK_END = "</think>"


def trigger_pattern(trigger_words: tuple[str, ...] = TRIGGER_WORDS) -> re.Pattern:
    """Standalone-word pattern: matches "Wait" but not "Waiting" or "await"."""
    alternation = "|".join(re.escape(word) for word in trigger_words)
    return re.compile(rf"(?<![A-Za-z])(?:{alternation})(?![A-Za-z])")


def count_semantic_steps(text: str, trigger_words: tuple[str, ...] = TRIGGER_WORDS) -> int:
    """Semantic step count k = 1 + N_trig (Eq. 3)."""
    return 1 + len(trigger_pattern(tuple(trigger_words)).findall(text))


def answer_confidence(
    logprobs: Optional[list[Optional[dict[int, Any]]]],
    qwen3_strict: bool = True,
) -> float:
    """Confidence of a tentative answer (Eq. 2), following DEER's reference
    implementation (`calculate_average_max_prob_from_logprobs`, policy=avg2):

    - skip the first generated token (it merely opens the "\\boxed{" scope and
      is near-deterministic, which would inflate the confidence),
    - at every position use the top-1 probability,
    - aggregate with the geometric mean, mapped to [0, 1] so it is comparable
      with the threshold delta.

    With `qwen3_strict` (DEER's Qwen3 over-confidence fix), the confidence is
    only accepted when the tentative answer itself reached `</think>`, i.e. the
    model closed the think block right after the induced answer. Otherwise the
    answer was cut off mid-way and the confidence is meaningless: return 0.0.

    Args:
        logprobs: per-position dict of {token_id: Logprob} from vLLM
            (SamplingParams(logprobs=1, detokenize=True)); each Logprob has
            `.logprob` and `.decoded_token` fields.
    """
    if not logprobs:
        return 0.0

    log_prob_sum, count = 0.0, 0
    for entry in logprobs[1:]:  # skip the first generated token
        if not entry:
            continue

        logprob_obj = next(iter(entry.values()))  # top-1
        log_prob_sum += math.log(max(math.exp(float(logprob_obj.logprob)), 1e-10))
        count += 1

    if count == 0:
        return 0.0

    if qwen3_strict:
        last_entry = logprobs[-1] or {}
        if not any(getattr(obj, "decoded_token", None) == THINK_END for obj in last_entry.values()):
            return 0.0

    return math.exp(log_prob_sum / count)


def should_early_exit(confidence: float, threshold: float) -> bool:
    """Truncation decision (Section 3.2, step 4): exit iff c(ans) > delta."""
    return confidence > threshold


def is_repetition(prev_chunk: str, chunk: str, n: int = 1) -> bool:
    """One round of DEER's repetition check (`seq_rep_n`).

    Two consecutive thinking chunks count as a repetition when their word-level
    n-gram sets are identical (mutual full overlap). The caller accumulates a
    counter and forces the answer phase once it reaches 3.
    """

    def ngram_set(text: str) -> tuple[set, int]:
        words = text.split(" ")
        grams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
        return set(grams), len(grams)

    prev_set, prev_len = ngram_set(prev_chunk)
    cur_set, cur_len = ngram_set(chunk)
    overlap = len(prev_set & cur_set)
    return overlap == prev_len and overlap == cur_len


def close_think_block(text: str) -> str:
    """Ensure the thinking history ends with a well-formed `</think>` closure."""
    if THINK_END in text:
        stripped = text.rstrip()
        if stripped.endswith(THINK_END):
            return stripped + "\n\n"

        return text if text.endswith("\n") else text + "\n"

    prefix = "" if (not text or text.endswith("\n")) else "\n"
    return text + prefix + THINK_END + "\n\n"
