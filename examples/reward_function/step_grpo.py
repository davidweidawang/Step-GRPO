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
"""Step-Aware Relative Reward for Step-GRPO.

Implements Sections 3.3 and 3.4 of "Step-GRPO: Internalizing Dynamic Early
Exit for Efficient Reasoning":

- Semantic Step Quantification (Eq. 3): k_i = 1 + N_trig(o_i), where
  N_trig counts occurrences of transition trigger words in the completion.
- Dynamic Step Baseline (Eq. 4): mu is the mean step count of the CORRECT
  completions within the same rollout group G (grouped by prompt uid).
- Final Reward (Eq. 5):
      R_i = alpha * acc_i * (1 - beta * tanh((k_i - mu) / mu)) + (1 - alpha) * fmt_i

Ablations:
- `use_correct_mean=False` reproduces the "w/ All-Sample Mean" ablation.
- `beta=0.0` (or `enable_step_reward=False`) reproduces "w/o Step Reward".
"""

import math
import re
from collections import defaultdict
from typing import Any, Optional

from mathruler.grader import extract_boxed_content, grade_answer


# Metadata read by AutoRewardManager
REWARD_NAME = "step_grpo"
REWARD_TYPE = "batch"

# Transition trigger words W_trig (Section 3.2, following ConCISE / DEER).
# Keep in sync with verl.workers.rollout.deer_utils.TRIGGER_WORDS.
TRIGGER_WORDS = ("Wait",)


def _trigger_pattern(trigger_words: tuple[str, ...]) -> re.Pattern:
    # Standalone-word match: avoid counting "Waiting", "await", etc.
    alternation = "|".join(re.escape(word) for word in trigger_words)
    return re.compile(rf"(?<![A-Za-z])(?:{alternation})(?![A-Za-z])")


def count_semantic_steps(response: str, trigger_words: tuple[str, ...] = TRIGGER_WORDS) -> int:
    """Semantic step count k_i = 1 + N_trig(o_i) (Eq. 3)."""
    return 1 + len(_trigger_pattern(tuple(trigger_words)).findall(response))


def format_reward(response: str) -> float:
    # Paper Appendix A: final answer must appear in \boxed{}.
    return 1.0 if extract_boxed_content(response) else 0.0


def accuracy_reward(response: str, ground_truth: str) -> float:
    answer = extract_boxed_content(response)
    return 1.0 if grade_answer(answer, ground_truth) else 0.0


def compute_score(
    reward_inputs: list[dict[str, Any]],
    alpha: float = 0.9,
    beta: float = 0.5,
    use_correct_mean: bool = True,
    enable_step_reward: bool = True,
    trigger_words: Optional[list[str]] = None,
) -> list[dict[str, float]]:
    """Compute Step-Aware Relative Rewards for a batch of rollouts.

    Args:
        reward_inputs: one dict per completion with keys `response`,
            `response_length`, `ground_truth` and `uid` (rollout group key).
        alpha: weight balancing accuracy against format (Eq. 5).
        beta: step penalty strength, bounds the efficiency term to (-beta, beta).
        use_correct_mean: compute mu over correct completions only (Eq. 4).
            Set False for the "w/ All-Sample Mean" ablation.
        enable_step_reward: set False for the "w/o Step Reward" ablation
            (plain GRPO reward with alpha/1-alpha weighting).
        trigger_words: override the default trigger word set.
    """
    if not isinstance(reward_inputs, list):
        raise ValueError("Please use `reward_type=batch` for the step_grpo reward function.")

    triggers = tuple(trigger_words) if trigger_words else TRIGGER_WORDS

    # First pass: per-sample accuracy, format and semantic steps.
    records = []
    for i, reward_input in enumerate(reward_inputs):
        response = re.sub(r"\s*(<|>|/)\s*", r"\1", reward_input["response"])
        uid = reward_input.get("uid")
        records.append(
            {
                "uid": uid if uid is not None else i,  # fallback: each sample is its own group
                "accuracy": accuracy_reward(response, reward_input["ground_truth"]),
                "format": format_reward(response),
                "steps": count_semantic_steps(response, triggers),
            }
        )

    # Second pass: dynamic step baseline mu per rollout group (Eq. 4).
    group_steps = defaultdict(list)
    for record in records:
        if use_correct_mean:
            if record["accuracy"] > 0.5:
                group_steps[record["uid"]].append(record["steps"])
        else:
            group_steps[record["uid"]].append(record["steps"])

    group_mu = {uid: sum(steps) / len(steps) for uid, steps in group_steps.items()}

    # Third pass: final reward (Eq. 5).
    scores = []
    for record in records:
        accuracy = record["accuracy"]
        mu = group_mu.get(record["uid"])

        if enable_step_reward and accuracy > 0.5 and mu is not None and mu > 0:
            efficiency = 1.0 - beta * math.tanh((record["steps"] - mu) / mu)
        else:
            # No correct completion in the group (mu undefined), incorrect
            # sample, or step reward disabled: omit the efficiency term.
            efficiency = 1.0

        overall = alpha * accuracy * efficiency + (1.0 - alpha) * record["format"]
        scores.append(
            {
                "overall": overall,
                "accuracy": accuracy,
                "format": record["format"],
                "steps": float(record["steps"]),
                "step_mean": float(mu) if mu is not None else 0.0,
            }
        )

    return scores
