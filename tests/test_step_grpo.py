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
"""Unit tests for the Step-GRPO reward function and DEER rollout utilities."""

import importlib.util
import math
import os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(TEST_DIR)


def _load_module(name: str, relative_path: str):
    # Load by file path to avoid importing the heavy verl package (torch/vllm).
    path = os.path.join(REPO_ROOT, relative_path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


deer_utils = _load_module("deer_utils", "verl/workers/rollout/deer_utils.py")
step_grpo = _load_module("step_grpo_reward", "examples/reward_function/step_grpo.py")


def _make_input(response: str, ground_truth: str, uid: str) -> dict:
    return {"response": response, "response_length": len(response), "ground_truth": ground_truth, "uid": uid}


def _correct_response(num_triggers: int) -> str:
    body = " some reasoning. ".join(["Wait,"] * num_triggers) if num_triggers else "reasoning"
    return f"{body} The answer is \\boxed{{42}}."


def test_count_semantic_steps():
    # k = 1 + N_trig (Eq. 3), standalone words only
    assert step_grpo.count_semantic_steps("no triggers here") == 1
    # "Alternatively" is no longer a trigger word (DEER uses "Wait" only)
    assert step_grpo.count_semantic_steps("Wait, hmm. Wait again. Alternatively, try") == 3
    assert step_grpo.count_semantic_steps("Waiting is not a trigger, nor is await") == 1
    assert deer_utils.count_semantic_steps("Wait Alternatively") == 2
    print("ok: count_semantic_steps")


def test_group_relative_reward():
    # Group A: two correct samples with different step counts.
    # Group B: one correct sample (mu = own k -> zero efficiency term).
    inputs = [
        _make_input(_correct_response(1), "42", "A"),  # k=2, below group mean
        _make_input(_correct_response(5), "42", "A"),  # k=6, above group mean
        _make_input(_correct_response(3), "42", "B"),  # k=4, mu=4 -> tanh(0)=0
        _make_input("Wait, no idea. The answer is \\boxed{7}.", "42", "B"),  # incorrect
    ]
    scores = step_grpo.compute_score(inputs, alpha=0.9, beta=0.5)

    mu_a = (2 + 6) / 2  # correct-only mean of group A
    expected_0 = 0.9 * (1 - 0.5 * math.tanh((2 - mu_a) / mu_a)) + 0.1 * 1.0
    expected_1 = 0.9 * (1 - 0.5 * math.tanh((6 - mu_a) / mu_a)) + 0.1 * 1.0
    assert abs(scores[0]["overall"] - expected_0) < 1e-9, scores[0]
    assert abs(scores[1]["overall"] - expected_1) < 1e-9, scores[1]
    assert scores[0]["overall"] > scores[1]["overall"]  # fewer steps -> bonus

    # Correct sample at group mean gets the plain accuracy reward.
    assert abs(scores[2]["overall"] - (0.9 + 0.1)) < 1e-9, scores[2]

    # Incorrect sample: no accuracy term, format only.
    assert abs(scores[3]["overall"] - 0.1) < 1e-9, scores[3]
    print("ok: group_relative_reward")


def test_group_isolation():
    # mu must be computed per group, not over the whole batch: the same
    # completion must receive the same reward regardless of other groups.
    base = [_make_input(_correct_response(2), "42", "A"), _make_input(_correct_response(4), "42", "A")]
    other = [_make_input(_correct_response(30), "42", "B") for _ in range(3)]
    scores_alone = step_grpo.compute_score(list(base))
    scores_mixed = step_grpo.compute_score(list(base) + other)
    assert abs(scores_alone[0]["overall"] - scores_mixed[0]["overall"]) < 1e-9
    assert abs(scores_alone[1]["overall"] - scores_mixed[1]["overall"]) < 1e-9
    print("ok: group_isolation")


def test_no_correct_in_group():
    # If the group contains no correct answers, the efficiency term is omitted.
    inputs = [_make_input("Wait. The answer is \\boxed{0}.", "42", "A") for _ in range(3)]
    scores = step_grpo.compute_score(inputs)
    for score in scores:
        assert abs(score["overall"] - 0.1) < 1e-9  # format-only reward
    print("ok: no_correct_in_group")


def test_ablation_flags():
    inputs = [
        _make_input(_correct_response(1), "42", "A"),
        _make_input(_correct_response(5), "42", "A"),
        _make_input("Wait, Wait, Wait, Wait, Wait, Wait, Wait. The answer is \\boxed{0}.", "42", "A"),  # k=8, wrong
    ]
    # w/o Step Reward: plain accuracy + format weighting
    scores = step_grpo.compute_score(inputs, enable_step_reward=False)
    assert abs(scores[0]["overall"] - 1.0) < 1e-9
    assert abs(scores[1]["overall"] - 1.0) < 1e-9

    # w/ All-Sample Mean: incorrect long sample inflates mu
    scores_correct = step_grpo.compute_score(inputs, use_correct_mean=True)
    scores_all = step_grpo.compute_score(inputs, use_correct_mean=False)
    assert scores_all[0]["step_mean"] > scores_correct[0]["step_mean"]
    print("ok: ablation_flags")


class _FakeLogprob:
    def __init__(self, logprob: float, decoded_token: str = ""):
        self.logprob = logprob
        self.decoded_token = decoded_token


def test_answer_confidence():
    # DEER avg2 policy: skip the first token, geometric mean of top-1 probs.
    # The last token must be </think> (qwen3_strict) for the value to count.
    probs = [0.9, 0.8, 0.95]
    logprobs = [
        {1: _FakeLogprob(math.log(probs[0]), "{")},
        {2: _FakeLogprob(math.log(probs[1]), "42")},
        {3: _FakeLogprob(math.log(probs[2]), "</think>")},
    ]
    expected = math.exp(sum(math.log(p) for p in probs[1:]) / 2)  # first token skipped
    assert abs(deer_utils.answer_confidence(logprobs, qwen3_strict=True) - expected) < 1e-9
    assert abs(deer_utils.answer_confidence(logprobs, qwen3_strict=False) - expected) < 1e-9

    # qwen3_strict: tentative answer that never reached </think> scores 0.0
    logprobs_cut = [
        {1: _FakeLogprob(math.log(0.99), "{")},
        {2: _FakeLogprob(math.log(0.99), "42")},
        {3: _FakeLogprob(math.log(0.99), "}")},
    ]
    assert deer_utils.answer_confidence(logprobs_cut, qwen3_strict=True) == 0.0
    assert deer_utils.answer_confidence(logprobs_cut, qwen3_strict=False) > 0.95

    assert deer_utils.answer_confidence(None) == 0.0
    assert deer_utils.answer_confidence([]) == 0.0
    # single-token answer: nothing after the skipped first token
    assert deer_utils.answer_confidence([{1: _FakeLogprob(math.log(0.5), "</think>")}]) == 0.0
    print("ok: answer_confidence")


def test_should_early_exit():
    assert deer_utils.should_early_exit(0.96, 0.95)
    assert not deer_utils.should_early_exit(0.95, 0.95)  # strict inequality
    assert not deer_utils.should_early_exit(0.5, 0.95)
    print("ok: should_early_exit")


def test_is_repetition():
    # identical chunks (word-level 1-gram sets match in both directions)
    assert deer_utils.is_repetition("let me check again", "let me check again")
    # same words, different order: 1-gram sets still identical
    assert deer_utils.is_repetition("check me let again", "let me check again")
    # different content
    assert not deer_utils.is_repetition("let me check again", "try another approach")
    # subset is not a repetition (mutual full overlap required)
    assert not deer_utils.is_repetition("let me check", "let me check again")
    print("ok: is_repetition")


def test_close_think_block():
    assert deer_utils.close_think_block("abc") == "abc\n</think>\n\n"
    assert deer_utils.close_think_block("abc\n") == "abc\n</think>\n\n"
    assert deer_utils.close_think_block("abc</think>") == "abc</think>\n\n"
    assert deer_utils.close_think_block("abc</think>\n") == "abc</think>\n\n"
    print("ok: close_think_block")


if __name__ == "__main__":
    test_count_semantic_steps()
    test_group_relative_reward()
    test_group_isolation()
    test_no_correct_in_group()
    test_ablation_flags()
    test_answer_confidence()
    test_should_early_exit()
    test_is_repetition()
    test_close_think_block()
    print("\nAll Step-GRPO unit tests passed.")
