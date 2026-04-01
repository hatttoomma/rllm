"""
TTRL (Test-Time Reinforcement Learning) reward implementation.

In TTRL, the label is obtained through majority vote from multiple rollouts
instead of using groundtruth. This allows the model to learn from its own
consistent predictions during test time.

Key difference from exact_match/llm_judge:
- exact_match/llm_judge: reward is computed against groundtruth labels
- ttrl/majority_vote: reward is computed against majority-voted predictions
"""

import logging
from collections import Counter, defaultdict
from collections.abc import Sequence

from rllm.agents.agent import Action, Episode
from rllm.engine.rollout.rollout_engine import ModelOutput


logger = logging.getLogger(__name__)


def _normalize_prediction(prediction: str | None) -> str:
    return " ".join((prediction or "").strip().lower().split())


def _base_task_id(episode_id: str | None) -> str:
    if not episode_id:
        return ""
    return episode_id.rsplit(":", 1)[0] if ":" in episode_id else episode_id


def _extract_prediction(episode: Episode) -> str:
    for trajectory in episode.trajectories:
        if not trajectory.steps:
            continue

        step = trajectory.steps[-1]
        assert step.model_response is not None
        return str(step.model_response)

        action = step.action
        if isinstance(action, Action):
            action = action.action
        if action is not None:
            return str(action)

        model_output = step.model_output
        if isinstance(model_output, ModelOutput):
            return str(model_output.content or model_output.text or "")



def _select_majority_prediction(predictions: Sequence[str]) -> str:
    if not predictions:
        return ""

    counts = Counter(predictions)
    first_seen_idx: dict[str, int] = {}
    for idx, prediction in enumerate(predictions):
        first_seen_idx.setdefault(prediction, idx)

    return max(counts, key=lambda prediction: (counts[prediction], -first_seen_idx[prediction]))


def _assign_episode_reward(episode: Episode, reward: float) -> None:
    for trajectory in episode.trajectories:
        trajectory.reward = reward
        trajectory.info["majority_vote_reward"] = reward


def apply_majority_vote_reward(
    episodes: list[Episode],
    *,
    overwrite_episode_correctness: bool = False,
) -> list[Episode]:
    grouped_episodes: dict[str, list[Episode]] = defaultdict(list)
    for episode in episodes:
        if episode is None:
            continue
        grouped_episodes[_base_task_id(episode.id)].append(episode)

    for task_id, group in grouped_episodes.items():
        if len(group) <= 1:
            continue

        raw_predictions = [_extract_prediction(episode) for episode in group]
        normalized_predictions = [_normalize_prediction(prediction) for prediction in raw_predictions]
        majority_prediction = _select_majority_prediction(normalized_predictions)
        majority_count = sum(prediction == majority_prediction for prediction in normalized_predictions)

        logger.info(
            "Apply majority-vote reward for task_id=%s group_size=%s majority_count=%s majority_prediction=%r",
            task_id,
            len(group),
            majority_count,
            majority_prediction,
        )

        for episode, raw_prediction, normalized_prediction in zip(group, raw_predictions, normalized_predictions, strict=False):
            reward = 1.0 if normalized_prediction == majority_prediction else 0.0
            _assign_episode_reward(episode, reward)

            episode.info.setdefault("ttrl", {})
            episode.info["ttrl"].update(
                {
                    "prediction": raw_prediction,
                    "normalized_prediction": normalized_prediction,
                    "majority_prediction": majority_prediction,
                    "majority_count": majority_count,
                    "group_size": len(group),
                }
            )

            episode.metrics["accuracy"] = reward
            episode.metrics["majority_vote_reward"] = reward
            episode.metrics["majority_vote_group_size"] = len(group)
            episode.metrics["majority_vote_count"] = majority_count

            if overwrite_episode_correctness:
                episode.is_correct = bool(reward)

    return episodes
