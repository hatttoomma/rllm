from PIL import Image

from .agent import MMSearchAgent
import base64
import binascii
import logging

from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason, Workflow
from io import BytesIO

from .reward.llm_judge import exact_match_reward_fn, make_llm_judge_reward_fn
from .reward.ttrl import apply_majority_vote_reward
import inspect
import os


logger = logging.getLogger(__name__)


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _string_match(pred: str, labels: list[str]) -> bool:
    p = _normalize(pred)
    return any(p == _normalize(x) for x in labels)

def decode_base64(images_b64: list[str] | None) -> list[Image.Image]:
    decoded_images: list[Image.Image] = []
    for idx, img in enumerate(images_b64 or []):
        if not img:
            continue
        if isinstance(img, bytes):
            img = img.decode("utf-8")
        if not isinstance(img, str):
            raise TypeError(f"Invalid image payload type at index {idx}: {type(img)}")

        # Support both pure base64 and data URLs like data:image/png;base64,....
        payload = img.split(",", 1)[1] if img.startswith("data:") and "," in img else img
        try:
            image_bytes = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise ValueError(f"Invalid base64 image at index {idx}") from exc

        try:
            pil_img = Image.open(BytesIO(image_bytes))
            pil_img.load()
            decoded_images.append(pil_img)
        except Exception as exc:
            raise ValueError(f"Decoded bytes are not a valid image at index {idx}") from exc

    return decoded_images


class MMSearchWorkflow(Workflow):
    """
    Minimal workflow:
    - call agent.run()
    - convert to Episode with a single-step trajectory
    """

    def __init__(
        self,
        rollout_engine,
        executor=None,
        reward_type: str = "exact_match",
        judge_temperature: float = 0.0,
        judge_max_tokens: int = 256,
        **kwargs,
    ):
        super().__init__(rollout_engine, executor, **kwargs)
        self.agent = MMSearchAgent(rollout_engine)
        self.reward_type = reward_type
        logger.info("MMSearchWorkflow initialized with reward_type=%s", reward_type)

        # Set reward function based on reward_type
        if reward_type == "llm_judge" or reward_type == "ttrl":
            try:
                self.reward_fn = make_llm_judge_reward_fn(
                    temperature=judge_temperature,
                    max_tokens=judge_max_tokens,
                )
            except ValueError as e:
                logger.error(e)
                raise e
        else:
            # exact_match, ttrl use exact_match_reward_fn
            self.reward_fn = exact_match_reward_fn

    @classmethod
    def postprocess_episode_batch(cls, episodes: list[Episode], *, task_ids=None, workflow_args=None):
        reward_type = (workflow_args or {}).get("reward_type", "exact_match")
        mode = (workflow_args or {}).get("mode", "train")

        # Only apply majority vote reward for ttrl type during training
        # For validation, always use groundtruth labels (no majority vote)
        # For exact_match and llm_judge, reward is already computed against groundtruth
        if reward_type not in {"ttrl"}:
            return episodes

        # During validation, use groundtruth labels instead of majority vote
        if mode == "val":
            logger.info("Validation mode: using groundtruth labels instead of majority vote")
            return episodes

        # During training with ttrl, apply majority vote to get labels
        return apply_majority_vote_reward(episodes, overwrite_episode_correctness=True)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)

        query = task["query"]
        images = task["images"]

        images = decode_base64(images)

        result = await self.agent.run(query=query, images=images, uid=uid, **kwargs)
        print(f"result: {result['output'].text}")
        prediction = result["prediction"]

        reward_result = self.reward_fn(task, Action(prediction))
        if inspect.isawaitable(reward_result):
            reward_result = await reward_result
        is_correct = bool(reward_result.is_correct)

        trajectory = Trajectory(name="mmsearch_agent", task=query)
        trajectory.steps.append(
            Step(
                model_response=prediction,
                model_output=result["output"],
            )
        )
        trajectory.reward = float(reward_result.reward)

        ep = Episode(
            id=uid,
            task=task,
            termination_reason=TerminationReason.ENV_DONE,
            is_correct=is_correct,
            trajectories=[trajectory],
            metrics={
                "accuracy": bool(is_correct),
            },
        )

        labels = task.get("answer", None)
        logger.info(
            "trajectory reward_type=%s prediction=%r labels=%r reward=%s is_correct=%s",
            self.reward_type,
            prediction,
            labels,
            reward_result.reward,
            is_correct,
        )
        print(f"trajectory reward_type={self.reward_type} prediction={prediction} labels={labels} is_correct={is_correct}")
        return ep