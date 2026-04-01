from PIL import Image

from .agent import MMSearchAgent
import base64
import binascii
import logging

from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason, Workflow
from io import BytesIO

from .reward import exact_match_reward_fn, make_llm_judge_reward_fn
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

        if reward_type == "llm_judge":
            # Only allow env-based config to keep workflow args minimal.
            judge_base_url = os.getenv("MMS_JUDGE_BASE_URL", "")
            judge_model = os.getenv("MMS_JUDGE_MODEL", "")
            judge_api_key = os.getenv("MMS_JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY")
            if not judge_base_url or not judge_model:
                raise ValueError(
                    "reward_type=llm_judge requires environment variables: "
                    "MMS_JUDGE_BASE_URL and MMS_JUDGE_MODEL (and MMS_JUDGE_API_KEY or OPENAI_API_KEY)."
                )
            self.reward_fn = make_llm_judge_reward_fn(
                base_url=judge_base_url,
                model=judge_model,
                api_key=judge_api_key,
                temperature=float(judge_temperature),
                max_tokens=int(judge_max_tokens),
            )
        else:
            self.reward_fn = exact_match_reward_fn

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)

        query = task["query"]
        images = task["images"]

        images = decode_base64(images)

        result = await self.agent.run(query=query, images=images, uid=uid, **kwargs)
        print(f"result: {result["output"]}")
        prediction = result["prediction"]

        reward_result = self.reward_fn(task, Action(prediction))
        if inspect.isawaitable(reward_result):
            reward_result = await reward_result
        is_correct = bool(reward_result.is_correct)

        trajectory = Trajectory(name="mmsearch_agent", task=query)
        trajectory.steps.append(
            Step(
                chat_completions=result["messages"] + [{"role": "assistant", "content": prediction}],
                thought=getattr(result["output"], "reasoning", "") or "",
                action=Action(prediction),
                model_response=prediction,
                model_output=result["output"],
                reward=float(reward_result.reward),
                done=True,
            )
        )

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

