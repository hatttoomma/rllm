from PIL import Image

from .agent import MMSearchAgent
import base64
import binascii
import logging

from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason, Workflow
from io import BytesIO


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

    def __init__(self, rollout_engine, executor=None, **kwargs):
        super().__init__(rollout_engine, executor, **kwargs)
        self.agent = MMSearchAgent(rollout_engine)

    async def run(self, task: dict, uid: str, **kwargs) -> Episode:
        self.reset(task=task, uid=uid)

        query = task["query"]
        images = task["images"]

        images = decode_base64(images)

        result = await self.agent.run(query=query, images=images, uid=uid, **kwargs)
        prediction = result["prediction"]

        labels = task["answer"] # Answer is a list here
        is_correct = _string_match(prediction, labels)

        trajectory = Trajectory(name="mmsearch_agent", task=query)
        trajectory.steps.append(
            Step(
                chat_completions=result["messages"] + [{"role": "assistant", "content": prediction}],
                thought=getattr(result["output"], "reasoning", "") or "",
                action=Action(prediction),
                model_response=prediction,
                model_output=result["output"],
                reward=1.0 if is_correct else 0.0,
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
                "exact_match": bool(is_correct),
            },
        )

        logger.info("trajectory prediction=%r labels=%r exact_match=%s", prediction, labels, is_correct)
        return ep

