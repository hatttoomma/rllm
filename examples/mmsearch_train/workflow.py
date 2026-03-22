from PIL import Image

from .agent import MMSearchAgent
import base64

from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason, Workflow


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _string_match(pred: str, labels: list[str]) -> bool:
    p = _normalize(pred)
    return any(p == _normalize(x) for x in labels)

def decode_base64(b64_str: list) -> Image.Image:
    # first encode str to base64 then decode to PIL.Image
    return [Image.open(BytesIO(base64.b64decode(img))) for img in b64_str]


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
                "prediction": prediction,
                "labels": labels,
                "exact_match": bool(is_correct),
                "tool_calls": result.get("tool_calls", []),
            },
        )
        return ep

