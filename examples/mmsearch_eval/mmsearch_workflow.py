from PIL import Image

from mmsearch_agent import MMSearchAgent

from rllm.agents.agent import Action, Episode, Step, Trajectory
from rllm.workflows.workflow import TerminationReason, Workflow


def _normalize(s: str) -> str:
    return " ".join((s or "").strip().lower().split())


def _string_match(pred: str, labels: list[str]) -> bool:
    p = _normalize(pred)
    return any(p == _normalize(x) for x in labels)


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
        query_image = task["query_image"]
        assert isinstance(query_image, Image.Image)

        result = await self.agent.run(query=query, query_image=query_image, uid=uid, **kwargs)
        prediction = result["prediction"]

        labels = [task["gt_answer"]] + list(task.get("alternative_gt_answers", []))
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
            },
        )
        return ep

