import base64
from io import BytesIO

from PIL import Image

from rllm.engine.rollout import ModelOutput, RolloutEngine


def _pil_to_data_url(image: Image.Image) -> str:
    image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

SYSTEM_PROMPT = "\nAnswer the question with a single word or phrasedirectly."
class MMSearchAgent:
    """
    Minimal single-turn agent for MMSearch.

    No tools. One model generation call per example.
    """

    def __init__(self, rollout_engine: RolloutEngine):
        self.rollout_engine = rollout_engine

    async def run(self, query: str, query_image: Image.Image, uid: str, **kwargs) -> dict:
        if query_image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query + SYSTEM_PROMPT},
                        {"type": "image_url", "image_url": {"url": _pil_to_data_url(query_image)}},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": query},
                    ],
                }
            ]

        output: ModelOutput = await self.rollout_engine.get_model_response(
            messages=messages,
            application_id=uid,
            **kwargs,
        )

        prediction = (output.content or output.text or "").strip()
        return {"messages": messages, "output": output, "prediction": prediction}

