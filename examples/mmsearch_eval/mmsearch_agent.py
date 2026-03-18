import base64
import json
from io import BytesIO

from PIL import Image

from mmsearch_tools import get_mmsearch_tools
from rllm.engine.rollout import ModelOutput, RolloutEngine


def _pil_to_data_url(image: Image.Image) -> str:
    image = image.convert("RGB")
    buf = BytesIO()
    image.save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"

SYSTEM_PROMPT = """
You can use the Search tool to look up web information when needed.
Put your final answer in the format of \boxed{answer}.
"""


class MMSearchAgent:
    """
    Minimal MMSearch agent with optional one-round tool calling.
    """

    def __init__(self, rollout_engine: RolloutEngine):
        self.rollout_engine = rollout_engine
        self.tools = get_mmsearch_tools()
        self.openai_tools = [tool.json for tool in self.tools.values()]

    @staticmethod
    def _parse_tool_call(tool_call, idx: int) -> tuple[str, str, dict]:
        tool_id = f"tool_call_{idx}"
        tool_name = ""
        args_raw = "{}"

        if isinstance(tool_call, dict):
            tool_id = tool_call.get("id", tool_id)
            if "function" in tool_call:
                fn = tool_call["function"]
                tool_name = fn.get("name", "")
                args_raw = fn.get("arguments", "{}")
            else:
                tool_name = tool_call.get("name", "")
                args_raw = tool_call.get("arguments", "{}")
        else:
            tool_id = getattr(tool_call, "id", tool_id)
            fn = getattr(tool_call, "function", None)
            if fn is not None:
                tool_name = getattr(fn, "name", "")
                args_raw = getattr(fn, "arguments", "{}")
            else:
                tool_name = getattr(tool_call, "name", "")
                args_raw = getattr(tool_call, "arguments", "{}")

        if not tool_name:
            raise ValueError(f"Invalid tool call without name: {tool_call}")

        if isinstance(args_raw, str):
            args = json.loads(args_raw)
        elif isinstance(args_raw, dict):
            args = args_raw
        else:
            raise ValueError(f"Invalid tool arguments type for {tool_name}: {type(args_raw)}")

        if not isinstance(args, dict):
            raise ValueError(f"Tool arguments must be dict for {tool_name}, got {type(args)}")

        return tool_id, tool_name, args

    async def run(self, query: str, query_image: Image.Image, uid: str, **kwargs) -> dict:
        prompt_text = query + "\n" + SYSTEM_PROMPT.strip()
        if query_image is not None:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        {"type": "image_url", "image_url": {"url": _pil_to_data_url(query_image)}},
                    ],
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ]

        first_output: ModelOutput = await self.rollout_engine.get_model_response(
            messages=messages,
            tools=self.openai_tools,
            tool_choice="auto",
            application_id=uid,
            **kwargs,
        )

        if getattr(first_output, "tool_calls", None):
            assistant_content = (first_output.content or first_output.text or "") or ""
            assistant_tool_calls = []
            tool_messages = []

            for idx, tool_call in enumerate(first_output.tool_calls):
                tool_id, tool_name, tool_args = self._parse_tool_call(tool_call, idx)
                tool = self.tools.get(tool_name)
                if tool is None:
                    raise ValueError(f"Unknown tool requested by model: {tool_name}")

                tool_output = await tool.async_forward(**tool_args)
                assistant_tool_calls.append(
                    {
                        "id": tool_id,
                        "type": "function",
                        "function": {
                            "name": tool_name,
                            "arguments": json.dumps(tool_args, ensure_ascii=False),
                        },
                    }
                )
                tool_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_id,
                        "content": tool_output.to_string(),
                    }
                )

            final_messages = messages + [
                {
                    "role": "assistant",
                    "content": assistant_content,
                    "tool_calls": assistant_tool_calls,
                }
            ]
            final_messages.extend(tool_messages)

            output: ModelOutput = await self.rollout_engine.get_model_response(
                messages=final_messages,
                tools=self.openai_tools,
                tool_choice="auto",
                application_id=uid,
                **kwargs,
            )
            prediction = (output.content or output.text or "").strip()
            return {"messages": final_messages, "output": output, "prediction": prediction}

        prediction = (first_output.content or first_output.text or "").strip()
        return {"messages": messages, "output": first_output, "prediction": prediction}

