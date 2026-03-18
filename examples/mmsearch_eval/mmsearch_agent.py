import base64
import json
import re
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
You can use a text search tool when needed.
If you need to call the tool, you MUST use exactly this format:
<tool_call>{"name":"search","args":{"query":"your search text"}}</tool_call>

Rules:
1) function-name must be "search"
2) args must contain the search content
3) Put your final answer in the format of \boxed{answer}.
"""

FIRST_ROUND_PROMPT = """
Now you should answer the question with information from the search tool.
Remember to put your final answer in the format of \boxed{answer}.
"""


class MMSearchAgent:
    """
    Minimal MMSearch agent with optional one-round tool calling.
    """

    def __init__(self, rollout_engine: RolloutEngine):
        self.rollout_engine = rollout_engine
        self.tools = get_mmsearch_tools()

    @staticmethod
    def _extract_tool_call(content: str) -> tuple[str, dict] | None:
        start_tag = "<tool_call>"
        end_tag = "</tool_call>"
        if start_tag not in content or end_tag not in content:
            return None
        start = content.index(start_tag) + len(start_tag)
        end = content.index(end_tag, start)
        payload = content[start:end].strip()
        if not payload:
            raise ValueError("Empty tool_call payload.")

        call = json.loads(payload)
        if not isinstance(call, dict):
            raise ValueError("tool_call payload must be a JSON object.")

        name = call.get("name")
        args = call.get("args")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_call 'name' must be a non-empty string.")
        if args is None:
            raise ValueError("tool_call must contain 'args'.")
        return name, args

    @staticmethod
    def _normalize_search_query(args) -> str:
        if isinstance(args, str):
            query = args
        elif isinstance(args, dict):
            query = args.get("query", args.get("text"))
        else:
            raise ValueError(f"Invalid args type for search: {type(args)}")

        if not isinstance(query, str) or not query.strip():
            raise ValueError("search args must include non-empty query text.")
        return query.strip()

    @staticmethod
    def _extract_prediction(content: str) -> str:
        """
        Extract answer strictly from \\boxed{...} as required by SYSTEM_PROMPT.
        """
        matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", content, flags=re.DOTALL)
        if not matches:
            answer = ""
        answer = matches[-1].strip()
        if not answer:
            raise ValueError("Boxed answer is empty.")
        return answer

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
            application_id=uid,
            **kwargs,
        )

        first_content = (first_output.content or first_output.text or "").strip()
        tool_call = self._extract_tool_call(first_content)
        if tool_call is not None:
            tool_name, tool_args = tool_call
            if tool_name != "search":
                raise ValueError(f"Unsupported tool name: {tool_name}. Expected 'search'.")

            search_query = self._normalize_search_query(tool_args)
            tool = self.tools.get("search")
            if tool is None:
                raise ValueError("search tool is not registered.")

            tool_output = await tool.async_forward(query=search_query)
            tool_response = f"<tool_response>\n{tool_output.to_string()}\n</tool_response>"

            final_messages = messages + [
                {"role": "assistant", "content": first_content},
                {"role": "user", "content": tool_response + FIRST_ROUND_PROMPT},
            ]

            output: ModelOutput = await self.rollout_engine.get_model_response(messages=final_messages, application_id=uid, **kwargs)
            final_content = (output.content or output.text or "").strip()
            prediction = self._extract_prediction(final_content)
            return {"messages": final_messages, "output": output, "prediction": prediction}

        prediction = self._extract_prediction(first_content)
        return {"messages": messages, "output": first_output, "prediction": prediction}

