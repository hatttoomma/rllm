import json
import re
from collections.abc import Sequence

from PIL import Image

from .tools import get_tools
from rllm.engine.rollout import ModelOutput, RolloutEngine
SYSTEM_PROMPT = """
You can use tools when needed.
If you need to call the tool, you MUST use exactly this format:
<tool_call>{"name":"tool_name","arguments":{...}}</tool_call>

Rules:
1) Available tools:
   - search: args must include "query" string
   - code_interpreter: args must include "code" string, optional "timeout" integer
2) Use tools only when necessary. You may call tools for multiple rounds if needed.
3) Put your final answer in the format of \boxed{answer}.
"""

FINAL_ROUND_PROMPT = """
Now you should answer the question with information from the tools.
Remember to put your final answer in the format of \boxed{answer}.
"""


class MMSearchAgent:
    """
    MMSearch agent with configurable multi-round tool calling.
    """

    def __init__(self, rollout_engine: RolloutEngine):
        self.rollout_engine = rollout_engine
        self.tools = get_tools()

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
        args = call.get("arguments")
        if not isinstance(name, str) or not name:
            raise ValueError("tool_call 'name' must be a non-empty string.")
        if args is None:
            raise ValueError("tool_call must contain 'arguments'.")
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
    def _normalize_code_args(args) -> tuple[str, int]:
        if not isinstance(args, dict):
            raise ValueError("code_interpreter args must be an object with 'code'.")
        code = args.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError("code_interpreter args must include non-empty code.")

        timeout = args.get("timeout", 12)
        if not isinstance(timeout, int):
            raise ValueError("code_interpreter timeout must be an integer.")
        return code, timeout

    @staticmethod
    def _format_tool_response(tool_name: str, tool_result: str) -> str:
        return f"<tool_response>\nTool: {tool_name}\n{tool_result}\n</tool_response>"

    async def _run_tool(self, tool_name: str, tool_args) -> str:
        tool = self.tools.get(tool_name)
        if tool is None:
            raise ValueError(f"Tool '{tool_name}' is not registered.")

        if tool_name == "search":
            search_query = self._normalize_search_query(tool_args)
            output = await tool.async_forward(query=search_query)
            return output.to_string()

        if tool_name == "code_interpreter":
            code, timeout = self._normalize_code_args(tool_args)
            output = await tool.async_forward(code=code, timeout=timeout)
            return output.to_string()

        raise ValueError(f"Unsupported tool name: {tool_name}.")

    @staticmethod
    def _extract_prediction(content: str) -> str:
        """
        Extract answer strictly from \\boxed{...} as required by SYSTEM_PROMPT.
        """
        matches = re.findall(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}", content, flags=re.DOTALL)
        if not matches:
            return ""
        answer = matches[-1].strip()
        return answer

    async def run(
        self,
        query: str,
        images: list[Image.Image],
        uid: str,
        max_tool_call: int = 1,
        **kwargs,
    ) -> dict:
        if max_tool_call < 0:
            raise ValueError("max_tool_call must be >= 0.")

        prompt_text = query + "\n" + SYSTEM_PROMPT.strip()
        query_images = self._normalize_query_images(images)
        messages = [{"role": "user", "content": prompt_text, "images": query_images if query_images else None}]

        curr_messages = messages
        output: ModelOutput = await self.rollout_engine.get_model_response(
            messages=curr_messages,
            application_id=uid,
            **kwargs,
        )
        content = (output.content or output.text or "").strip()
        tool_calls = 0
        tool_calls_trace: list[dict] = []

        while True:
            tool_call = self._extract_tool_call(content)
            if tool_call is None:
                prediction = self._extract_prediction(content)
                return {"messages": curr_messages, "output": output, "prediction": prediction, "tool_calls": tool_calls_trace}

            if tool_calls >= max_tool_call:
                prediction = self._extract_prediction(content)
                return {"messages": curr_messages, "output": output, "prediction": prediction, "tool_calls": tool_calls_trace}

            tool_name, tool_args = tool_call
            tool_calls_trace.append({"round": tool_calls + 1, "tool": tool_name, "args": tool_args})
            tool_result = await self._run_tool(tool_name, tool_args)
            tool_calls += 1

            next_prompt = self._format_tool_response(tool_name, tool_result)
            if tool_calls == max_tool_call:
                next_prompt += FINAL_ROUND_PROMPT

            curr_messages = curr_messages + [
                {"role": "assistant", "content": content},
                {"role": "user", "content": next_prompt},
            ]
            output = await self.rollout_engine.get_model_response(messages=curr_messages, application_id=uid, **kwargs)
            content = (output.content or output.text or "").strip()

    @staticmethod
    def _normalize_query_images(query_image) -> list[Image.Image]:
        if query_image is None:
            return []
        if isinstance(query_image, Image.Image):
            return [query_image]
        if isinstance(query_image, Sequence):
            images = []
            for image in query_image:
                if image is None:
                    continue
                if not isinstance(image, Image.Image):
                    raise ValueError(f"Unsupported query image type: {type(image)}")
                images.append(image)
            return images
        if isinstance(query_image, list):
            images = []
            for image in query_image:
                if image is None:
                    continue
                if not isinstance(image, Image.Image):
                    raise ValueError(f"Unsupported query image type: {type(image)}")
                images.append(image)
            return images
        raise ValueError(f"Unsupported query image container type: {type(query_image)}")

