"""
MMSearch eval tools.
"""

import http.client
import json
import os

from rllm.tools.code_tools.python_interpreter import PythonInterpreter
from rllm.tools.tool_base import Tool, ToolOutput


class SerperSearchTool(Tool):
    """Minimal Serper-based web search tool."""

    def __init__(self):
        self._json = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search the web using Serper and return concise snippets.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query",
                        }
                    },
                    "required": ["query"],
                },
            },
        }
        super().__init__(name="search", description="Search the web using Serper.")

    async def async_forward(self, query: str, **kwargs) -> ToolOutput:
        api_key = os.getenv("SERPER_API_KEY")
        if not api_key:
            raise RuntimeError("SERPER_API_KEY is required for Search tool.")

        conn = http.client.HTTPSConnection("google.serper.dev", timeout=20)
        payload = json.dumps({"q": query, "num": 5})
        headers = {
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        }

        conn.request("POST", "/search", payload, headers)
        resp = conn.getresponse()
        raw = resp.read().decode("utf-8")
        conn.close()

        if resp.status != 200:
            raise RuntimeError(f"Serper API request failed: HTTP {resp.status}, body={raw[:500]}")

        data = json.loads(raw)
        organic = data.get("organic")
        if organic is None:
            raise RuntimeError(f"Serper response missing 'organic': {raw[:500]}")

        lines = []
        for idx, item in enumerate(organic[:5], start=1):
            title = item.get("title", "Untitled")
            link = item.get("link", "")
            snippet = item.get("snippet", "")
            lines.append(f"{idx}. {title}\nURL: {link}\nSnippet: {snippet}")

        output = f"Search results for: {query}\n\n" + ("\n\n".join(lines) if lines else "No organic results.")
        return ToolOutput(name=self.name, output=output)


def get_tools() -> dict[str, Tool]:
    """Return tools used by MMSearch agent."""
    return {
        "search": SerperSearchTool(),
        "code_interpreter": PythonInterpreter(
            backend="local",
            name="code_interpreter",
            description="Execute Python code to compute, transform data, or verify intermediate results.",
        ),
    }



