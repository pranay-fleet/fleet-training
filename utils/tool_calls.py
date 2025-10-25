"""Tool-call formatting and parsing utilities."""

import json
from typing import Optional


TOOL_NAME = "pick_image"


def make_tool_call_str(i: int) -> str:
    """Canonical tool-call JSON string."""
    return json.dumps({"tool_name": TOOL_NAME, "arguments": {"i": int(i)}})


def parse_tool_call_str(s: str) -> Optional[int]:
    """
    Try to parse generated text into the expected tool-call JSON and
    extract 'i'. Return None if badly formatted.
    """
    s = s.strip()
    # Heuristic: find the first JSON-looking segment
    # (You can replace this with a JSON-aware constrained decoder later.)
    start = s.find("{")
    end = s.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        obj = json.loads(s[start:end+1])
        if obj.get("tool_name") == TOOL_NAME and "arguments" in obj:
            i = obj["arguments"].get("i", None)
            if isinstance(i, int):
                return i
    except Exception:
        pass
    return None

