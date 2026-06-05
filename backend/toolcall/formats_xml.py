from __future__ import annotations

import json
import re
from xml.etree import ElementTree

from .normalize import normalize_tool_name


def parse_xml_format(text: str, allowed_names: set[str]) -> list[dict[str, object]]:
    stripped = text.strip()

    tool_call_match = re.search(r"<tool_call>\s*(\{[\s\S]*?\})\s*</tool_call>", stripped, re.IGNORECASE)
    if tool_call_match:
        try:
            payload = json.loads(tool_call_match.group(1))
        except (json.JSONDecodeError, TypeError, ValueError):
            payload = None
        if isinstance(payload, dict) and payload.get("name"):
            raw_input = payload.get("input", payload.get("arguments", payload.get("args", payload.get("parameters", {}))))
            if isinstance(raw_input, str):
                try:
                    raw_input = json.loads(raw_input)
                except (json.JSONDecodeError, TypeError, ValueError):
                    raw_input = {"value": raw_input}
            return [{
                "name": normalize_tool_name(str(payload.get("name", "")), allowed_names),
                "input": raw_input if isinstance(raw_input, dict) else {},
            }]

    return []
