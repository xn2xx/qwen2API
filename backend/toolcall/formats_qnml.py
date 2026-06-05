from __future__ import annotations

import html
import json
import re
import unicodedata
from typing import Any
from xml.etree import ElementTree

from .normalize import normalize_tool_name

QNML_OPEN = "<|QNML|tool_calls>"
QNML_CLOSE = "</|QNML|tool_calls>"
QNML_INVOKE_OPEN = "<|QNML|invoke"
QNML_INVOKE_CLOSE = "</|QNML|invoke>"
QNML_PARAMETER_OPEN = "<|QNML|parameter"
QNML_PARAMETER_CLOSE = "</|QNML|parameter>"

_RAW_STRING_PARAM_NAMES = {
    "content",
    "command",
    "cmd",
    "script",
    "code",
    "prompt",
    "file_content",
    "old_string",
    "new_string",
    "insert_text",
    "patch",
    "pattern",
    "text",
    "query",
    "url",
    "path",
    "file_path",
}


_QNML_TAG_RE = re.compile(
    r"<\s*(/?)\s*(?:\|\s*)?QNML\s*(?:\|\s*|\s+)?"
    r"(tool_calls|tool-calls|toolcalls|invoke|parameter)\b([^>]*)>",
    re.IGNORECASE,
)
_CDATA_SPAN_RE = re.compile(r"<!\[CDATA\[[\s\S]*?\]\]>", re.IGNORECASE)
_XML_TAG_RE = re.compile(r"<\s*(/?)\s*([^<>]*?)\s*(/?)\s*>", re.DOTALL)
_TOOL_LOCAL_NAME_RE = re.compile(
    r"(?i)(tool\s*[-_ ]\s*calls|toolcalls|invoke|parameter)"
)
_NAME_ATTR_RE = re.compile(
    r"(?is)(?:^|[\s|])name\s*=\s*(?:\"([^\"]*)\"|'([^']*)'|([^\s|/]+))"
)

_MARKUP_TRANSLATION = str.maketrans({
    "＜": "<",
    "＞": ">",
    "／": "/",
    "∕": "/",
    "⁄": "/",
    "＝": "=",
    "｜": "|",
    "│": "|",
    "┃": "|",
    "▏": "|",
    "▕": "|",
    "“": '"',
    "”": '"',
    "„": '"',
    "‟": '"',
    "‘": "'",
    "’": "'",
    "‛": "'",
    "﹤": "<",
    "﹥": ">",
    "Ο": "O",
    "ο": "o",
    "О": "O",
    "о": "o",
    "А": "A",
    "а": "a",
    "С": "C",
    "с": "c",
    "Е": "E",
    "е": "e",
    "Т": "T",
    "т": "t",
    "М": "M",
    "м": "m",
    "Ѕ": "S",
    "ѕ": "s",
    "Ι": "I",
    "І": "I",
    "і": "i",
    "Ν": "N",
    "η": "n",
})


def _canonicalize_markup_piece(piece: str) -> str:
    if not piece:
        return piece
    piece = unicodedata.normalize("NFKC", piece)
    piece = piece.translate(_MARKUP_TRANSLATION)
    piece = piece.replace("\u3000", " ").replace("\u00a0", " ").replace("▁", " ")
    # Drop common zero-width/BOM characters that models sometimes insert inside tags.
    piece = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", piece)
    return piece


def canonicalize_qnml_markup(text: str) -> str:
    """Canonicalize QNML tag-shell variants without touching CDATA payloads."""
    if not text:
        return text
    out: list[str] = []
    last = 0
    for match in _CDATA_SPAN_RE.finditer(text):
        out.append(_canonicalize_markup_piece(text[last:match.start()]))
        out.append(match.group(0))
        last = match.end()
    out.append(_canonicalize_markup_piece(text[last:]))
    return "".join(out)


def strip_markdown_fenced_tool_examples(text: str) -> str:
    """Strip markdown fenced examples while preserving QNML/XML inside CDATA."""
    if not text:
        return text
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    in_cdata = False

    for line in lines:
        if in_cdata:
            out.append(line)
            if "]]>" in line:
                in_cdata = False
            continue

        cdata_idx = line.find("<![CDATA[")
        fence_idx = min([idx for idx in (line.find("```"), line.find("~~~")) if idx >= 0], default=-1)
        if cdata_idx >= 0 and (fence_idx < 0 or cdata_idx < fence_idx):
            out.append(line)
            if "]]>" not in line[cdata_idx:]:
                in_cdata = True
            continue

        stripped = line.lstrip()
        if not in_fence:
            if stripped.startswith("```") or stripped.startswith("~~~"):
                fence_char = stripped[0]
                fence_len = len(stripped) - len(stripped.lstrip(fence_char))
                if fence_len >= 3:
                    in_fence = True
                    continue
            out.append(line)
            continue

        if stripped.startswith(fence_char * fence_len):
            tail = stripped[fence_len:].strip()
            if not tail:
                in_fence = False
                fence_char = ""
                fence_len = 0

    return "".join(out)


def _rewrite_qnml_tags_to_xml(text: str) -> str:
    """Rewrite QNML/QNML/prefixed tool markup to canonical XML tags.

    This mirrors ds2api's tolerant scanner at a practical level: canonical XML
    is kept, QNML remains accepted, and QNML variants such as
    ``<dsml-tool-calls>``, ``<vendor_invoke>`` or ``<DSmartParameter>`` are
    normalized before ElementTree parses the candidate.
    """

    def repl(match: re.Match[str]) -> str:
        raw = match.group(0)
        tag = _canonical_tool_tag(match)
        return tag if tag is not None else raw

    return _XML_TAG_RE.sub(repl, text)


def _canonical_tool_tag(match: re.Match[str]) -> str | None:
    closing = bool(match.group(1))
    body = (match.group(2) or "").strip()
    self_closing = bool(match.group(3)) or body.endswith("/")
    if body.endswith("/"):
        body = body[:-1].rstrip()

    name, attrs_start = _detect_tool_local_name(body)
    if not name:
        return None

    if closing:
        return f"</{name}>"

    attrs = ""
    attr_text = body[attrs_start:].strip(" |") if attrs_start < len(body) else ""
    attr_name = _extract_name_attr(attr_text)
    if attr_name is not None and name in {"invoke", "parameter"}:
        attrs = f' name="{html.escape(attr_name, quote=True)}"'
    suffix = "/" if self_closing else ""
    return f"<{name}{attrs}{suffix}>"


def _detect_tool_local_name(body: str) -> tuple[str | None, int]:
    for match in _TOOL_LOCAL_NAME_RE.finditer(body):
        before = body[:match.start()]
        after = body[match.end():]
        if not _tool_name_prefix_allowed(before):
            continue
        if not _tool_name_suffix_allowed(after):
            continue
        raw_name = re.sub(r"[\s_-]+", "_", match.group(1).lower())
        if raw_name in {"tool_calls", "toolcalls", "tool_call", "tool-call"}:
            return "tool_calls", match.end()
        if raw_name in {"invoke", "parameter"}:
            return raw_name, match.end()
    return None, 0


def _tool_name_prefix_allowed(prefix: str) -> bool:
    prefix = prefix.strip()
    if not prefix:
        return True
    # Accept known protocol/vendor prefixes while rejecting ordinary XML names
    # that merely contain a tool word in the middle.
    if any(sep in prefix for sep in ("|", "_", "-", " ")):
        return True
    return bool(re.fullmatch(r"(?i)(?:q?n?d?s?ml|dsmart|agent|tool|[a-z0-9]{1,32})", prefix))


def _tool_name_suffix_allowed(suffix: str) -> bool:
    suffix = suffix.lstrip()
    return not suffix or suffix[0] in {' ', '\t', '\r', '\n', '|', '/', '>'} or suffix.startswith('name=')


def _extract_name_attr(attr_text: str) -> str | None:
    if not attr_text:
        return None
    match = _NAME_ATTR_RE.search(attr_text)
    if not match:
        return None
    value = next((group for group in match.groups() if group is not None), "")
    return html.unescape(value.strip())


def _cdata(value: str) -> str:
    # XML CDATA cannot contain ]]>; split it into adjacent CDATA sections.
    return "<![CDATA[" + str(value).replace("]]>", "]]]]><![CDATA[>") + "]]>"


def _render_param_value(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return _cdata(value)
    if isinstance(value, list):
        return "".join(f"<item>{_render_param_value(item)}</item>" for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key, item in value.items():
            safe_key = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(key).strip() or "field")
            parts.append(f"<{safe_key}>{_render_param_value(item)}</{safe_key}>")
        return "".join(parts)
    return _cdata(str(value))


_CONTROL_TOOL_ALIASES = {
    "agent",
    "u_agent",
    "askuserquestion",
    "u_askuserquestion",
    "croncreate",
    "u_croncreate",
    "crondelete",
    "u_crondelete",
    "cronlist",
    "u_cronlist",
    "enterplanmode",
    "u_enterplanmode",
    "exitplanmode",
    "u_exitplanmode",
    "enterworktree",
    "u_enterworktree",
    "exitworktree",
    "u_exitworktree",
    "monitor",
    "u_monitor",
    "pushnotification",
    "u_pushnotification",
    "schedulewakeup",
    "u_schedulewakeup",
    "taskcreate",
    "u_taskcreate",
    "taskdelete",
    "u_taskdelete",
    "taskget",
    "u_taskget",
    "tasklist",
    "u_tasklist",
    "taskoutput",
    "u_taskoutput",
    "taskstop",
    "u_taskstop",
    "taskupdate",
    "u_taskupdate",
}


def _tool_alias_key(name: str) -> str:
    return re.sub(r"[^a-z0-9_]+", "", (name or "").lower())


def _preferred_example_names(names: list[str]) -> list[str]:
    safe = [name for name in names if _tool_alias_key(name) not in _CONTROL_TOOL_ALIASES]
    preferred_order = (
        "fs_open_file",
        "shell_run",
        "fs_put_file",
        "fs_patch_file",
        "path_find",
        "text_search",
        "http_get_url",
        "web_query",
    )
    ordered: list[str] = []
    for preferred in preferred_order:
        for name in safe:
            if _tool_alias_key(name) == preferred and name not in ordered:
                ordered.append(name)
    for name in safe:
        if name not in ordered:
            ordered.append(name)
    return ordered or names


def render_qnml_tool_calls(tool_calls: list[dict[str, Any]]) -> str:
    """Render one QNML wrapper containing one or more invoke blocks."""
    lines = [QNML_OPEN]
    for tool_call in tool_calls:
        if not isinstance(tool_call, dict):
            continue
        name = str(tool_call.get("name", "")).strip()
        if not name:
            continue
        input_data = tool_call.get("input", tool_call.get("arguments", tool_call.get("args", {})))
        if not isinstance(input_data, dict):
            input_data = {}
        lines.append(f'  <|QNML|invoke name="{html.escape(name, quote=True)}">')
        for key, value in input_data.items():
            safe_key = html.escape(str(key), quote=True)
            lines.append(f'    <|QNML|parameter name="{safe_key}">{_render_param_value(value)}</|QNML|parameter>')
        lines.append("  </|QNML|invoke>")
    lines.append(QNML_CLOSE)
    return "\n".join(lines)


def render_qnml_tool_call(name: str, input_data: dict[str, Any] | None) -> str:
    return render_qnml_tool_calls([{"name": name, "input": input_data if isinstance(input_data, dict) else {}}])


def build_qnml_tool_instructions(names: list[str], tool_schemas: list[str] | None = None, *, heavy_profile: bool = False) -> str:
    schema_block = ""
    if tool_schemas:
        schema_block = "You have access to these tools:\n\n" + "\n\n".join(tool_schemas).strip() + "\n\n"

    available = ", ".join(names)
    extra_heavy = ""
    if heavy_profile:
        extra_heavy = """
EXECUTION RULES - CRITICAL:
- When the user gives a task, start immediately by emitting the required QNML tool block if a tool is needed.
- If multiple operations are required, keep using tool calls across turns until the task is complete.
- Do NOT ask for confirmation unless the user explicitly asks you to ask.
- For file/config tasks prefer Read/Edit/Write style tools. Use shell tools only when shell behavior is required.
- Prefer direct project tools. Use Agent-like delegation tools only when clearly necessary for the current task context or explicitly requested; if uncertain, keep using direct tools.
- If a Read/read_file-style result says the file is unchanged or provides no body, do not repeatedly call the same read request.
- Control/task/scheduling tools are available but should be used only when they directly serve the current task context or the current user explicitly requests them; if uncertain, do not call them.
- MCP/plugin tools are ordinary available tools. If the CURRENT user asks for an MCP tool and it appears in the action names, call that exact MCP/plugin action through QNML; do not replace it with shell commands.
- For project-file tasks, keep all relative file paths rooted at the active workspace notice.
- Bash/shell_run executes under bash, not cmd.exe or PowerShell. Prefer workspace-relative commands. Do not use Windows cmd syntax such as `cd /d`; avoid raw Windows drive paths inside bash commands unless the tool result proves they work.
""".strip() + "\n\n"

    examples = []
    example_names = _preferred_example_names(names)
    if example_names:
        n = example_names[0]
        examples.append(
            "Example A — Single tool:\n"
            "<|QNML|tool_calls>\n"
            f"  <|QNML|invoke name=\"{n}\">\n"
            "    <|QNML|parameter name=\"query\"><![CDATA[actual value here]]></|QNML|parameter>\n"
            "  </|QNML|invoke>\n"
            "</|QNML|tool_calls>"
        )
    if len(example_names) >= 2:
        examples.append(
            "Example B — Two tools in parallel:\n"
            "<|QNML|tool_calls>\n"
            f"  <|QNML|invoke name=\"{example_names[0]}\">\n"
            "    <|QNML|parameter name=\"query\"><![CDATA[first actual value]]></|QNML|parameter>\n"
            "  </|QNML|invoke>\n"
            f"  <|QNML|invoke name=\"{example_names[1]}\">\n"
            "    <|QNML|parameter name=\"path\"><![CDATA[second actual value]]></|QNML|parameter>\n"
            "  </|QNML|invoke>\n"
            "</|QNML|tool_calls>"
        )

    examples_block = "\n\n".join(examples)
    if examples_block:
        examples_block = "\n\nCORRECT EXAMPLES:\n\n" + examples_block

    return f"""=== QNML TOOL CALL PROTOCOL ===
{schema_block}QNML blocks are client-parsed text markers, not native function calls. Use tools only when needed.
Available action names: {available}

FORMAT:
<|QNML|tool_calls>
  <|QNML|invoke name="TOOL_NAME">
    <|QNML|parameter name="ARG"><![CDATA[value]]></|QNML|parameter>
  </|QNML|invoke>
</|QNML|tool_calls>

RULES:
1) If calling tools, output only one <|QNML|tool_calls> block; no markdown fences or extra prose.
2) Put one or more <|QNML|invoke> nodes under the wrapper. Use exact tool and parameter names from the schema.
3) Strings use <![CDATA[...]]>; objects use nested XML elements; arrays repeat <item>; numbers/bools/null stay plain text.
4) Never emit empty required parameters, especially shell commands. If required info is unknown, ask normally.
5) After [Tool Result], continue with more QNML calls only if needed; otherwise answer normally.
6) Runtime also accepts legacy <tool_calls>/<invoke>/<parameter>, but QNML is preferred.
7) Bash/shell_run runs in bash. Prefer relative commands from the active workspace; do not use cmd.exe-only `cd /d` or unverified raw Windows drive paths.
{extra_heavy}Remember: the preferred tool-call form is <|QNML|tool_calls>...</|QNML|tool_calls>.{examples_block}
=== END QNML TOOL INSTRUCTIONS ==="""


def _normalize_qnml_to_xml(candidate: str) -> str:
    text = canonicalize_qnml_markup(candidate.strip())
    return _repair_missing_parameter_closes(_rewrite_qnml_tags_to_xml(text))


def _repair_missing_parameter_closes(xml_text: str) -> str:
    """Repair generated QNML where a parameter body is closed by </invoke>.

    This is intentionally narrow: it only inserts a missing </parameter> when a
    parameter opener reaches the next parameter or invoke close without a
    parameter close. It does not try to repair arbitrary XML.
    """
    if "<parameter" not in xml_text:
        return xml_text

    def repl(match: re.Match[str]) -> str:
        opener = match.group(1)
        body = match.group(2)
        if opener.rstrip().endswith("/>") or "</parameter" in body.lower():
            return match.group(0)
        return f"{opener}{body}</parameter>"

    pattern = re.compile(
        r"(<parameter\b[^>]*>)([\s\S]*?)(?=<parameter\b|</invoke\s*>)",
        re.IGNORECASE,
    )
    previous = None
    repaired = xml_text
    while previous != repaired:
        previous = repaired
        repaired = pattern.sub(repl, repaired)
    return repaired


def _extract_wrapped_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    canonical = _normalize_qnml_to_xml(strip_markdown_fenced_tool_examples(text))
    for match in re.finditer(r"<tool_calls\b[\s\S]*?</tool_calls>", canonical, flags=re.IGNORECASE):
        candidates.append(match.group(0))
    if candidates:
        return candidates

    # Repair the ds2api-compatible case where the opening wrapper is missing but
    # a closing tool_calls tag exists after an invoke block.
    invoke_match = re.search(r"<invoke\b[\s\S]*", canonical, flags=re.IGNORECASE)
    if invoke_match:
        close_match = re.search(r"</tool_calls\s*>", canonical[invoke_match.start():], flags=re.IGNORECASE)
        if close_match:
            start = invoke_match.start()
            end = start + close_match.start()
            candidates.append("<tool_calls>" + canonical[start:end] + "</tool_calls>")

    # Some upstream responses complete the invoke block but omit the outer close.
    # Treat that as a recoverable wrapper omission instead of leaking QNML text.
    open_match = re.search(r"<tool_calls\b[\s\S]*", canonical, flags=re.IGNORECASE)
    if open_match and not re.search(r"</tool_calls\s*>", canonical[open_match.start():], flags=re.IGNORECASE):
        tail = canonical[open_match.start():]
        if re.search(r"</invoke\s*>", tail, flags=re.IGNORECASE):
            candidates.append(tail + "</tool_calls>")
    return candidates


def _coerce_scalar(text: str, *, raw_string: bool = False) -> Any:
    value = html.unescape((text or "").strip())
    if raw_string:
        return value
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == "null":
        return None
    if re.fullmatch(r"-?\d+", value):
        try:
            return int(value)
        except ValueError:
            pass
    if re.fullmatch(r"-?\d+\.\d+", value):
        try:
            return float(value)
        except ValueError:
            pass
    return value


def _inner_xml(node: ElementTree.Element) -> str:
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in list(node):
        parts.append(ElementTree.tostring(child, encoding="unicode"))
    return "".join(parts).strip()


def _node_to_value(node: ElementTree.Element, *, param_name: str = "") -> Any:
    children = list(node)
    raw_string = param_name in _RAW_STRING_PARAM_NAMES
    if not children:
        return _coerce_scalar("".join(node.itertext()), raw_string=raw_string)
    if raw_string:
        # For code/file/script parameters, preserve all visible text rather than
        # structure-promoting incidental XML/HTML fragments.
        return _inner_xml(node)
    if all(child.tag == "item" for child in children):
        return [_node_to_value(child) for child in children]
    result: dict[str, Any] = {}
    for child in children:
        value = _node_to_value(child)
        if child.tag in result:
            existing = result[child.tag]
            if not isinstance(existing, list):
                result[child.tag] = [existing]
            result[child.tag].append(value)
        else:
            result[child.tag] = value
    return result


def _parse_root(root: ElementTree.Element, allowed_names: set[str]) -> list[dict[str, object]]:
    if root.tag != "tool_calls":
        return []
    calls: list[dict[str, object]] = []
    for invoke in root.findall("invoke"):
        name = invoke.attrib.get("name", "").strip()
        if not name:
            continue
        arguments: dict[str, Any] = {}
        for child in invoke.findall("parameter"):
            param_name = child.attrib.get("name", "").strip()
            if not param_name:
                continue
            arguments[param_name] = _node_to_value(child, param_name=param_name)
        calls.append({"name": normalize_tool_name(name, allowed_names), "input": arguments})
    return calls


def _decode_regex_param_body(body: str, *, param_name: str = "") -> Any:
    body = body or ""
    cdata_parts = re.findall(r"<!\[CDATA\[([\s\S]*?)\]\]>", body, flags=re.IGNORECASE)
    if cdata_parts:
        value = "".join(cdata_parts)
        return value if param_name in _RAW_STRING_PARAM_NAMES else _coerce_scalar(value)

    if param_name in _RAW_STRING_PARAM_NAMES:
        return html.unescape(body.strip())
    return _coerce_scalar(body)


def _parse_candidate_with_regex(candidate: str, allowed_names: set[str]) -> list[dict[str, object]]:
    """Tolerant QNML parser for raw HTML/CSS inside string parameters.

    ElementTree is strict XML; generated Write/content parameters sometimes omit
    CDATA and contain HTML tags. The protocol delimiters are still unambiguous, so
    parse invoke/parameter shells with regex and preserve raw string bodies.
    """
    xml_text = _normalize_qnml_to_xml(candidate)
    calls: list[dict[str, object]] = []
    for invoke in re.finditer(r"<invoke\b([^>]*)>([\s\S]*?)</invoke\s*>", xml_text, flags=re.IGNORECASE):
        name = _extract_name_attr(invoke.group(1) or "") or ""
        if not name.strip():
            continue
        body = invoke.group(2) or ""
        arguments: dict[str, Any] = {}
        for param in re.finditer(r"<parameter\b([^>]*)>([\s\S]*?)</parameter\s*>", body, flags=re.IGNORECASE):
            param_name = _extract_name_attr(param.group(1) or "") or ""
            if not param_name.strip():
                continue
            arguments[param_name.strip()] = _decode_regex_param_body(param.group(2) or "", param_name=param_name.strip())
        calls.append({"name": normalize_tool_name(name.strip(), allowed_names), "input": arguments})
    return calls


def parse_qnml_format(text: str, allowed_names: set[str]) -> list[dict[str, object]]:
    calls_out: list[dict[str, object]] = []
    for candidate in _extract_wrapped_candidates(text):
        xml_text = _normalize_qnml_to_xml(candidate)
        try:
            root = ElementTree.fromstring(xml_text)
        except ElementTree.ParseError:
            calls = _parse_candidate_with_regex(candidate, allowed_names)
            if calls:
                calls_out.extend(calls)
            continue
        calls = _parse_root(root, allowed_names)
        if calls:
            calls_out.extend(calls)
    if calls_out:
        return calls_out

    return []
