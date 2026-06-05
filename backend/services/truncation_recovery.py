"""Tool-call truncation detection and continuation helpers.

The primary prompt-visible protocol is QNML:

<|QNML|tool_calls>
  <|QNML|invoke name="TOOL_NAME">
    <|QNML|parameter name="ARG"><![CDATA[value]]></|QNML|parameter>
  </|QNML|invoke>
</|QNML|tool_calls>

Compatibility parsing still recognizes legacy XML ``<tool_calls>`` / ``<tool_call>``
and old marker blocks, but those are not the main prompt protocol.
"""

from __future__ import annotations

import re


_QNML_TOOL_CALLS_OPEN_RE = re.compile(r"<\s*\|\s*QNML\s*\|\s*tool_calls\b[^>]*>", re.IGNORECASE)
_QNML_TOOL_CALLS_CLOSE_RE = re.compile(r"<\s*/\s*\|\s*QNML\s*\|\s*tool_calls\s*>", re.IGNORECASE)
_QNML_INVOKE_OPEN_RE = re.compile(r"<\s*\|\s*QNML\s*\|\s*invoke\b[^>]*>", re.IGNORECASE)
_QNML_INVOKE_CLOSE_RE = re.compile(r"<\s*/\s*\|\s*QNML\s*\|\s*invoke\s*>", re.IGNORECASE)
_QNML_PARAMETER_OPEN_RE = re.compile(r"<\s*\|\s*QNML\s*\|\s*parameter\b[^>]*>", re.IGNORECASE)
_QNML_PARAMETER_CLOSE_RE = re.compile(r"<\s*/\s*\|\s*QNML\s*\|\s*parameter\s*>", re.IGNORECASE)

_LEGACY_TOOL_CALLS_OPEN_RE = re.compile(r"<\s*tool_calls\b[^>]*>", re.IGNORECASE)
_LEGACY_TOOL_CALLS_CLOSE_RE = re.compile(r"<\s*/\s*tool_calls\s*>", re.IGNORECASE)
_LEGACY_INVOKE_OPEN_RE = re.compile(r"<\s*invoke\b[^>]*>", re.IGNORECASE)
_LEGACY_INVOKE_CLOSE_RE = re.compile(r"<\s*/\s*invoke\s*>", re.IGNORECASE)
_LEGACY_PARAMETER_OPEN_RE = re.compile(r"<\s*parameter\b[^>]*>", re.IGNORECASE)
_LEGACY_PARAMETER_CLOSE_RE = re.compile(r"<\s*/\s*parameter\s*>", re.IGNORECASE)
_LEGACY_TOOL_CALL_OPEN_RE = re.compile(r"<\s*tool_call\b[^>]*>", re.IGNORECASE)
_LEGACY_TOOL_CALL_CLOSE_RE = re.compile(r"<\s*/\s*tool_call\s*>", re.IGNORECASE)

_TOOL_CALL_OPEN_RE = re.compile(r"##TOOL_CALL##", re.IGNORECASE)
_TOOL_CALL_CLOSE_RE = re.compile(r"##END_CALL##", re.IGNORECASE)
_CDATA_OPEN_RE = re.compile(r"<!\[CDATA\[", re.IGNORECASE)
_CDATA_CLOSE_RE = re.compile(r"\]\]>")
_PARTIAL_TOOL_MARKER_RE = re.compile(
    r"(?:<\s*/?\s*(?:\|\s*QNML(?:\s*\|\s*(?:tool_calls|invoke|parameter)?)?|tool_calls?|invoke|parameter)"
    r"|##\s*(?:TOOL_CALL|END_CALL)?)\s*$",
    re.IGNORECASE,
)


def _count(pattern: re.Pattern[str], text: str) -> int:
    return len(pattern.findall(text))


def _has_unclosed(open_re: re.Pattern[str], close_re: re.Pattern[str], text: str) -> bool:
    return _count(open_re, text) > _count(close_re, text)


def _contains_tool_marker(text: str) -> bool:
    lowered = text.lower()
    return any(
        marker in lowered
        for marker in (
            "<|qnml|tool_calls",
            "</|qnml|tool_calls",
            "<|qnml|invoke",
            "</|qnml|invoke",
            "<|qnml|parameter",
            "</|qnml|parameter",
            "<tool_calls",
            "</tool_calls",
            "<invoke",
            "</invoke",
            "<parameter",
            "</parameter",
            "<tool_call",
            "</tool_call",
            "##tool_call##",
            "##end_call##",
        )
    )


def is_truncated(text: str) -> bool:
    """Return True when output appears cut off inside a tool-call block."""
    if not text or not text.strip():
        return False

    trimmed = text.rstrip()
    if _PARTIAL_TOOL_MARKER_RE.search(trimmed):
        return True

    if not _contains_tool_marker(trimmed):
        return False

    # Primary protocol: QNML.
    if _has_unclosed(_QNML_TOOL_CALLS_OPEN_RE, _QNML_TOOL_CALLS_CLOSE_RE, trimmed):
        return True
    if _has_unclosed(_QNML_INVOKE_OPEN_RE, _QNML_INVOKE_CLOSE_RE, trimmed):
        return True
    if _has_unclosed(_QNML_PARAMETER_OPEN_RE, _QNML_PARAMETER_CLOSE_RE, trimmed):
        return True

    # Compatibility: legacy XML / canonical XML.
    if _has_unclosed(_LEGACY_TOOL_CALLS_OPEN_RE, _LEGACY_TOOL_CALLS_CLOSE_RE, trimmed):
        return True
    if _has_unclosed(_LEGACY_INVOKE_OPEN_RE, _LEGACY_INVOKE_CLOSE_RE, trimmed):
        return True
    if _has_unclosed(_LEGACY_PARAMETER_OPEN_RE, _LEGACY_PARAMETER_CLOSE_RE, trimmed):
        return True
    if _has_unclosed(_LEGACY_TOOL_CALL_OPEN_RE, _LEGACY_TOOL_CALL_CLOSE_RE, trimmed):
        return True

    # Compatibility: old marker JSON block.
    if _has_unclosed(_TOOL_CALL_OPEN_RE, _TOOL_CALL_CLOSE_RE, trimmed):
        return True

    # Unclosed CDATA usually means a QNML/legacy parameter is still incomplete.
    if _count(_CDATA_OPEN_RE, trimmed) > _count(_CDATA_CLOSE_RE, trimmed):
        return True

    return False


def deduplicate_continuation(existing: str, continuation: str) -> str:
    """Remove the longest duplicate overlap between existing tail and continuation head."""
    if not existing or not continuation:
        return continuation
    max_overlap = min(500, len(existing), len(continuation))
    if max_overlap < 10:
        return continuation

    best_overlap = 0
    for length in range(max_overlap, 9, -1):
        prefix = continuation[:length]
        if existing.endswith(prefix):
            best_overlap = length
            break

    if best_overlap >= 10:
        return continuation[best_overlap:]

    tail_lines = existing.splitlines()[-20:]
    cont_lines = continuation.splitlines()
    if tail_lines and cont_lines:
        first_cont = cont_lines[0].strip()
        if first_cont:
            for i in range(len(tail_lines)):
                if tail_lines[i].strip() != first_cont:
                    continue
                matched = 1
                for k in range(1, len(cont_lines)):
                    if i + k >= len(tail_lines):
                        break
                    if cont_lines[k].strip() == tail_lines[i + k].strip():
                        matched += 1
                    else:
                        break
                if matched >= 2:
                    return "\n".join(cont_lines[matched:])

    return continuation


def build_continuation_prompt(partial_response: str, anchor_chars: int = 2000) -> tuple[str, str]:
    """Build the ``(assistant_context, user_followup)`` continuation prompt."""
    anchor = partial_response[-anchor_chars:] if len(partial_response) > anchor_chars else partial_response
    assistant_ctx = ("...\n" + anchor) if len(partial_response) > anchor_chars else anchor
    followup = (
        "Your previous response was cut off in the middle of a QNML tool-call block. "
        "The last part was:\n\n"
        "```\n"
        f"...{anchor[-300:] if len(anchor) > 300 else anchor}\n"
        "```\n\n"
        "Continue EXACTLY from where you stopped. DO NOT repeat any content already generated. "
        "DO NOT restart the response. Output ONLY the remaining QNML/tool-call text, "
        "starting immediately from the cut-off point."
    )
    return assistant_ctx, followup
