from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any


_CWD_PATTERNS = (
    re.compile(r"<cwd>\s*([^\r\n<]+)\s*</cwd>", re.IGNORECASE),
    re.compile(r"(?:cwd|workdir|workspace|working directory)\s*[:=]\s*([^\r\n]+)", re.IGNORECASE),
    re.compile(r"(?:current directory|current project|project root)\s*[:=]\s*([^\r\n]+)", re.IGNORECASE),
    re.compile(r"(?:当前目录|当前项目|项目根目录|工作目录|项目路径)\s*[:：]\s*([^\r\n]+)"),
)

_ABS_PATH_RE = re.compile(r"(?:[A-Za-z]:[\\/][^\r\n\s\"'`<>|]+|/[^\r\n\s\"'`<>|]+)")


def _flatten_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in {"text", "input_text"}:
                    parts.append(str(part.get("text", "")))
                elif part.get("type") == "tool_result":
                    inner = part.get("content", "")
                    parts.append(_flatten_content(inner))
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(p for p in parts if p)
    return ""


def _iter_payload_text(payload: dict[str, Any]) -> list[str]:
    texts: list[str] = []
    system = payload.get("system", "")
    if isinstance(system, str):
        texts.append(system)
    elif isinstance(system, list):
        texts.extend(_flatten_content(item) for item in system)
    for message in payload.get("messages", []) or []:
        if isinstance(message, dict):
            texts.append(_flatten_content(message.get("content", "")))
    return [text for text in texts if text]


def _clean_candidate(value: str) -> str:
    value = (value or "").strip().strip("`'\"")
    value = re.split(r"\s+(?:\(|\[|#|//|&&|\|\||and\b|-{1,2}\w|prompt_len=)", value, maxsplit=1)[0].strip()
    value = re.split(r"(?:%s|%d|%r|\{\}|\$\{|\"|')", value, maxsplit=1)[0].strip()
    value = value.rstrip(".,;，。；:：)]}>")
    if re.fullmatch(r"[A-Za-z]:[\\/]?", value):
        return ""
    return value


def _project_root_for(path_text: str) -> str:
    try:
        path = Path(path_text).expanduser().resolve(strict=False)
    except Exception:
        return path_text

    try:
        cwd = Path.cwd().resolve(strict=False)
        path_str = str(path).casefold()
        cwd_str = str(cwd).casefold()
        if path_str == cwd_str or path_str.startswith(cwd_str.rstrip("\\/") + os.sep.casefold()):
            return str(cwd)
    except Exception:
        pass

    probe = path if path.suffix == "" else path.parent
    strong_markers = (
        ("backend", "frontend"),
        ("backend", "start.py"),
        ("backend", "package.json"),
        (".git",),
    )
    for current in (probe, *probe.parents):
        try:
            if any(all((current / marker).exists() for marker in markers) for markers in strong_markers):
                return str(current)
        except OSError:
            continue
    for current in (probe, *probe.parents):
        try:
            if any((current / marker).exists() for marker in ("pyproject.toml", "package.json", "start.py", ".git")):
                return str(current)
        except OSError:
            continue
    return _default_workspace_root()


def _project_score(path_text: str) -> int:
    try:
        path = Path(path_text).expanduser()
    except Exception:
        return 0
    score = 1
    try:
        if path.exists():
            score += 4
        if path.is_dir():
            score += 3
        for marker in (".git", "backend", "frontend", "pyproject.toml", "package.json", "start.py"):
            if (path / marker).exists():
                score += 2
    except OSError:
        pass
    return score


def _normalize_path(path_text: str) -> str:
    try:
        return str(Path(path_text).expanduser().resolve(strict=False))
    except Exception:
        return path_text


def _default_workspace_root() -> str:
    configured = os.environ.get("QWEN2API_WORKSPACE_ROOT") or os.environ.get("WORKSPACE_ROOT")
    if configured:
        return _normalize_path(configured)
    return _normalize_path(str(Path.cwd()))


def derive_workspace_root(payload: dict[str, Any]) -> str:
    """Infer the client workspace root from request text, falling back to server cwd."""
    candidates: list[tuple[int, str]] = []
    for text in _iter_payload_text(payload):
        for pattern in _CWD_PATTERNS:
            for match in pattern.finditer(text):
                candidate = _clean_candidate(match.group(1))
                if candidate:
                    root = _project_root_for(candidate)
                    candidates.append((_project_score(root) + 10, root))
        for match in _ABS_PATH_RE.finditer(text):
            candidate = _clean_candidate(match.group(0))
            if candidate:
                root = _project_root_for(candidate)
                candidates.append((_project_score(root), root))

    if not candidates:
        return _default_workspace_root()

    _, best = max(candidates, key=lambda item: item[0])
    return _normalize_path(best)


def build_workspace_notice(workspace_root: str | None) -> str:
    if not workspace_root:
        return ""
    return (
        "[WORKSPACE ROOT - MUST OBEY]\n"
        f"Current project workspace root: {workspace_root}\n"
        "Resolve relative file paths under this root. Prefer file paths inside this root for Read/Write/Edit/Grep/Glob.\n"
        "Do not use stale paths from previous conversations or unrelated Desktop/Temp folders unless the current user explicitly asks for them.\n"
        "For shell commands, assume the command already starts in this workspace; prefer relative paths and do not emit `cd` into a Windows drive path.\n"
        "Only use an absolute shell path when a previous tool result proves that exact path works in the shell environment.\n"
        "[/WORKSPACE ROOT]"
    )


def build_workspace_final_reminder(workspace_root: str | None) -> str:
    if not workspace_root:
        return ""
    return (
        f"[WORKSPACE REMINDER]: The active workspace is {workspace_root}. "
        "Do not switch to stale or unrelated project paths."
    )
