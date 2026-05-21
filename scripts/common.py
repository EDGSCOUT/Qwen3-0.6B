from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

IGNORE_INDEX = -100


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    records: list[dict[str, Any]] = []
    with file_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}:{line_no} 不是合法 JSON：{exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{file_path}:{line_no} 应为 JSON 对象")
            records.append(record)
    return records


def write_jsonl(path: str | Path, records: list[dict[str, Any]]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def normalize_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """将支持的样本格式统一转换为聊天消息列表。"""
    if isinstance(record.get("messages"), list):
        messages: list[dict[str, str]] = []
        for msg in record["messages"]:
            if not isinstance(msg, dict):
                raise ValueError(f"消息必须是对象：{msg!r}")
            role = _stringify(msg.get("role", "")).strip()
            content = _stringify(msg.get("content", "")).strip()
            if not role:
                raise ValueError(f"消息缺少 role 字段：{msg!r}")
            messages.append({"role": role, "content": content})
        if not messages:
            raise ValueError("messages 不能为空")
        return messages

    system = _stringify(record.get("system", "")).strip()
    instruction = _stringify(
        record.get("instruction", record.get("prompt", record.get("question", "")))
    ).strip()
    extra_input = _stringify(record.get("input", "")).strip()
    output = record.get("output", record.get("response", record.get("answer")))

    user_parts = [part for part in [instruction, extra_input] if part]
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    if user_parts:
        messages.append({"role": "user", "content": "\n".join(user_parts)})
    if output is not None:
        messages.append({"role": "assistant", "content": _stringify(output).strip()})
    if not messages:
        raise ValueError(f"不支持的样本格式：{record!r}")
    return messages


def split_prompt_and_reference(
    messages: list[dict[str, str]],
) -> tuple[list[dict[str, str]], str | None]:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "assistant":
            return messages[:idx], messages[idx].get("content", "")
    return messages, None


def messages_from_prompt(prompt: str, system: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def apply_chat_template_text(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": add_generation_prompt,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def split_thinking_and_answer(text: str) -> tuple[str, str]:
    stripped = text.strip()
    if "</think>" not in stripped:
        return "", stripped
    thinking, answer = stripped.rsplit("</think>", 1)
    thinking = thinking.replace("<think>", "", 1).strip()
    return thinking, answer.strip()


def normalize_for_match(text: str) -> str:
    lowered = text.strip().lower()
    return re.sub(r"\s+", "", lowered)
