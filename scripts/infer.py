from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_accelerate_available

from common import apply_chat_template_text, messages_from_prompt, normalize_messages, split_thinking_and_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Qwen3 推理。")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-0.6B")
    parser.add_argument("--prompt", default=None, help="单轮推理的用户提示词。")
    parser.add_argument("--system", default=None, help="可选的 system prompt。")
    parser.add_argument("--messages_file", default=None, help="包含 messages 或单条样本对象的 JSON 文件。")
    parser.add_argument("--interactive", action="store_true", help="启动命令行多轮对话。")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_context_tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable_thinking", "--enable-thinking", dest="enable_thinking", action="store_true")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument(
        "--torch_dtype",
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="加载模型时使用的数据类型。",
    )
    return parser.parse_args()


def resolve_dtype(name: str, device: torch.device | None = None) -> torch.dtype | str:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if device is not None:
        if device.type in {"cuda", "mps"}:
            return torch.float16
        return torch.float32
    return "auto"


def load_messages_from_file(path: str | Path) -> list[dict[str, str]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        return [{"role": str(item["role"]), "content": str(item.get("content", ""))} for item in data]
    if isinstance(data, dict):
        return normalize_messages(data)
    raise ValueError("--messages_file 必须包含 JSON 对象或 JSON 消息列表")


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_device_map_auto() -> bool:
    return is_accelerate_available() and torch.cuda.is_available()


def token_count(tokenizer: Any, messages: list[dict[str, str]], enable_thinking: bool) -> int:
    text = apply_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def trim_history(
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_context_tokens: int,
    enable_thinking: bool,
) -> list[dict[str, str]]:
    if len(messages) <= 2:
        return messages
    system_messages = [msg for msg in messages[:1] if msg["role"] == "system"]
    turns = messages[len(system_messages) :]
    while len(turns) > 2:
        candidate = system_messages + turns
        if token_count(tokenizer, candidate, enable_thinking) <= max_context_tokens:
            return candidate
        turns = turns[2:]
    return system_messages + turns


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[str, str, str]:
    prompt_text = apply_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    generation_kwargs: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    thinking, answer = split_thinking_and_answer(raw_text)
    return raw_text, thinking, answer


def run_interactive(model: Any, tokenizer: Any, args: argparse.Namespace) -> None:
    history: list[dict[str, str]] = []
    if args.system:
        history.append({"role": "system", "content": args.system})
    print("输入空行退出。")
    while True:
        prompt = input("\n用户> ").strip()
        if not prompt:
            break
        messages = history + [{"role": "user", "content": prompt}]
        messages = trim_history(tokenizer, messages, args.max_context_tokens, args.enable_thinking)
        _, thinking, answer = generate_response(model, tokenizer, messages, args)
        if thinking:
            print(f"\n思考>\n{thinking}")
        print(f"\n助手>\n{answer}")
        history.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])


def main() -> None:
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = default_device()
    model_kwargs: dict[str, Any] = {
        "dtype": resolve_dtype(args.torch_dtype, device),
        "trust_remote_code": args.trust_remote_code,
    }
    if use_device_map_auto():
        model_kwargs["device_map"] = "auto"
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, **model_kwargs)
    if not use_device_map_auto():
        model.to(device)
    model.eval()
    if not args.do_sample:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None

    if args.interactive:
        run_interactive(model, tokenizer, args)
        return

    if args.messages_file:
        messages = load_messages_from_file(args.messages_file)
    elif args.prompt:
        messages = messages_from_prompt(args.prompt, args.system)
    else:
        raise SystemExit("请传入 --prompt、--messages_file 或 --interactive。")

    _, thinking, answer = generate_response(model, tokenizer, messages, args)
    if thinking:
        print(f"思考：\n{thinking}\n")
    print(answer)


if __name__ == "__main__":
    main()
