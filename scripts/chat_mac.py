from __future__ import annotations

import argparse
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import apply_chat_template_text, split_thinking_and_answer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="适用于 macOS 的 Qwen3 多轮对话脚本。")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-0.6B")
    parser.add_argument("--system", default="你是一个简洁、可靠的中文助手。")
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
        help="在 macOS 上使用 auto 时，MPS 默认 float16，CPU 默认 float32。",
    )
    return parser.parse_args()


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def resolve_dtype(name: str, device: torch.device) -> torch.dtype | str:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    if device.type in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


def load_model_and_tokenizer(args: argparse.Namespace) -> tuple[Any, Any, torch.device]:
    device = default_device()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=resolve_dtype(args.torch_dtype, device),
        trust_remote_code=args.trust_remote_code,
    )
    model.to(device)
    model.eval()
    if not args.do_sample:
        model.generation_config.do_sample = False
        model.generation_config.temperature = None
        model.generation_config.top_p = None
        model.generation_config.top_k = None
    return model, tokenizer, device


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


def generate_once(
    model: Any,
    tokenizer: Any,
    device: torch.device,
    messages: list[dict[str, str]],
    args: argparse.Namespace,
) -> tuple[str, str]:
    prompt_text = apply_chat_template_text(
        tokenizer,
        messages,
        add_generation_prompt=True,
        enable_thinking=args.enable_thinking,
    )
    inputs = tokenizer([prompt_text], return_tensors="pt")
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

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generation_kwargs)

    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    thinking, answer = split_thinking_and_answer(raw_text)
    return thinking, answer


def print_help() -> None:
    print("命令：/exit 退出，/reset 清空上下文，/history 查看轮数，/help 查看命令")


def main() -> None:
    args = parse_args()
    model, tokenizer, device = load_model_and_tokenizer(args)
    print(f"已在 {device} 上加载模型：{args.model_name_or_path}")
    print_help()

    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})

    while True:
        try:
            user_input = input("\n你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break

        if not user_input:
            continue
        if user_input in {"/exit", "/quit", "exit", "quit"}:
            print("已退出。")
            break
        if user_input == "/help":
            print_help()
            continue
        if user_input == "/reset":
            messages = [{"role": "system", "content": args.system}] if args.system else []
            print("上下文已清空。")
            continue
        if user_input == "/history":
            turns = sum(1 for msg in messages if msg["role"] == "user")
            print(f"当前保留 {turns} 轮用户输入。")
            continue

        messages.append({"role": "user", "content": user_input})
        messages = trim_history(tokenizer, messages, args.max_context_tokens, args.enable_thinking)

        thinking, answer = generate_once(model, tokenizer, device, messages, args)
        if thinking:
            print(f"\n思考>\n{thinking}")
        print(f"\n助手>\n{answer}")
        messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
