from __future__ import annotations

import argparse
from typing import Any

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import is_accelerate_available

from common import (
    apply_chat_template_text,
    normalize_for_match,
    normalize_messages,
    read_jsonl,
    split_prompt_and_reference,
    split_thinking_and_answer,
    write_json,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="在 JSONL 文件上评测 Qwen3。")
    parser.add_argument("--model_name_or_path", default="outputs/qwen3-0.6b-sft")
    parser.add_argument("--eval_file", default="data/test.jsonl")
    parser.add_argument("--output_file", default="outputs/eval_predictions.jsonl")
    parser.add_argument("--metrics_file", default="outputs/eval_metrics.json")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--do_sample", action=argparse.BooleanOptionalAction, default=False)
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


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_device_map_auto() -> bool:
    return is_accelerate_available() and torch.cuda.is_available()


def generate_answer(
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
        "do_sample": args.do_sample,
    }
    if args.do_sample:
        generation_kwargs.update(
            {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            }
        )

    with torch.no_grad():
        output_ids = model.generate(**inputs, **generation_kwargs)
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1] :]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    thinking, answer = split_thinking_and_answer(raw_text)
    return raw_text, thinking, answer


def compute_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [item for item in predictions if item.get("reference") is not None]
    if not scored:
        return {"total": len(predictions), "with_reference": 0}

    exact = 0
    contains = 0
    for item in scored:
        pred = normalize_for_match(item["prediction"])
        ref = normalize_for_match(item["reference"])
        exact += int(pred == ref)
        contains += int(bool(ref) and ref in pred)
    count = len(scored)
    return {
        "total": len(predictions),
        "with_reference": count,
        "exact_match": exact / count,
        "contains_reference": contains / count,
    }


def main() -> None:
    args = parse_args()
    records = read_jsonl(args.eval_file)

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

    predictions = []
    for idx, record in enumerate(tqdm(records, desc="评测中")):
        messages = normalize_messages(record)
        prompt_messages, reference = split_prompt_and_reference(messages)
        raw_text, thinking, answer = generate_answer(model, tokenizer, prompt_messages, args)
        predictions.append(
            {
                "id": record.get("id", idx),
                "prompt_messages": prompt_messages,
                "reference": reference,
                "raw_prediction": raw_text,
                "thinking": thinking,
                "prediction": answer,
            }
        )

    metrics = compute_metrics(predictions)
    write_jsonl(args.output_file, predictions)
    write_json(args.metrics_file, metrics)
    print(metrics)


if __name__ == "__main__":
    main()
