from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from common import IGNORE_INDEX, apply_chat_template_text, normalize_messages, read_jsonl, split_prompt_and_reference


@dataclass
class ModelDataArguments:
    model_name_or_path: str = field(
        default="models/Qwen3-0.6B",
        metadata={"help": "本地模型路径或 Hugging Face 模型 ID。"},
    )
    train_file: str = field(default="data/train.jsonl", metadata={"help": "训练 JSONL 文件。"})
    validation_file: str | None = field(
        default="data/valid.jsonl",
        metadata={"help": "可选的验证 JSONL 文件。"},
    )
    model_max_length: int = field(default=2048, metadata={"help": "分词后的最大长度。"})
    enable_thinking: bool = field(
        default=False,
        metadata={"help": "在聊天模板中启用 Qwen3 thinking 模式。"},
    )
    response_only_loss: bool = field(
        default=True,
        metadata={"help": "屏蔽 prompt token，只训练最后一条 assistant 回复。"},
    )
    skip_empty_labels: bool = field(
        default=True,
        metadata={"help": "过滤截断后没有任何可训练 label 的样本。"},
    )
    trust_remote_code: bool = field(default=False, metadata={"help": "是否向 Hugging Face 加载接口传入 trust_remote_code。"})
    resume_from_checkpoint: str | None = field(default=None, metadata={"help": "checkpoint 路径。"})


class JsonlSFTDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        model_max_length: int,
        enable_thinking: bool,
        response_only_loss: bool,
        skip_empty_labels: bool = True,
    ) -> None:
        self.path = Path(path)
        self.records = read_jsonl(self.path)
        self.tokenizer = tokenizer
        self.model_max_length = model_max_length
        self.enable_thinking = enable_thinking
        self.response_only_loss = response_only_loss
        self.indices = list(range(len(self.records)))
        if not self.records:
            raise ValueError(f"{self.path} 中没有样本")
        if skip_empty_labels:
            self.indices = [
                idx
                for idx in self.indices
                if any(label != IGNORE_INDEX for label in self.encode_record(self.records[idx])["labels"][1:])
            ]
            if not self.indices:
                raise ValueError(f"{self.path} 中没有包含可训练 label 的样本，请检查长度截断或数据格式")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        return self.encode_record(self.records[self.indices[index]])

    def encode_record(self, record: dict[str, Any]) -> dict[str, list[int]]:
        messages = normalize_messages(record)
        full_text = apply_chat_template_text(
            self.tokenizer,
            messages,
            add_generation_prompt=False,
            enable_thinking=self.enable_thinking,
        )
        tokenized = self.tokenizer(
            full_text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.model_max_length,
        )
        labels = list(tokenized["input_ids"])

        if self.response_only_loss:
            prompt_messages, _ = split_prompt_and_reference(messages)
            if len(prompt_messages) < len(messages):
                prompt_text = apply_chat_template_text(
                    self.tokenizer,
                    prompt_messages,
                    add_generation_prompt=True,
                    enable_thinking=self.enable_thinking,
                )
                prompt_ids = self.tokenizer(
                    prompt_text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=self.model_max_length,
                )["input_ids"]
                prompt_len = min(len(prompt_ids), len(labels))
                labels[:prompt_len] = [IGNORE_INDEX] * prompt_len

        return {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": labels,
        }


@dataclass
class DataCollatorForCausalLM:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int | None = 8
    label_pad_token_id: int = IGNORE_INDEX

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        model_inputs = [
            {"input_ids": feature["input_ids"], "attention_mask": feature["attention_mask"]}
            for feature in features
        ]
        batch = self.tokenizer.pad(
            model_inputs,
            padding=True,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        max_length = batch["input_ids"].shape[1]
        labels = []
        for feature in features:
            label = list(feature["labels"])
            pad_length = max_length - len(label)
            padding = [self.label_pad_token_id] * pad_length
            if self.tokenizer.padding_side == "left":
                labels.append(padding + label)
            else:
                labels.append(label + padding)
        batch["labels"] = torch.tensor(labels, dtype=torch.long)
        return batch


def resolve_train_dtype(training_args: TrainingArguments) -> torch.dtype | str:
    if training_args.bf16:
        return torch.bfloat16
    if training_args.fp16:
        return torch.float16
    return "auto"


def main() -> None:
    parser = HfArgumentParser((ModelDataArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
        trust_remote_code=model_args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = model_args.model_max_length

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=resolve_train_dtype(training_args),
        trust_remote_code=model_args.trust_remote_code,
    )
    if training_args.gradient_checkpointing:
        model.config.use_cache = False

    train_dataset = JsonlSFTDataset(
        model_args.train_file,
        tokenizer,
        model_args.model_max_length,
        model_args.enable_thinking,
        model_args.response_only_loss,
        model_args.skip_empty_labels,
    )
    eval_dataset = None
    if model_args.validation_file and Path(model_args.validation_file).exists():
        eval_dataset = JsonlSFTDataset(
            model_args.validation_file,
            tokenizer,
            model_args.model_max_length,
            model_args.enable_thinking,
            model_args.response_only_loss,
            model_args.skip_empty_labels,
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorForCausalLM(tokenizer),
    )
    trainer.train(resume_from_checkpoint=model_args.resume_from_checkpoint)
    trainer.save_model(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == "__main__":
    main()
