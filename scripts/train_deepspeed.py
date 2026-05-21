from __future__ import annotations

import argparse
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler, IterableDataset, RandomSampler, SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from common import IGNORE_INDEX, apply_chat_template_text, normalize_messages, read_jsonl, split_prompt_and_reference


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


class DataCollatorForCausalLM:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pad_to_multiple_of: int | None = 8,
        label_pad_token_id: int = IGNORE_INDEX,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.label_pad_token_id = label_pad_token_id

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用原生 DeepSpeed 训练循环对 Qwen3 做 SFT。")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-0.6B")
    parser.add_argument("--train_file", default="data/train.jsonl")
    parser.add_argument("--validation_file", default="data/valid.jsonl")
    parser.add_argument("--output_dir", default="outputs/qwen3-0.6b-sft-ds")
    parser.add_argument("--deepspeed_config", default="configs/ds_zero2.json")
    parser.add_argument("--model_max_length", type=int, default=2048)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler_type", default="cosine", choices=["linear", "cosine", "constant"])
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--enable_thinking", "--enable-thinking", dest="enable_thinking", action="store_true")
    parser.add_argument("--response_only_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_empty_labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)

    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_dist_initialized() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_dist_initialized() else int(os.environ.get("WORLD_SIZE", "1"))


def is_main_process() -> bool:
    return get_rank() == 0


def print_rank0(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def model_dtype(args: argparse.Namespace) -> torch.dtype | str:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return "auto"


def prepare_deepspeed_config(
    args: argparse.Namespace,
    total_train_steps: int,
    world_size: int,
) -> dict[str, Any]:
    with Path(args.deepspeed_config).open("r", encoding="utf-8") as f:
        config = json.load(f)

    micro_batch = args.per_device_train_batch_size
    grad_accum = args.gradient_accumulation_steps
    config["train_micro_batch_size_per_gpu"] = micro_batch
    config["gradient_accumulation_steps"] = grad_accum
    config["train_batch_size"] = micro_batch * grad_accum * world_size
    config["gradient_clipping"] = args.max_grad_norm

    config.setdefault("bf16", {})["enabled"] = bool(args.bf16)
    config.setdefault("fp16", {})["enabled"] = bool(args.fp16)

    # 优化器和调度器在 Python 中显式创建，避免依赖 Trainer 对 DeepSpeed 的 auto 配置解析。
    config.pop("optimizer", None)
    config.pop("scheduler", None)

    zero_config = config.get("zero_optimization", {})
    if zero_config.get("reduce_bucket_size") == "auto":
        zero_config["reduce_bucket_size"] = 50_000_000
    if zero_config.get("stage3_prefetch_bucket_size") == "auto":
        zero_config["stage3_prefetch_bucket_size"] = 50_000_000
    if zero_config.get("stage3_param_persistence_threshold") == "auto":
        zero_config["stage3_param_persistence_threshold"] = 100_000

    unresolved = find_auto_values(config)
    if unresolved:
        joined = ", ".join(unresolved)
        raise ValueError(f"DeepSpeed 配置中仍包含未解析的 auto 值：{joined}")

    if total_train_steps <= 0:
        raise ValueError("total_train_steps 必须为正数")
    return config


def find_auto_values(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        paths: list[str] = []
        for key, child in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            paths.extend(find_auto_values(child, child_prefix))
        return paths
    if isinstance(value, list):
        paths = []
        for idx, child in enumerate(value):
            paths.extend(find_auto_values(child, f"{prefix}[{idx}]"))
        return paths
    return [prefix] if value == "auto" else []


def create_optimizer(model: torch.nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    no_decay = ("bias", "layer_norm.weight", "norm.weight", "ln_f.weight")
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(key in name.lower() for key in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer_groups = [
        {"params": decay_params, "weight_decay": args.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        optimizer_groups,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epsilon,
    )


def create_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> tuple[DataLoader, DistributedSampler | None]:
    if isinstance(dataset, IterableDataset):
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=DataCollatorForCausalLM(tokenizer),
            num_workers=num_workers,
            pin_memory=True,
        )
        return dataloader, None

    sampler: DistributedSampler | RandomSampler | SequentialSampler
    distributed_sampler = None
    if is_dist_initialized():
        distributed_sampler = DistributedSampler(dataset, shuffle=train, drop_last=False)
        sampler = distributed_sampler
    elif train:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        collate_fn=DataCollatorForCausalLM(tokenizer),
        num_workers=num_workers,
        pin_memory=True,
    )
    return dataloader, distributed_sampler


def has_length(value: Any) -> bool:
    try:
        len(value)
    except TypeError:
        return False
    return True


def compute_total_steps(
    train_loader: DataLoader,
    gradient_accumulation_steps: int,
    num_train_epochs: int,
    max_steps: int,
) -> int:
    if not has_length(train_loader):
        if max_steps <= 0:
            raise ValueError("streaming 数据无法自动推断训练步数，请显式设置 --max_steps")
        return max_steps
    steps_per_epoch = max(1, math.ceil(len(train_loader) / gradient_accumulation_steps))
    if max_steps > 0:
        return max_steps
    return steps_per_epoch * num_train_epochs


def batch_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def resume_epoch_and_batch(global_step: int, gradient_accumulation_steps: int, batches_per_epoch: int) -> tuple[int, int]:
    if global_step <= 0 or batches_per_epoch <= 0:
        return 0, 0
    consumed_batches = global_step * gradient_accumulation_steps
    return consumed_batches // batches_per_epoch, consumed_batches % batches_per_epoch


def reduce_mean(value: float, device: torch.device) -> float:
    if not is_dist_initialized():
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float32)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= get_world_size()
    return tensor.item()


@torch.no_grad()
def evaluate(model_engine: Any, eval_loader: DataLoader) -> dict[str, float]:
    model_engine.eval()
    total_loss = 0.0
    total_tokens = 0
    for batch in eval_loader:
        batch = batch_to_device(batch, model_engine.device)
        outputs = model_engine(**batch)
        valid_tokens = (batch["labels"][..., 1:] != IGNORE_INDEX).sum().item()
        if valid_tokens == 0:
            continue
        total_loss += outputs.loss.detach().float().item() * valid_tokens
        total_tokens += valid_tokens

    stats = torch.tensor([total_loss, total_tokens], device=model_engine.device, dtype=torch.float32)
    if is_dist_initialized():
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    mean_loss = (stats[0] / stats[1].clamp_min(1)).item()
    perplexity = float(math.exp(min(mean_loss, 20.0)))
    model_engine.train()
    return {"eval_loss": mean_loss, "eval_ppl": perplexity, "eval_tokens": int(stats[1].item())}


def checkpoint_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / "ds_checkpoints"


def rotate_checkpoints(output_dir: str | Path, save_total_limit: int) -> None:
    if save_total_limit <= 0 or not is_main_process():
        return
    root = checkpoint_root(output_dir)
    if not root.exists():
        return
    checkpoints = sorted(
        [path for path in root.iterdir() if path.is_dir() and path.name.startswith("global_step")],
        key=lambda path: int(path.name.replace("global_step", "")),
    )
    for old_checkpoint in checkpoints[:-save_total_limit]:
        shutil.rmtree(old_checkpoint, ignore_errors=True)


def save_deepspeed_checkpoint(model_engine: Any, args: argparse.Namespace, global_step: int) -> None:
    root = checkpoint_root(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    tag = f"global_step{global_step}"
    model_engine.save_checkpoint(
        str(root),
        tag=tag,
        client_state={"global_step": global_step},
    )
    rotate_checkpoints(args.output_dir, args.save_total_limit)


def load_deepspeed_checkpoint(model_engine: Any, checkpoint: str) -> int:
    path = Path(checkpoint)
    if path.name.startswith("global_step"):
        load_dir = str(path.parent)
        tag = path.name
    else:
        load_dir = str(path)
        tag = None
    loaded_path, client_state = model_engine.load_checkpoint(load_dir, tag=tag)
    if loaded_path is None:
        raise RuntimeError(f"无法从 {checkpoint} 加载 DeepSpeed checkpoint")
    return int((client_state or {}).get("global_step", 0))


def save_hf_model(model_engine: Any, tokenizer: Any, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # ZeRO-3 聚合参数时所有训练进程都必须参与。
    saved = model_engine.save_16bit_model(str(output_path), save_filename="pytorch_model.bin")
    if is_main_process():
        tokenizer.save_pretrained(output_path)
        model_engine.module.config.save_pretrained(output_path)
        generation_config = getattr(model_engine.module, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(output_path)
        if not saved:
            print_rank0(
                "DeepSpeed 没有写出 16-bit 模型。"
                "如果使用 ZeRO-3，请设置 stage3_gather_16bit_weights_on_model_save=true。"
            )


def main() -> None:
    args = parse_args()
    try:
        import deepspeed
    except ImportError as exc:
        raise SystemExit("请先安装 deepspeed：pip install deepspeed") from exc

    if not torch.cuda.is_available():
        raise SystemExit("原生 DeepSpeed 训练需要 CUDA GPU。macOS 仅适合运行推理。")

    set_seed(args.seed)
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)

    output_dir = Path(args.output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.model_max_length

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=model_dtype(args),
        trust_remote_code=args.trust_remote_code,
    )
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    train_dataset = JsonlSFTDataset(
        args.train_file,
        tokenizer,
        args.model_max_length,
        args.enable_thinking,
        args.response_only_loss,
        args.skip_empty_labels,
    )
    train_loader, train_sampler = create_dataloader(
        train_dataset,
        tokenizer,
        args.per_device_train_batch_size,
        args.num_workers,
        train=True,
    )

    eval_loader = None
    if args.validation_file and Path(args.validation_file).exists():
        eval_dataset = JsonlSFTDataset(
            args.validation_file,
            tokenizer,
            args.model_max_length,
            args.enable_thinking,
            args.response_only_loss,
            args.skip_empty_labels,
        )
        eval_loader, _ = create_dataloader(
            eval_dataset,
            tokenizer,
            args.per_device_eval_batch_size,
            args.num_workers,
            train=False,
        )

    total_train_steps = compute_total_steps(
        train_loader,
        args.gradient_accumulation_steps,
        args.num_train_epochs,
        args.max_steps,
    )
    warmup_steps = args.warmup_steps or int(total_train_steps * args.warmup_ratio)
    ds_config = prepare_deepspeed_config(args, total_train_steps, get_world_size())

    optimizer = create_optimizer(model, args)
    scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_train_steps,
    )

    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        config=ds_config,
    )

    resumed_step = 0
    if args.resume_from_checkpoint:
        resumed_step = load_deepspeed_checkpoint(model_engine, args.resume_from_checkpoint)
        print_rank0(f"已从 {args.resume_from_checkpoint} 恢复，global_step={resumed_step}")
    step_offset = max(0, resumed_step - int(model_engine.global_steps))

    start_epoch = 0
    start_batch = 0
    if has_length(train_loader):
        start_epoch, start_batch = resume_epoch_and_batch(
            step_offset + int(model_engine.global_steps),
            args.gradient_accumulation_steps,
            len(train_loader),
        )

    print_rank0(
        "训练配置："
        f"world_size={get_world_size()}, total_steps={total_train_steps}, warmup_steps={warmup_steps}, "
        f"train_samples={len(train_dataset)}, resume_epoch={start_epoch}, resume_batch={start_batch}"
    )

    global_step = step_offset + int(model_engine.global_steps)
    loss_sum = 0.0
    loss_count = 0
    stop_training = False

    model_engine.train()
    for epoch in range(start_epoch, args.num_train_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue

            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break

            batch = batch_to_device(batch, model_engine.device)
            outputs = model_engine(**batch)
            loss = outputs.loss
            model_engine.backward(loss)

            loss_sum += loss.detach().float().item()
            loss_count += 1
            step_before = model_engine.global_steps
            model_engine.step()

            if model_engine.global_steps <= step_before:
                continue

            global_step = step_offset + int(model_engine.global_steps)

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                local_loss = loss_sum / max(1, loss_count)
                mean_loss = reduce_mean(local_loss, model_engine.device)
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else args.learning_rate
                print_rank0(
                    f"step={global_step}/{total_train_steps} "
                    f"epoch={epoch + 1} loss={mean_loss:.4f} lr={current_lr:.6g}"
                )
                loss_sum = 0.0
                loss_count = 0

            if eval_loader is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                metrics = evaluate(model_engine, eval_loader)
                print_rank0(
                    f"eval step={global_step} "
                    f"loss={metrics['eval_loss']:.4f} ppl={metrics['eval_ppl']:.4f}"
                )

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                print_rank0(f"正在保存第 {global_step} 步的 DeepSpeed checkpoint")
                save_deepspeed_checkpoint(model_engine, args, global_step)

            if global_step >= total_train_steps:
                stop_training = True
                break

        if stop_training:
            break

    print_rank0("正在保存最终 Hugging Face 格式模型...")
    save_hf_model(model_engine, tokenizer, args.output_dir)
    print_rank0(f"完成。模型已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
