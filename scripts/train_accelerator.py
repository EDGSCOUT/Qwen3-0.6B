from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from common import IGNORE_INDEX
from train_deepspeed import (
    DataCollatorForCausalLM,
    JsonlSFTDataset,
    compute_total_steps,
    create_optimizer,
    has_length,
    model_dtype,
    resume_epoch_and_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 Hugging Face Accelerate 对 Qwen3 做 SFT。")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-0.6B", help="本地模型路径或 Hugging Face 模型 ID。")
    parser.add_argument("--train_file", default="data/train.jsonl", help="训练 JSONL 文件。")
    parser.add_argument("--validation_file", default="data/valid.jsonl", help="可选的验证 JSONL 文件。")
    parser.add_argument("--output_dir", default="outputs/qwen3-0.6b-sft-accelerate", help="训练输出目录。")
    parser.add_argument("--model_max_length", type=int, default=2048, help="分词后的最大长度。")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数。")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大更新步数；大于 0 时优先于训练轮数。")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每个进程/设备的训练 micro batch size。")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="每个进程/设备的验证 batch size。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数。")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率。")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="权重衰减。")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1。")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="AdamW beta2。")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon。")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值；小于等于 0 表示不裁剪。")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="warmup 占总训练步数的比例。")
    parser.add_argument("--warmup_steps", type=int, default=0, help="指定 warmup 步数；大于 0 时优先于 warmup_ratio。")
    parser.add_argument("--lr_scheduler_type", default="cosine", choices=["linear", "cosine", "constant"], help="学习率调度器类型。")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志打印间隔。")
    parser.add_argument("--eval_steps", type=int, default=100, help="验证间隔。")
    parser.add_argument("--save_steps", type=int, default=100, help="Accelerate checkpoint 保存间隔。")
    parser.add_argument("--save_total_limit", type=int, default=2, help="最多保留的 Accelerate checkpoint 数量。")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker 数量。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点以节省显存。")
    parser.add_argument("--enable_thinking", "--enable-thinking", dest="enable_thinking", action="store_true")
    parser.add_argument("--response_only_loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip_empty_labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--trust_remote_code", action="store_true", help="是否允许加载远程自定义模型代码。")
    parser.add_argument("--resume_from_checkpoint", default=None, help="要恢复的 Accelerate checkpoint 路径。")

    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def mixed_precision(args: argparse.Namespace) -> str:
    if args.bf16:
        return "bf16"
    if args.fp16:
        return "fp16"
    return "no"


def create_dataloader(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    batch_size: int,
    num_workers: int,
    train: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train and not isinstance(dataset, IterableDataset),
        collate_fn=DataCollatorForCausalLM(tokenizer),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def checkpoint_root(output_dir: str | Path) -> Path:
    return Path(output_dir) / "accelerate_checkpoints"


def checkpoint_state_path(checkpoint_dir: str | Path) -> Path:
    return Path(checkpoint_dir) / "trainer_state.json"


def rotate_checkpoints(output_dir: str | Path, save_total_limit: int) -> None:
    if save_total_limit <= 0:
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


def write_trainer_state(checkpoint_dir: str | Path, global_step: int, epoch: int, batch_idx: int) -> None:
    state = {
        "global_step": global_step,
        "epoch": epoch,
        "batch_idx": batch_idx,
    }
    checkpoint_state_path(checkpoint_dir).write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_global_step_from_checkpoint(checkpoint: str | Path) -> int:
    path = Path(checkpoint)
    state_path = checkpoint_state_path(path)
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        return int(state.get("global_step", 0))
    if path.name.startswith("global_step"):
        return int(path.name.replace("global_step", ""))
    return 0


def save_accelerate_checkpoint(
    accelerator: Any,
    args: argparse.Namespace,
    global_step: int,
    epoch: int,
    batch_idx: int,
) -> None:
    checkpoint_dir = checkpoint_root(args.output_dir) / f"global_step{global_step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(str(checkpoint_dir))
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        write_trainer_state(checkpoint_dir, global_step, epoch, batch_idx)
        rotate_checkpoints(args.output_dir, args.save_total_limit)
    accelerator.wait_for_everyone()


def load_accelerate_checkpoint(accelerator: Any, checkpoint: str | Path) -> int:
    accelerator.load_state(str(checkpoint))
    return read_global_step_from_checkpoint(checkpoint)


def mean_across_processes(accelerator: Any, value: float) -> float:
    tensor = torch.tensor(value, device=accelerator.device, dtype=torch.float32)
    gathered = accelerator.gather(tensor)
    return gathered.float().mean().item()


@torch.no_grad()
def evaluate(accelerator: Any, model: torch.nn.Module, eval_loader: DataLoader) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    for batch in eval_loader:
        outputs = model(**batch)
        valid_tokens = (batch["labels"][..., 1:] != IGNORE_INDEX).sum().item()
        if valid_tokens == 0:
            continue
        total_loss += outputs.loss.detach().float().item() * valid_tokens
        total_tokens += valid_tokens

    stats = torch.tensor([total_loss, total_tokens], device=accelerator.device, dtype=torch.float32)
    gathered = accelerator.gather(stats).reshape(-1, 2).sum(dim=0)
    mean_loss = (gathered[0] / gathered[1].clamp_min(1)).item()
    perplexity = float(math.exp(min(mean_loss, 20.0)))
    model.train()
    return {"eval_loss": mean_loss, "eval_ppl": perplexity, "eval_tokens": int(gathered[1].item())}


def save_hf_model(accelerator: Any, model: torch.nn.Module, tokenizer: Any, output_dir: str | Path) -> None:
    output_path = Path(output_dir)
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    state_dict = accelerator.get_state_dict(model)
    unwrapped_model.save_pretrained(
        output_path,
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
        state_dict=state_dict,
    )
    if accelerator.is_main_process:
        tokenizer.save_pretrained(output_path)
        generation_config = getattr(unwrapped_model, "generation_config", None)
        if generation_config is not None:
            generation_config.save_pretrained(output_path)
    accelerator.wait_for_everyone()


def set_dataloader_epoch(dataloader: DataLoader, epoch: int) -> None:
    if hasattr(dataloader, "set_epoch"):
        dataloader.set_epoch(epoch)


def main() -> None:
    args = parse_args()
    try:
        from accelerate import Accelerator
        from accelerate.utils import DataLoaderConfiguration
    except ImportError as exc:
        raise SystemExit("请先安装 accelerate：pip install accelerate") from exc

    dataloader_config = DataLoaderConfiguration(
        use_seedable_sampler=True,
        data_seed=args.seed,
        non_blocking=torch.cuda.is_available(),
    )
    accelerator = Accelerator(
        mixed_precision=mixed_precision(args),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=dataloader_config,
    )
    set_seed(args.seed)

    if accelerator.is_main_process:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()

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
    train_loader = create_dataloader(
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
        eval_loader = create_dataloader(
            eval_dataset,
            tokenizer,
            args.per_device_eval_batch_size,
            args.num_workers,
            train=False,
        )

    optimizer = create_optimizer(model, args)
    if eval_loader is None:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)
    else:
        model, optimizer, train_loader, eval_loader = accelerator.prepare(model, optimizer, train_loader, eval_loader)

    total_train_steps = compute_total_steps(
        train_loader,
        args.gradient_accumulation_steps,
        args.num_train_epochs,
        args.max_steps,
    )
    if total_train_steps <= 0:
        raise ValueError("total_train_steps 必须为正数")
    steps_per_epoch = (
        max(1, math.ceil(len(train_loader) / args.gradient_accumulation_steps))
        if has_length(train_loader)
        else total_train_steps
    )
    num_train_epochs = args.num_train_epochs
    if args.max_steps > 0:
        num_train_epochs = max(num_train_epochs, math.ceil(args.max_steps / steps_per_epoch))

    warmup_steps = args.warmup_steps or int(total_train_steps * args.warmup_ratio)
    scheduler_total_steps = total_train_steps * accelerator.num_processes
    scheduler_warmup_steps = warmup_steps * accelerator.num_processes
    lr_scheduler = get_scheduler(
        name=args.lr_scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=scheduler_warmup_steps,
        num_training_steps=scheduler_total_steps,
    )
    lr_scheduler = accelerator.prepare_scheduler(lr_scheduler)

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = load_accelerate_checkpoint(accelerator, args.resume_from_checkpoint)
        accelerator.print(f"已从 {args.resume_from_checkpoint} 恢复，global_step={global_step}")

    start_epoch = 0
    start_batch = 0
    if has_length(train_loader):
        start_epoch, start_batch = resume_epoch_and_batch(
            global_step,
            args.gradient_accumulation_steps,
            len(train_loader),
        )

    accelerator.print(
        "训练配置："
        f"world_size={accelerator.num_processes}, device={accelerator.device}, "
        f"mixed_precision={accelerator.mixed_precision}, total_steps={total_train_steps}, "
        f"warmup_steps={warmup_steps}, train_samples={len(train_dataset)}, "
        f"resume_epoch={start_epoch}, resume_batch={start_batch}"
    )

    loss_sum = 0.0
    loss_count = 0
    stop_training = False
    model.train()

    for epoch in range(start_epoch, num_train_epochs):
        set_dataloader_epoch(train_loader, epoch)

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < start_batch:
                continue
            if args.max_steps > 0 and global_step >= args.max_steps:
                stop_training = True
                break

            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                loss_sum += loss.detach().float().item()
                loss_count += 1

                accelerator.backward(loss)
                if accelerator.sync_gradients and args.max_grad_norm > 0:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if not accelerator.sync_gradients:
                continue
            if accelerator.optimizer_step_was_skipped:
                continue

            global_step += 1

            if args.logging_steps > 0 and global_step % args.logging_steps == 0:
                local_loss = loss_sum / max(1, loss_count)
                mean_loss = mean_across_processes(accelerator, local_loss)
                current_lr = lr_scheduler.get_last_lr()[0]
                accelerator.print(
                    f"step={global_step}/{total_train_steps} "
                    f"epoch={epoch + 1} loss={mean_loss:.4f} lr={current_lr:.6g}"
                )
                loss_sum = 0.0
                loss_count = 0

            if eval_loader is not None and args.eval_steps > 0 and global_step % args.eval_steps == 0:
                metrics = evaluate(accelerator, model, eval_loader)
                accelerator.print(
                    f"eval step={global_step} "
                    f"loss={metrics['eval_loss']:.4f} ppl={metrics['eval_ppl']:.4f}"
                )

            if args.save_steps > 0 and global_step % args.save_steps == 0:
                accelerator.print(f"正在保存第 {global_step} 步的 Accelerate checkpoint")
                save_accelerate_checkpoint(accelerator, args, global_step, epoch, batch_idx)

            if global_step >= total_train_steps:
                stop_training = True
                break

        if stop_training:
            break

    accelerator.print("正在保存最终 Hugging Face 格式模型...")
    save_hf_model(accelerator, model, tokenizer, args.output_dir)
    accelerator.print(f"完成。模型已保存到 {args.output_dir}")


if __name__ == "__main__":
    main()
