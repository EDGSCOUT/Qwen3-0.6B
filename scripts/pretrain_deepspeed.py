from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler, set_seed
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from common import read_jsonl
from train_deepspeed import (
    batch_to_device,
    compute_total_steps,
    create_dataloader,
    create_optimizer,
    evaluate,
    get_rank,
    get_world_size,
    has_length,
    is_main_process,
    load_deepspeed_checkpoint,
    model_dtype,
    prepare_deepspeed_config,
    print_rank0,
    reduce_mean,
    resume_epoch_and_batch,
    save_deepspeed_checkpoint,
    save_hf_model,
)


class TextPretrainDataset(Dataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int,
        text_field: str,
        line_by_line: bool,
        append_eos: bool,
        drop_last_block: bool,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.examples = self.build_examples(
            path=self.path,
            text_field=text_field,
            line_by_line=line_by_line,
            append_eos=append_eos,
            drop_last_block=drop_last_block,
        )
        if not self.examples:
            raise ValueError(f"{self.path} 中没有可用于预训练的文本块")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        input_ids = self.examples[index]
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "labels": list(input_ids),
        }

    def build_examples(
        self,
        path: Path,
        text_field: str,
        line_by_line: bool,
        append_eos: bool,
        drop_last_block: bool,
    ) -> list[list[int]]:
        documents = self.read_documents(path, text_field, line_by_line)
        all_token_ids: list[int] = []
        for document in documents:
            document = document.strip()
            if not document:
                continue
            token_ids = self.tokenizer(document, add_special_tokens=False)["input_ids"]
            if append_eos and self.tokenizer.eos_token_id is not None:
                token_ids.append(self.tokenizer.eos_token_id)
            all_token_ids.extend(token_ids)

        examples: list[list[int]] = []
        for start in range(0, len(all_token_ids), self.block_size):
            block = all_token_ids[start : start + self.block_size]
            if len(block) < 2:
                continue
            if drop_last_block and len(block) < self.block_size:
                break
            examples.append(block)
        return examples

    def read_documents(self, path: Path, text_field: str, line_by_line: bool) -> list[str]:
        if not path.exists():
            raise FileNotFoundError(f"找不到预训练数据文件：{path}")
        if path.suffix == ".jsonl":
            records = read_jsonl(path)
            documents = []
            for idx, record in enumerate(records, start=1):
                value = record.get(text_field, record.get("content", record.get("text")))
                if value is None:
                    raise ValueError(f"{path}:{idx} 缺少文本字段 {text_field!r}")
                documents.append(str(value))
            return documents

        text = path.read_text(encoding="utf-8")
        if line_by_line:
            return [line.strip() for line in text.splitlines() if line.strip()]
        return [text]


class StreamingTextPretrainDataset(IterableDataset):
    def __init__(
        self,
        path: str | Path,
        tokenizer: PreTrainedTokenizerBase,
        block_size: int,
        text_field: str,
        line_by_line: bool,
        append_eos: bool,
        drop_last_block: bool,
    ) -> None:
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.text_field = text_field
        self.line_by_line = line_by_line
        self.append_eos = append_eos
        self.drop_last_block = drop_last_block
        if not self.path.exists():
            raise FileNotFoundError(f"找不到预训练数据文件：{self.path}")

    def __iter__(self):
        worker = get_worker_info()
        worker_id = worker.id if worker is not None else 0
        num_workers = worker.num_workers if worker is not None else 1
        rank = get_rank()
        world_size = get_world_size()
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers

        token_buffer: list[int] = []
        for doc_idx, document in self.iter_documents():
            if doc_idx % num_shards != shard_id:
                continue
            document = document.strip()
            if not document:
                continue
            token_ids = self.tokenizer(document, add_special_tokens=False)["input_ids"]
            if self.append_eos and self.tokenizer.eos_token_id is not None:
                token_ids.append(self.tokenizer.eos_token_id)
            token_buffer.extend(token_ids)

            while len(token_buffer) >= self.block_size:
                block = token_buffer[: self.block_size]
                token_buffer = token_buffer[self.block_size :]
                yield {
                    "input_ids": block,
                    "attention_mask": [1] * len(block),
                    "labels": list(block),
                }

        if not self.drop_last_block and len(token_buffer) >= 2:
            yield {
                "input_ids": token_buffer,
                "attention_mask": [1] * len(token_buffer),
                "labels": list(token_buffer),
            }

    def iter_documents(self):
        if self.path.suffix == ".jsonl":
            with self.path.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    value = record.get(self.text_field, record.get("content", record.get("text")))
                    if value is None:
                        raise ValueError(f"{self.path}:{idx + 1} 缺少文本字段 {self.text_field!r}")
                    yield idx, str(value)
            return

        with self.path.open("r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                if self.line_by_line:
                    line = line.strip()
                if line:
                    yield idx, line


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用原生 DeepSpeed 做因果语言模型预训练或继续预训练。")
    parser.add_argument("--model_name_or_path", default="models/Qwen3-0.6B", help="本地模型路径或 Hugging Face 模型 ID。")
    parser.add_argument("--train_file", default="data/pretrain.txt", help="预训练训练文本文件，支持 txt 或 jsonl。")
    parser.add_argument("--validation_file", default="data/pretrain_valid.txt", help="可选的预训练验证文本文件。")
    parser.add_argument("--output_dir", default="outputs/qwen3-0.6b-pretrain-ds", help="训练输出目录。")
    parser.add_argument("--deepspeed_config", default="configs/ds_zero2.json", help="DeepSpeed 配置文件。")
    parser.add_argument("--block_size", type=int, default=2048, help="文本分词后切块长度。")
    parser.add_argument("--text_field", default="text", help="JSONL 数据中的文本字段名。")
    parser.add_argument("--line_by_line", action="store_true", help="将 txt 文件的每个非空行作为独立文档读取。")
    parser.add_argument("--streaming", action="store_true", help="使用流式数据集，边读边分词，适合大文件；需要显式设置 --max_steps。")
    parser.add_argument("--append_eos", action=argparse.BooleanOptionalAction, default=True, help="是否在每段文档后追加 EOS。")
    parser.add_argument("--drop_last_block", action=argparse.BooleanOptionalAction, default=False, help="是否丢弃最后一个不足 block_size 的文本块。")
    parser.add_argument("--from_scratch", action="store_true", help="只读取模型配置并随机初始化参数。")
    parser.add_argument("--num_train_epochs", type=int, default=3, help="训练轮数。")
    parser.add_argument("--max_steps", type=int, default=-1, help="最大训练步数；大于 0 时优先于训练轮数。")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="每张 GPU 的训练 micro batch size。")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=1, help="每张 GPU 的验证 micro batch size。")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="梯度累积步数。")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="学习率。")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="权重衰减。")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="AdamW beta1。")
    parser.add_argument("--adam_beta2", type=float, default=0.95, help="AdamW beta2。")
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="AdamW epsilon。")
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="梯度裁剪阈值。")
    parser.add_argument("--warmup_ratio", type=float, default=0.03, help="warmup 占总训练步数的比例。")
    parser.add_argument("--warmup_steps", type=int, default=0, help="指定 warmup 步数；大于 0 时优先于 warmup_ratio。")
    parser.add_argument("--lr_scheduler_type", default="cosine", choices=["linear", "cosine", "constant"], help="学习率调度器类型。")
    parser.add_argument("--logging_steps", type=int, default=10, help="日志打印间隔。")
    parser.add_argument("--eval_steps", type=int, default=100, help="验证间隔。")
    parser.add_argument("--save_steps", type=int, default=100, help="DeepSpeed checkpoint 保存间隔。")
    parser.add_argument("--save_total_limit", type=int, default=2, help="最多保留的 DeepSpeed checkpoint 数量。")
    parser.add_argument("--num_workers", type=int, default=2, help="DataLoader worker 数量。")
    parser.add_argument("--seed", type=int, default=42, help="随机种子。")
    parser.add_argument("--local_rank", type=int, default=-1, help="DeepSpeed 启动器传入的本地进程编号。")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="启用梯度检查点以节省显存。")
    parser.add_argument("--trust_remote_code", action="store_true", help="是否允许加载远程自定义模型代码。")
    parser.add_argument("--resume_from_checkpoint", default=None, help="要恢复的 DeepSpeed checkpoint 路径。")

    precision = parser.add_mutually_exclusive_group()
    precision.add_argument("--bf16", action="store_true")
    precision.add_argument("--fp16", action="store_true")
    return parser.parse_args()


def load_model(args: argparse.Namespace) -> Any:
    if args.from_scratch:
        config = AutoConfig.from_pretrained(
            args.model_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        model = AutoModelForCausalLM.from_config(config, trust_remote_code=args.trust_remote_code)
        if args.bf16:
            model.to(torch.bfloat16)
        elif args.fp16:
            model.to(torch.float16)
        return model

    return AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=model_dtype(args),
        trust_remote_code=args.trust_remote_code,
    )


def main() -> None:
    args = parse_args()
    try:
        import deepspeed
    except ImportError as exc:
        raise SystemExit("请先安装 deepspeed：pip install deepspeed") from exc

    if not torch.cuda.is_available():
        raise SystemExit("原生 DeepSpeed 预训练需要 CUDA GPU。macOS 仅适合运行推理。")

    set_seed(args.seed)
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    torch.cuda.set_device(local_rank)

    if is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_max_length = args.block_size

    model = load_model(args)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    dataset_cls = StreamingTextPretrainDataset if args.streaming else TextPretrainDataset
    train_dataset = dataset_cls(
        args.train_file,
        tokenizer,
        args.block_size,
        args.text_field,
        args.line_by_line,
        args.append_eos,
        args.drop_last_block,
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
        eval_dataset = TextPretrainDataset(
            args.validation_file,
            tokenizer,
            args.block_size,
            args.text_field,
            args.line_by_line,
            args.append_eos,
            args.drop_last_block,
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
    global_step = step_offset + int(model_engine.global_steps)
    start_epoch = 0
    start_batch = 0
    streaming_skip_batches = 0
    if has_length(train_loader):
        start_epoch, start_batch = resume_epoch_and_batch(
            global_step,
            args.gradient_accumulation_steps,
            len(train_loader),
        )
    else:
        streaming_skip_batches = global_step * args.gradient_accumulation_steps

    mode_name = "从零预训练" if args.from_scratch else "继续预训练"
    train_size = len(train_dataset) if has_length(train_dataset) else "streaming"
    print_rank0(
        f"{mode_name}配置："
        f"world_size={get_world_size()}, total_steps={total_train_steps}, warmup_steps={warmup_steps}, "
        f"train_blocks={train_size}, block_size={args.block_size}, "
        f"resume_epoch={start_epoch}, resume_batch={start_batch}, streaming_skip_batches={streaming_skip_batches}"
    )

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
            if streaming_skip_batches > 0:
                streaming_skip_batches -= 1
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
