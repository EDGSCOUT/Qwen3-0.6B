# Qwen3-0.6B 本地训练、推理与评测工程

这是一个面向 Qwen3-0.6B 的本地训练、推理、评测示例工程。代码基于 `transformers`，训练提供三类脚本：

- `scripts/train.py`：Hugging Face `Trainer` + DeepSpeed 配置
- `scripts/train_deepspeed.py`：原生 DeepSpeed SFT 训练循环
- `scripts/pretrain_deepspeed.py`：原生 DeepSpeed 预训练/继续预训练循环

当前目录已经包含模型、样例数据、训练脚本、推理脚本、评测脚本、DeepSpeed 配置，以及 `transformers` / `deepspeed` 的源码跳转目录。

官方模型仓库：<https://huggingface.co/Qwen/Qwen3-0.6B>

## 目录结构

```text
.
├── configs/
│   ├── ds_zero2.json
│   └── ds_zero3_offload.json
├── data/
│   ├── train.jsonl
│   ├── valid.jsonl
│   ├── test.jsonl
│   ├── pretrain.txt
│   └── pretrain_valid.txt
├── models/
│   └── Qwen3-0.6B/
├── outputs/
├── scripts/
│   ├── chat_mac.py
│   ├── common.py
│   ├── download_model.py
│   ├── evaluate.py
│   ├── infer.py
│   ├── pretrain_deepspeed.py
│   ├── train.py
│   └── train_deepspeed.py
├── third_party/
│   ├── deepspeed-0.14.5/
│   └── transformers-4.56.2/
├── requirements.txt
└── README.md
```

主要路径说明：

- `models/Qwen3-0.6B`：本地 Qwen3-0.6B 模型目录，当前已下载权重
- `data/*.jsonl`：SFT 样例训练、验证、测试数据
- `data/pretrain*.txt`：预训练/继续预训练样例纯文本数据
- `outputs/`：训练输出、评测输出、checkpoint 输出目录
- `configs/ds_zero2.json`：常用 ZeRO-2 配置
- `configs/ds_zero3_offload.json`：显存紧张时可用的 ZeRO-3 + CPU offload 配置
- `third_party/transformers-4.56.2`：用于 IDE 跳转的 Transformers 源码
- `third_party/deepspeed-0.14.5`：用于 IDE 跳转的 DeepSpeed 源码

## 环境要求

### macOS

macOS 当前主要用于：

- 加载本地模型
- 单轮推理
- 多轮对话
- 数据处理
- 阅读和调试代码

macOS 上可以使用 Apple Silicon 的 MPS 后端运行推理，但不适合运行 DeepSpeed 训练。DeepSpeed 训练一般需要 Linux + NVIDIA GPU + CUDA。

### Linux + NVIDIA GPU

DeepSpeed 训练建议使用：

- Linux
- NVIDIA GPU
- CUDA 版本匹配的 PyTorch
- `deepspeed`
- `transformers`
- `accelerate`

先按你的 CUDA 版本安装 PyTorch，再安装本工程依赖。

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

如果你使用 conda，也可以：

```bash
conda create -n qwen3-ds python=3.12 -y
conda activate qwen3-ds
pip install --upgrade pip
pip install -r requirements.txt
```

检查关键依赖：

```bash
python -c "import torch, transformers; print(torch.__version__); print(transformers.__version__); print(torch.cuda.is_available())"
python -c "import deepspeed; print(deepspeed.__version__)"
```

## 下载模型

默认模型路径是：

```text
models/Qwen3-0.6B
```

下载命令：

```bash
python scripts/download_model.py \
  --repo_id Qwen/Qwen3-0.6B \
  --local_dir models/Qwen3-0.6B
```

如果下载卡在 Hugging Face Xet 传输层，可以禁用 Xet 后重试：

```bash
HF_HUB_DISABLE_XET=1 python scripts/download_model.py \
  --repo_id Qwen/Qwen3-0.6B \
  --local_dir models/Qwen3-0.6B
```

模型目录至少应包含：

```text
config.json
generation_config.json
model.safetensors
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
```

## 数据格式

### SFT 数据

训练、验证、测试数据均为 JSONL 格式，每行一个 JSON 对象。

推荐格式是 `messages`：

```json
{"id":"example-001","messages":[{"role":"system","content":"你是一个简洁可靠的中文助手。"},{"role":"user","content":"什么是梯度累积？"},{"role":"assistant","content":"梯度累积是在多次小批量前向和反向传播后，再执行一次优化器更新。"}]}
```

也兼容 `instruction` / `input` / `output`：

```json
{"id":"example-002","instruction":"解释 DeepSpeed ZeRO。","input":"","output":"DeepSpeed ZeRO 通过切分优化器状态、梯度和参数来降低每张 GPU 的显存占用。"}
```

默认文件：

- `data/train.jsonl`：训练集
- `data/valid.jsonl`：验证集
- `data/test.jsonl`：测试集

训练脚本默认 `response_only_loss=True`，也就是只对最后一条 assistant 回复计算 loss，system/user prompt 部分会被 mask 成 `-100`。

### 预训练数据

预训练/继续预训练使用纯文本数据，不需要 `system/user/assistant` 对话格式。

默认文件：

- `data/pretrain.txt`：预训练样例训练文本
- `data/pretrain_valid.txt`：预训练样例验证文本

TXT 示例：

```text
大语言模型的预训练通常使用大规模纯文本语料。训练目标是根据前面的 token 预测下一个 token。
```

JSONL 示例：

```json
{"text":"这里是一段用于预训练的纯文本。"}
```

预训练脚本会把文本分词、拼接，并按 `--block_size` 切成固定长度训练块。训练目标是完整文本的下一个 token 预测，等价于：

```python
labels = input_ids
```

## 推理

### 单轮推理

使用原始模型：

```bash
python scripts/infer.py \
  --model_name_or_path models/Qwen3-0.6B \
  --prompt "用三句话说明什么是 DeepSpeed ZeRO。" \
  --max_new_tokens 256
```

使用训练后的模型：

```bash
python scripts/infer.py \
  --model_name_or_path outputs/qwen3-0.6b-sft \
  --prompt "给我一个学习大模型微调的计划。" \
  --max_new_tokens 256
```

确定性输出：

```bash
python scripts/infer.py \
  --model_name_or_path models/Qwen3-0.6B \
  --prompt "你好，用一句话介绍自己。" \
  --max_new_tokens 64 \
  --no-do_sample
```

### 通用交互模式

```bash
python scripts/infer.py \
  --model_name_or_path models/Qwen3-0.6B \
  --interactive
```

### macOS 多轮对话

当前 Mac 环境建议使用专门的多轮对话脚本。它会自动选择 `mps` 或 `cpu`：

```bash
python scripts/chat_mac.py \
  --model_name_or_path models/Qwen3-0.6B
```

更稳定的确定性输出：

```bash
python scripts/chat_mac.py \
  --model_name_or_path models/Qwen3-0.6B \
  --no-do_sample \
  --torch_dtype float16
```

交互命令：

- `/reset`：清空上下文
- `/history`：查看当前保留的用户轮数
- `/help`：查看命令
- `/exit`：退出

常用参数：

- `--max_new_tokens`：每轮最多生成的新 token 数
- `--max_context_tokens`：多轮上下文最大 token 数，超过后会丢弃较早轮次
- `--enable_thinking`：启用 Qwen3 chat template 的 thinking 模式
- `--system`：自定义 system prompt

## 训练方式 A：Trainer + DeepSpeed

`scripts/train.py` 使用 Hugging Face `Trainer`，通过 `--deepspeed` 参数接入 DeepSpeed。

单机单卡示例：

```bash
deepspeed --num_gpus 1 scripts/train.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/train.jsonl \
  --validation_file data/valid.jsonl \
  --output_dir outputs/qwen3-0.6b-sft \
  --deepspeed configs/ds_zero2.json \
  --model_max_length 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.03 \
  --logging_steps 5 \
  --eval_strategy steps \
  --eval_steps 20 \
  --save_steps 20 \
  --save_total_limit 2 \
  --bf16 \
  --gradient_checkpointing \
  --report_to none
```

显存较紧时可以换成 ZeRO-3 offload：

```bash
deepspeed --num_gpus 1 scripts/train.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/train.jsonl \
  --validation_file data/valid.jsonl \
  --output_dir outputs/qwen3-0.6b-sft-zero3 \
  --deepspeed configs/ds_zero3_offload.json \
  --model_max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --bf16 \
  --gradient_checkpointing \
  --report_to none
```

恢复训练：

```bash
deepspeed --num_gpus 1 scripts/train.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/train.jsonl \
  --validation_file data/valid.jsonl \
  --output_dir outputs/qwen3-0.6b-sft \
  --deepspeed configs/ds_zero2.json \
  --resume_from_checkpoint outputs/qwen3-0.6b-sft/checkpoint-20 \
  --bf16
```

适合使用这种方式的情况：

- 想快速使用 Transformers 训练生态
- 想少写训练循环
- 想复用 `Trainer` 的日志、保存、评估、恢复训练能力

## 训练方式 B：原生 DeepSpeed

`scripts/train_deepspeed.py` 是 SFT 手写训练循环，核心代码会直接调用：

```python
deepspeed.init_distributed()
deepspeed.initialize(...)
model_engine.backward(loss)
model_engine.step()
model_engine.save_checkpoint(...)
```

单机单卡示例：

```bash
deepspeed --num_gpus 1 scripts/train_deepspeed.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/train.jsonl \
  --validation_file data/valid.jsonl \
  --output_dir outputs/qwen3-0.6b-sft-ds \
  --deepspeed_config configs/ds_zero2.json \
  --model_max_length 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.03 \
  --logging_steps 5 \
  --eval_steps 20 \
  --save_steps 20 \
  --save_total_limit 2 \
  --bf16 \
  --gradient_checkpointing
```

ZeRO-3 offload 示例：

```bash
deepspeed --num_gpus 1 scripts/train_deepspeed.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/train.jsonl \
  --validation_file data/valid.jsonl \
  --output_dir outputs/qwen3-0.6b-sft-ds-zero3 \
  --deepspeed_config configs/ds_zero3_offload.json \
  --model_max_length 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --bf16 \
  --gradient_checkpointing
```

多卡示例：

```bash
deepspeed --num_gpus 4 scripts/train_deepspeed.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/train.jsonl \
  --validation_file data/valid.jsonl \
  --output_dir outputs/qwen3-0.6b-sft-ds-4gpu \
  --deepspeed_config configs/ds_zero2.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --bf16 \
  --gradient_checkpointing
```

原生 DeepSpeed 输出结构：

```text
outputs/qwen3-0.6b-sft-ds/
├── config.json
├── generation_config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── ds_checkpoints/
    ├── global_step20/
    └── latest
```

恢复原生 DeepSpeed 训练：

```bash
deepspeed --num_gpus 1 scripts/train_deepspeed.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/train.jsonl \
  --validation_file data/valid.jsonl \
  --output_dir outputs/qwen3-0.6b-sft-ds \
  --deepspeed_config configs/ds_zero2.json \
  --resume_from_checkpoint outputs/qwen3-0.6b-sft-ds/ds_checkpoints/global_step20 \
  --bf16
```

适合使用这种方式的情况：

- 想学习 DeepSpeed 原生 API
- 想清楚看到训练循环每一步
- 想自己控制 optimizer、scheduler、checkpoint 和评估逻辑

## 训练方式 C：原生 DeepSpeed 预训练/继续预训练

`scripts/pretrain_deepspeed.py` 使用纯文本做因果语言模型训练。它和 SFT 的区别是：

- 不使用 chat template
- 不区分 user/assistant
- 不 mask prompt
- 对完整文本计算下一个 token 预测损失

默认情况下，脚本会从 `models/Qwen3-0.6B` 加载已有权重，这更准确地叫“继续预训练”。如果加上 `--from_scratch`，脚本会只读取模型配置并随机初始化参数，做真正从零预训练。

继续预训练示例：

```bash
deepspeed --num_gpus 1 scripts/pretrain_deepspeed.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/pretrain.txt \
  --validation_file data/pretrain_valid.txt \
  --output_dir outputs/qwen3-0.6b-pretrain-ds \
  --deepspeed_config configs/ds_zero2.json \
  --block_size 2048 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --warmup_ratio 0.03 \
  --logging_steps 5 \
  --eval_steps 20 \
  --save_steps 20 \
  --save_total_limit 2 \
  --bf16 \
  --gradient_checkpointing
```

大文件流式继续预训练示例：

```bash
deepspeed --num_gpus 1 scripts/pretrain_deepspeed.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/pretrain_examples_10.jsonl \
  --validation_file data/pretrain_valid.txt \
  --text_field text \
  --output_dir outputs/qwen3-0.6b-pretrain-streaming-ds \
  --deepspeed_config configs/ds_zero2.json \
  --block_size 2048 \
  --streaming \
  --max_steps 1000 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 2e-5 \
  --bf16 \
  --gradient_checkpointing
```

流式模式不会把全部语料一次性读入内存，适合较大的 TXT/JSONL 文件。由于流式数据集没有固定长度，必须显式传入 `--max_steps`。在多 GPU 或多 dataloader worker 下，脚本会按分布式进程和 worker 对文档做分片，避免每个进程重复读取同一批样本。

从零预训练示例：

```bash
deepspeed --num_gpus 1 scripts/pretrain_deepspeed.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/pretrain.txt \
  --validation_file data/pretrain_valid.txt \
  --output_dir outputs/qwen3-0.6b-pretrain-from-scratch-ds \
  --deepspeed_config configs/ds_zero2.json \
  --block_size 2048 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-4 \
  --num_train_epochs 3 \
  --bf16 \
  --gradient_checkpointing \
  --from_scratch
```

使用 JSONL 文本字段：

```bash
deepspeed --num_gpus 1 scripts/pretrain_deepspeed.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/domain_text.jsonl \
  --validation_file data/domain_valid.jsonl \
  --text_field text \
  --output_dir outputs/qwen3-0.6b-domain-pretrain-ds \
  --deepspeed_config configs/ds_zero2.json \
  --bf16
```

原生 DeepSpeed 预训练输出结构：

```text
outputs/qwen3-0.6b-pretrain-ds/
├── config.json
├── generation_config.json
├── pytorch_model.bin
├── tokenizer.json
├── tokenizer_config.json
└── ds_checkpoints/
    ├── global_step20/
    └── latest
```

恢复预训练：

```bash
deepspeed --num_gpus 1 scripts/pretrain_deepspeed.py \
  --model_name_or_path models/Qwen3-0.6B \
  --train_file data/pretrain.txt \
  --validation_file data/pretrain_valid.txt \
  --output_dir outputs/qwen3-0.6b-pretrain-ds \
  --deepspeed_config configs/ds_zero2.json \
  --resume_from_checkpoint outputs/qwen3-0.6b-pretrain-ds/ds_checkpoints/global_step20 \
  --bf16
```

## 关键训练参数

- `--model_name_or_path`：模型路径，默认 `models/Qwen3-0.6B`
- `--train_file`：训练 JSONL 文件
- `--validation_file`：验证 JSONL 文件
- `--output_dir`：输出目录
- `--model_max_length`：最大上下文长度
- `--block_size`：预训练文本切块长度
- `--text_field`：JSONL 预训练数据中的文本字段名
- `--line_by_line`：把 TXT 文件每一行当作独立文档读取
- `--streaming`：启用流式预训练数据集，边读边分词；必须设置 `--max_steps`
- `--append_eos` / `--no-append_eos`：是否在每段文本后追加 EOS
- `--drop_last_block` / `--no-drop_last_block`：是否丢弃最后一个不足 `block_size` 的文本块
- `--from_scratch`：只读取配置并随机初始化模型参数
- `--per_device_train_batch_size`：每张 GPU 的 micro batch size
- `--gradient_accumulation_steps`：梯度累积步数
- `--learning_rate`：学习率
- `--num_train_epochs`：训练轮数
- `--max_steps`：最大训练 step，设置后优先于 epoch
- `--warmup_ratio` / `--warmup_steps`：学习率 warmup
- `--bf16` / `--fp16`：混合精度
- `--gradient_checkpointing`：启用梯度检查点，省显存但会更慢
- `--enable_thinking`：启用 Qwen3 thinking 模式
- `--response_only_loss`：只对 assistant 回复算 loss，默认开启
- `--no-response_only_loss`：对完整文本都算 loss
- `--skip_empty_labels` / `--no-skip_empty_labels`：是否过滤截断后没有可训练 label 的 SFT 样本，默认开启

全局 batch size 计算方式：

```text
global_batch_size = per_device_train_batch_size * gradient_accumulation_steps * num_gpus
```

## 评测

使用训练后的模型评测：

```bash
python scripts/evaluate.py \
  --model_name_or_path outputs/qwen3-0.6b-sft \
  --eval_file data/test.jsonl \
  --output_file outputs/eval_predictions.jsonl \
  --metrics_file outputs/eval_metrics.json
```

使用原始模型做冒烟评测：

```bash
python scripts/evaluate.py \
  --model_name_or_path models/Qwen3-0.6B \
  --eval_file data/test.jsonl \
  --output_file outputs/smoke_eval_predictions.jsonl \
  --metrics_file outputs/smoke_eval_metrics.json \
  --max_new_tokens 64
```

评测输出：

- `outputs/*predictions.jsonl`：每条样本的 prompt、reference、prediction
- `outputs/*metrics.json`：基础指标

当前内置指标：

- `exact_match`：预测与参考答案完全匹配
- `contains_reference`：预测中包含参考答案

真实任务中建议按业务替换为 ROUGE、BLEU、分类准确率、人工偏好评测或 LLM-as-judge。

DeepSpeed 训练脚本内部的验证 loss 会按有效 token 数加权统计，不再按 batch 简单平均；padding 和被 mask 成 `-100` 的 token 不参与统计。

## 源码跳转

本工程已经把 `transformers` 和 `deepspeed` 源码下载到 `third_party`，并配置了 VS Code/Pylance 路径：

- `.vscode/settings.json`
- `pyrightconfig.json`

常用入口：

- `third_party/transformers-4.56.2/src/transformers/models/qwen3/modeling_qwen3.py`
- `third_party/transformers-4.56.2/src/transformers/models/auto/modeling_auto.py`
- `third_party/deepspeed-0.14.5/deepspeed/__init__.py`
- `third_party/deepspeed-0.14.5/deepspeed/runtime/engine.py`
- `third_party/deepspeed-0.14.5/deepspeed/runtime/zero/stage_1_and_2.py`
- `third_party/deepspeed-0.14.5/deepspeed/runtime/zero/stage3.py`

如果 VS Code 仍然跳到 `site-packages`：

1. 执行 `Python: Restart Language Server`
2. 确认打开的是当前工程根目录
3. 确认使用的是包含 `pyrightconfig.json` 的工作区

## 快速验证

语法检查：

```bash
python -m py_compile \
  scripts/common.py \
  scripts/download_model.py \
  scripts/infer.py \
  scripts/chat_mac.py \
  scripts/evaluate.py \
  scripts/train.py \
  scripts/train_deepspeed.py \
  scripts/pretrain_deepspeed.py
```

本地模型推理冒烟测试：

```bash
python scripts/infer.py \
  --model_name_or_path models/Qwen3-0.6B \
  --prompt "你好，请用一句话介绍自己。" \
  --max_new_tokens 32 \
  --no-do_sample \
  --torch_dtype float16
```

macOS 多轮对话冒烟测试：

```bash
printf '你好\n/exit\n' | python scripts/chat_mac.py \
  --model_name_or_path models/Qwen3-0.6B \
  --max_new_tokens 16 \
  --no-do_sample \
  --torch_dtype float16
```

## 常见问题

### macOS 能训练吗？

可以运行普通 PyTorch/MPS 的小规模实验，但本工程的 DeepSpeed 训练脚本要求 CUDA GPU。当前 `scripts/train_deepspeed.py` 会在没有 CUDA 时直接提示：

```text
原生 DeepSpeed 训练需要 CUDA GPU。macOS 仅适合运行推理。
```

### 为什么 `train_deepspeed.py` 里 `import deepspeed` 在 `main()` 内部？

这是为了让当前 Mac 环境可以正常运行 `--help` 和语法检查。DeepSpeed 在 macOS 上导入时可能触发 CUDA 或编译相关依赖问题，因此只有真正开始训练时才导入。

### 什么时候用 ZeRO-2，什么时候用 ZeRO-3 offload？

优先用 `configs/ds_zero2.json`，速度通常更好。如果显存不够，再换 `configs/ds_zero3_offload.json`。ZeRO-3 offload 会把部分参数或优化器状态放到 CPU 内存，显存更省，但速度更慢。

### bf16 和 fp16 怎么选？

支持 bf16 的 NVIDIA GPU 上优先用 `--bf16`，通常更稳定。不支持 bf16 时再用 `--fp16`。

### 训练后怎么推理？

Trainer 输出：

```bash
python scripts/infer.py \
  --model_name_or_path outputs/qwen3-0.6b-sft \
  --prompt "解释一下梯度累积。"
```

原生 DeepSpeed 输出：

```bash
python scripts/infer.py \
  --model_name_or_path outputs/qwen3-0.6b-sft-ds \
  --prompt "解释一下梯度累积。"
```

### 数据太少怎么办？

当前 `data/*.jsonl` 是样例数据，只用于跑通流程。真实微调需要替换为你的任务数据，并至少保留独立的验证集和测试集。

### 当前训练脚本哪些是 SFT，哪些是预训练？

`scripts/train.py` 和 `scripts/train_deepspeed.py` 是 SFT。它们读取 `messages` 或 `instruction/input/output`，默认只对 assistant 回复算 loss。

`scripts/pretrain_deepspeed.py` 是预训练/继续预训练。它读取纯文本，默认对完整文本算因果语言模型 loss。

### 恢复训练时会从哪里继续？

原生 DeepSpeed 脚本会从 checkpoint 恢复模型、优化器、学习率调度器和全局 step，并根据已完成的更新步数推算已经消费的 micro batch 数，跳到对应 epoch 和 batch 后继续训练。对于 streaming 预训练数据，脚本会从流开头跳过已经消费的 batch；如果 checkpoint 很靠后，这个跳过过程会需要一些时间。

### SFT 样本过长会怎样？

如果 prompt 太长导致 assistant 回复在截断后完全消失，脚本会过滤掉这类没有可训练 label 的样本，避免 loss 里全是 `-100`。如果过滤后没有剩余样本，会直接报错提示检查长度截断或数据格式。
