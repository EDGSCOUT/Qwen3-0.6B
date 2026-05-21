# 数据目录

默认文件：

- `train.jsonl`：训练集
- `valid.jsonl`：验证集
- `test.jsonl`：测试集
- `pretrain.txt`：预训练/继续预训练样例训练文本
- `pretrain_valid.txt`：预训练/继续预训练样例验证文本
- `sft_examples_10.jsonl`：10 条 SFT 样例数据
- `pretrain_examples_10.jsonl`：10 条预训练/继续预训练 JSONL 样例数据

每行一个 JSON 对象，推荐字段：

```json
{"messages":[{"role":"system","content":"你是一个助手。"},{"role":"user","content":"问题"},{"role":"assistant","content":"答案"}]}
```

也兼容：

```json
{"instruction":"问题","input":"补充上下文，可为空","output":"答案"}
```

预训练脚本使用纯文本数据，默认每个 `.txt` 文件会被作为连续文本读取、分词、拼接并切成固定长度块。也可以使用 JSONL：

```json
{"text":"这里是一段用于预训练的纯文本。"}
```

使用 10 条样例数据训练时，可以把训练命令中的文件路径替换为：

```bash
--train_file data/sft_examples_10.jsonl
```

或：

```bash
--train_file data/pretrain_examples_10.jsonl --text_field text
```
