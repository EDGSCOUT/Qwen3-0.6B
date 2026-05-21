from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="将 Qwen3 模型权重下载到本地目录。")
    parser.add_argument("--repo_id", default="Qwen/Qwen3-0.6B", help="Hugging Face 仓库 ID。")
    parser.add_argument("--local_dir", default="models/Qwen3-0.6B", help="本地模型目录。")
    parser.add_argument("--revision", default=None, help="可选的分支、标签或提交 ID。")
    parser.add_argument("--token", default=None, help="可选的 Hugging Face token。")
    parser.add_argument(
        "--allow_pattern",
        action="append",
        default=None,
        help="可选的允许下载规则，可以传入多次。",
    )
    parser.add_argument(
        "--ignore_pattern",
        action="append",
        default=None,
        help="可选的忽略下载规则，可以传入多次。",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    output_dir = snapshot_download(
        repo_id=args.repo_id,
        revision=args.revision,
        local_dir=str(local_dir),
        token=args.token,
        allow_patterns=args.allow_pattern,
        ignore_patterns=args.ignore_pattern,
    )
    print(f"模型已下载到：{output_dir}")


if __name__ == "__main__":
    main()
