# 第三方源码

本目录保存当前工程依赖的第三方源码，方便 IDE 跳转和阅读实现。

- `transformers-4.56.2/`：与当前 Python 环境中 `transformers==4.56.2` 对应的 PyPI 源码包
- `transformers-4.56.2/src/transformers/`：IDE 跳转使用的包源码目录
- `accelerate-1.13.0/`：与 `requirements.txt` 中 Accelerate 最低版本对应的 PyPI 源码包
- `accelerate-1.13.0/src/accelerate/`：IDE 跳转使用的 Accelerate 包源码目录
- `deepspeed-0.14.5/`：与 `requirements.txt` 中 DeepSpeed 最低版本对应的 PyPI 源码包
- `deepspeed-0.14.5/deepspeed/`：IDE 跳转使用的 DeepSpeed 包源码目录

VS Code/Pylance 的跳转路径已经配置在：

- `.vscode/settings.json`
- `pyrightconfig.json`

如果 IDE 仍然跳到 site-packages，可以执行一次 “Python: Restart Language Server”，或重新打开当前工作区。
