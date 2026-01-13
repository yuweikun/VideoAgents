# CAMEL RolePlaying（Ollama）多角色对话

## 编排逻辑（官方风格）
- 使用 CAMEL 的 `RolePlaying` 进行两角色对话：**Report Writer**（助手）与 **Safety Reviewer**（用户角色）。
- 对话只跑一轮：Writer 先给初稿，Reviewer 给审阅要点。
- 第三个角色 **Final Editor**（独立 `ChatAgent`）整合初稿与审阅要点，输出最终报告。

## 脚本入口
- `run_camel_roleplay_ollama.py`

## 运行方式（与 ollama_single 一致）
```powershell
cd "C:\Program Files\code\ollama\camel_roleplay"
python run_camel_roleplay_ollama.py --video-path ..\..\sample_data\01-1\start_at_min02sec03.mp4 --folder 01-1 --file-stem start_at_min02sec03 --output outputs\out.txt
```
参数：`--video-path`/`--video-url`，`--folder`，`--file-stem`，`--num-frames`，`--model`，`--base-url`，`--output`。

## 模型与采样
- 模型：本地 Ollama（OpenAI 兼容），默认 `qwen3-vl:2b`。
- 采样：视频→均匀抽取 `--num-frames` 帧→以 `image_list` 方式传给 CAMEL。
- 修改默认模型/端点：`DEFAULT_MODEL` / `DEFAULT_BASE_URL`。

## 输入与提示词
- 信号来源：`sample_data` 的 CSV + `labels.json`。
- 提示词入口：`build_prompt`（模板来自根目录 `template.py`）。
- 如需换数据格式：重写 `build_payload`。

## 输出
- 默认只输出最终报告文本；若传 `--output` 则写入文件。
