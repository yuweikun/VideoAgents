# 视频+信号→报告（单轮生成）与 VideoDetective 方法说明

## 论文核心（VideoDetective, arXiv:2512.17229）
- 问题感知记忆压缩：长视频分段，每段末尾加入少量记忆 token，问题放在记忆前，引导注意力只“记住”与问题相关的关键信息。
- 跨段递归聚合：每段处理后提取记忆 KV 累加到记忆库，作为下一段上下文；最终仅用少量记忆 token 作答，显著节省显存/算力。
- 实现位置：`models/modeling_beacon.py`（Beacon/记忆压缩与跨段累积）、`configs/config.py`（窗口/步长/压缩比等），训练/微调脚本在 `scripts/pretrain`、`scripts/finetune2`。

## 我们的单轮脚本（Ollama）
- 入口：`run_qwen_template.py`（不启用 Beacon 机制，直接单轮生成）。
- 输入：本地/URL 视频（mp4）+ `sample_data` 中的车辆/注视信号 CSV + `labels.json` + 根目录 `template.py` 的 one_shot + INSTRUCTION。
- 模型：本地 Ollama（OpenAI 兼容），默认 `qwen3-vl:2b`。
- 输出：JSON（`video_uri`、`payload`、`report`）。

## 运行示例
```powershell
cd "C:\Program Files\code\ollama\VideoDetective"
python run_qwen_template.py --video-path ..\..\sample_data\01-1\start_at_min02sec03.mp4 --folder 01-1 --file-stem start_at_min02sec03 --output outputs\output_qwen_template.json
```
参数：`--video-path`/`--video-url`，`--folder`，`--file-stem`，`--num-frames`，`--model`，`--base-url`，`--extra-text`，`--output`。

## 如何换模型
- 推荐：用 `--model`/`--base-url` 替换。
- 如需固定默认：修改 `DEFAULT_MODEL`/`DEFAULT_BASE_URL`。
- 生成风格：在 `qwen_chat` 调整 `temperature`/`max_tokens`。

## 如何改输入
- `build_payload`：读取 CSV、labels、动作集；输入格式变动在此重写。
- 视频路径由 `--video-path`/`--video-url` 决定；不传则默认 `sample_data/{folder}/{file_stem}.mp4`。
- 采样帧数由 `--num-frames` 控制。

## 如何改输出/提示词
- Prompt 位置：`build_prompt`（模板来自 `template.py`），可用 `--extra-text` 追加说明。
- 输出结构在 `run_once` 中定义，默认返回 `payload` + `report`。

## 若需多轮/批判重写
- 参考 `ollama/VideoAgent/run_videoagent_qwen_iter.py` 的迭代评分+重写流程；若要使用 VideoDetective 原生记忆机制，需走其训练/推理链路，不在本脚本启用。
