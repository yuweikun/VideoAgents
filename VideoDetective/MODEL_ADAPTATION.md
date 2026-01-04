# 视频+信号→报告（单轮生成）与 VideoDetective 方法说明

## 论文核心（VideoDetective, arXiv:2512.17229）
- 问题感知记忆压缩：长视频分段，每段末尾加少量记忆 token，问题放在记忆前，引导注意力只“记住”与问题相关的关键信息。
- 跨段递归聚合：每段处理后提取记忆 KV 累加到记忆库，作为下一段上下文；最终仅用少量记忆 token 作答，无需全量帧，显著节省显存/算力。
- 实现位置：`models/modeling_beacon.py`（Beacon/记忆压缩与跨段累积）、`configs/config.py`（窗口/步长/压缩比等），训练/微调脚本在 `scripts/pretrain`、`scripts/finetune2`。

## 我们的单轮脚本（视频+信号→报告）
- 入口：`run_qwen_template.py`（不启用 Beacon 机制，直接单轮生成）。
- 输入：`sample_data` 下的 mp4 + 车辆/注视 CSV + `labels.json`，提示模板来自根目录 `template.py`（one_shot + INSTRUCTION）。
- 模型：DashScope `qwen3-vl-flash`（环境变量 `QWEN_API_KEY` / `DASHSCOPE_API_KEY`）。
- 输出：JSON，包含 `video_uri`、`payload`、`report`（5 段报告）。

### 运行示例
```powershell
$env:QWEN_API_KEY="your_key"
$env:DASHSCOPE_API_KEY=$env:QWEN_API_KEY
cd "C:\Program Files\code\VideoDetective"
python run_qwen_template.py --folder 01-1 --file-stem start_at_min02sec03 --output outputs/output_qwen_template.json
```
可选参数：`--video-dir`（自定义视频目录）、`--extra-text`（附加提示）、`--output`（留空则写入 `VideoDetective/outputs/<file_stem>.json`）。

## 如何换模型
- 修改 `run_qwen_template.py` 中的 `DEFAULT_MODEL` 与 `qwen_chat` 调用为你的 SDK/模型名；调整鉴权、`max_tokens`、`temperature` 等。

## 如何改输入
- 重写 `build_payload` 读取你的数据格式（CSV/JSON/其他信号）；视频路径由 `--folder`/`--file-stem`/`--video-dir` 决定，可在此处增删字段。

## 如何改输出/提示词
- 提示构造在 `build_prompt`（模板源自 `template.py`）；可直接改模板或用 `--extra-text` 补充。
- 输出结构在 `run_once`（默认 `payload`+`report`），需要更多字段可在此扩展。

## 若需多轮/批判重写
- 可参考 `VideoAgent/run_videoagent_qwen_iter.py` 的迭代评分+重写流程；若要使用 VideoDetective 原生记忆机制，需走其训练/推理链路，不在本脚本启用。
