# 视频+信号→报告（单轮生成）说明

## 方法概述
- 使用根目录 `template.py` 的 one_shot + INSTRUCTION，直接生成 5 段驾驶评估报告（Scene / Attention / HMI / Evaluation & Suggestions / Recommended Actions）。
- 输入：本地视频（mp4）＋车辆动力/刹车/转向信号和注视数据（CSV），以及 `labels.json` 中的推荐动作和自动驾驶状态。
- 模型：DashScope `qwen3-vl-flash`，多模态对话接口（文本 prompt + 视频）。

## 快速运行
```powershell
$env:QWEN_API_KEY="your_key"
$env:DASHSCOPE_API_KEY=$env:QWEN_API_KEY
cd "C:\Program Files\code\VideoDetective"
python run_qwen_template.py --folder 01-1 --file-stem start_at_min02sec03 --output outputs/output_qwen_template.json
```
参数：
- `--folder` / `--file-stem`：确定视频与对应 CSV/label（默认 `sample_data/<folder>/<file_stem>.mp4`）。
- `--video-dir`：自定义视频目录（可选）。
- `--extra-text`：附加提示文本（可选）。
- `--output`：保存路径（相对路径推荐 `outputs/xxx.json`；留空则自动写入 `VideoDetective/outputs/<file_stem>.json`）。

## 修改模型
- 打开 `run_qwen_template.py`：
  - 改 `DEFAULT_MODEL` 和 `qwen_chat` 的调用为你的 SDK/模型名。
  - 更新鉴权方式/环境变量（若不用 DashScope，可去掉相关设置）。
  - 调整 `max_tokens`、`temperature` 等参数。

## 修改输入
- `build_payload`：读取 `vehicle_dynamics_data/<folder>.csv`、`vehicle_gaze_data/<folder>.csv`、`labels.json`，并采样信号。若数据格式不同，在此重写读取/采样逻辑。
- 视频路径由 `--folder`/`--file-stem`/`--video-dir` 决定；若使用其他模态或元数据，可在 `build_payload` 增加字段，并在 prompt 中引用。

## 修改输出与提示词
- 提示构造：`build_prompt`（使用 `template.py`），可直接改模板或在 prompt 中追加结构/说明（`--extra-text` 也可补充）。
- 输出结构：`run_once` 返回 `{"video_uri","payload","report"}`，需要更多字段可在此扩展。
- 如模型输出格式需约束，可在 prompt 中明确格式要求，或在 `qwen_chat` 后做简单解析/校验。

## 与多轮版本的关系
- 本脚本为单轮生成，无批判/迭代重写。如需多轮优化，可参考 `VideoAgent/run_videoagent_qwen_iter.py` 的迭代/评分/重写逻辑。
