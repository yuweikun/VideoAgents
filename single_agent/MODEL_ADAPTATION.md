# Single-Agent（LangChain + Qwen3-VL）说明

## 方法概览
- 直接用根目录 `template.py`（one_shot + INSTRUCTION）构造提示，单轮生成 5 段驾驶报告。
- 输入：本地视频（mp4）+ 车辆动力/刹车/转向信号和注视数据（CSV）+ `labels.json`（推荐动作/自动驾驶状态）。
- 模型：DashScope `qwen3-vl-flash` 负责多模态生成；LangChain 单代理（zero-shot-react）仅用于调用一个 Tool。

## 脚本入口
- `single_agent/run_single_agent.py`
  - 构建 payload（采样信号/注视，组装动作集）
  - 构建 prompt（模板 + 可选附加文本）
  - Tool：调用 Qwen3-VL（文本 + 视频）返回报告
  - 代理：LangChain `zero-shot-react-description`，调用工具一次
  - 输出：JSON，包含 `video_uri`、`payload`、`report`

## 运行示例
```powershell
$env:QWEN_API_KEY="your_key"
$env:DASHSCOPE_API_KEY=$env:QWEN_API_KEY
cd "C:\Program Files\code"
python single_agent/run_single_agent.py --folder 01-1 --file-stem start_at_min02sec03 --output single_agent/outputs/output_single_agent.json
```
可选参数：`--video-dir`（自定义视频目录），`--extra-text`（附加提示），`--output`（默认写入 `single_agent/outputs/<file_stem>.json`）。

## 更换模型
- 修改 `run_single_agent.py` 的 `DEFAULT_VIDEO_MODEL` / `qwen_video_report` 为你的多模态模型；`DEFAULT_TEXT_MODEL`/`ChatOpenAI` 为你的文本代理。
- 更新鉴权/endpoint（当前用 DashScope 兼容模式）。

## 修改输入
- 重写 `build_payload` 适配你的数据（CSV/JSON/其他信号）；视频路径由 `--folder`/`--file-stem`/`--video-dir` 决定，可新增字段。

## 修改输出/提示词
- 提示构造：`build_prompt`（源自 `template.py`），可直接改模板或用 `--extra-text` 补充。
- 输出结构：`run_agent` 返回 `payload` + `report`，需要更多字段可在返回字典中扩展。

## 注意
- 单轮生成，不含批判/迭代重写；如需多轮，可参考 `VideoAgent/run_videoagent_qwen_iter.py`。 
