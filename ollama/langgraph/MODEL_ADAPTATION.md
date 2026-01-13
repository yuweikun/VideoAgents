# LangGraph（Ollama）多代理单轮评估

## 方法概述
- 框架：LangGraph，以“有向图 + 状态”方式编排多代理。
- 目标：视频 + 车辆信号 → 驾驶评估报告（五段模板）。
- 逻辑：拆成三位 agent，各做一件事，单轮完成，不做迭代。

## 三个 agent 的职责
- Drafter：根据模板 + 采样帧生成初稿。
- Critic：只做审阅要点（检查结构与动作集合），不重写。
- Synthesizer：结合初稿 + 审阅要点生成最终报告。

## 脚本位置
- 入口脚本：`run_langgraph_ollama.py`

## 运行方式（与 ollama_single 一致）
```powershell
cd "C:\Program Files\code\ollama\langgraph"
python run_langgraph_ollama.py --video-path ..\..\sample_data\01-1\start_at_min02sec03.mp4 --folder 01-1 --file-stem start_at_min02sec03 --output outputs\out.txt
```
参数：`--video-path`/`--video-url`，`--folder`，`--file-stem`，`--num-frames`，`--model`，`--base-url`，`--output`。

## 模型与采样
- 模型：本地 Ollama（OpenAI 兼容），默认 `qwen3-vl:2b`。
- 采样：视频→均匀抽取 `--num-frames` 帧→以 `images` 方式传给 Ollama。
- 修改默认模型/端点：`DEFAULT_MODEL` / `DEFAULT_BASE_URL`。

## 输入与提示词
- 信号来源：`sample_data` 的 CSV + `labels.json`。
- 提示词入口：`build_prompt`（模板来自根目录 `template.py`）。
- 如需换数据格式：重写 `build_payload`。

## 输出
- 默认只输出最终报告文本；若传 `--output` 则写入文件。
- 如需输出中间结果，可在 `run_langgraph_ollama.py` 增加 `draft`/`critique` 的保存。
