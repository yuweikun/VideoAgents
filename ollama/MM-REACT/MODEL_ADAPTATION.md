# MM-REACT 方法概览
- 多模态代理框架：基于 LangChain 的“conversational-mm-assistant”代理，LLM（原版 Azure OpenAI）+ 一组工具（图像理解、OCR、收据/名片/布局/发票、名人识别、Bing 搜索、PAL 计算、图片编辑等）。
- 代理负责解析用户问题，按需调用工具，组合结果作答；训练免费，工具可插拔。
- 代码入口示例：`sample.py`，初始化 LLM、工具列表，`initialize_agent(..., agent="conversational-mm-assistant")` 后对话。

# 我们的脚本（驾驶评估，Qwen3-VL）
- 入口：`driver_eval_qwen_mmreact.py`（单轮生成报告，用 Qwen3-VL，多模态视频+信号输入）。
- 输入：`sample_data` 下的 mp4 + 车辆/注视信号 CSV + `labels.json`，模板 `template.py`（one_shot + INSTRUCTION）。
- 模型：DashScope `qwen3-vl-flash`（环境变量 `QWEN_API_KEY` / `DASHSCOPE_API_KEY`）。
- 输出：JSON，包含 `video_uri`、`payload`、`draft_report`。

# 运行示例
```powershell
$env:QWEN_API_KEY="your_key"
$env:DASHSCOPE_API_KEY=$env:QWEN_API_KEY
cd "C:\Program Files\code\MM-REACT"
python driver_eval_qwen_mmreact.py --folder 01-1 --file-stem start_at_min02sec03 --output outputs/output_qwen_mmreact.json
```
参数：`--video-dir`（自定义视频目录，默认 `sample_data/<folder>`），`--output`（建议相对路径 `outputs/xxx.json`）。

# 更换模型
- 在 `driver_eval_qwen_mmreact.py` 修改 `DEFAULT_MODEL` / `qwen_chat` 调用为你的 SDK/模型名；调整鉴权方式、`max_tokens`、`temperature` 等。
- 环境变量按新模型要求设置；不用 DashScope 时移除相关设置。

# 修改输入
- 数据读取在 `build_payload`（读取车辆/注视 CSV、labels；采样信号、构造动作集）。若格式不同，在此重写。
- 视频路径由 `--folder`/`--file-stem`/`--video-dir` 决定；可在 payload 中添加自定义字段并在 prompt 中引用。

# 修改输出/提示词
- 提示构造：`build_prompt`（模板源自根目录 `template.py`），可直接改模板或在此追加字段。
- 输出结构：`run_pipeline` 返回 `draft` 等字段；末尾可追加格式化或后处理，按需扩展返回字典。
- 如需更严格的格式，可在 prompt 中明确结构或在生成后解析/校验。

# 注意
- 本脚本为单轮生成，不含多轮批判/反思；如需迭代可参考 `VideoAgent/run_videoagent_qwen_iter.py` 的改写/评分流程。
- 原版 `sample.py` 的工具链（Azure OpenAI + 多工具）与本脚本独立，使用时请勿混用模型/鉴权。 
