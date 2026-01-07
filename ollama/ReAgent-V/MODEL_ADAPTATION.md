# ReAgent-V 方法概览（[arXiv:2506.01300](https://arxiv.org/abs/2506.01300)）
- 多代理 + 奖励驱动：主模型先出初稿，批判代理按多维度打分生成 scalar_reward，保守/中立/激进三路反思，元代理融合得最终答案。
- 工具化：可接入多模态工具（OCR/ASR/检测/场景图等），按需调用；支持奖励感知的数据筛选/对齐。
- 目标：通过奖励与自反思提升视频理解/问答的可靠性，并可产出对齐/训练数据。

# 我们的脚本（仅保留并推荐）
- `run_driver_qwen_reagent.py`：多代理反思链路（初稿→批判评分→三路反思→元融合→格式化），输入 `sample_data` 视频 + 信号 + 根目录 `template.py`，模型 Qwen3-VL（DashScope）。

# 运行示例
```powershell
$env:QWEN_API_KEY="your_key"
$env:DASHSCOPE_API_KEY=$env:QWEN_API_KEY
cd "C:\Program Files\code\ReAgent-V"
python run_driver_qwen_reagent.py --folder 01-1 --file-stem start_at_min02sec03 --output outputs/output_qwen_reagent.json
```
参数：`--video-dir`（自定义视频目录，默认 `sample_data/<folder>`），`--output`（建议相对路径 `outputs/xxx.json`）。

# 更换模型
- 在 `run_driver_qwen_reagent.py` 中修改 `DEFAULT_MODEL` / `qwen_chat` 为你的 SDK/模型名；更新鉴权、`max_tokens`、`temperature` 等。
- 环境变量按新模型要求设置；不用 DashScope 时可移除相关设置。

# 修改输入
- 重写 `build_payload` 以适配你的数据格式（车辆/注视 CSV、labels 等）；视频路径由 `--folder`/`--file-stem`/`--video-dir` 决定，可在 payload 中添加自定义字段并在 prompt 中引用。

# 修改输出与提示词
- Prompt 位置：初稿 `build_prompt`，批判 `eval_reward_prompt_template`，三路反思/重写 prompt，末尾格式化 prompt（五段+Top3 动作）。
- 输出结构：脚本返回 JSON（草稿/评估/反思/最终格式化）；需要额外字段可在返回字典中扩展。

# 注意
- 需外网访问 Qwen3-VL API，确保 key 正确且账户可用。若只需单轮输出，可参考 `VideoDetective/run_qwen_template.py`。 
