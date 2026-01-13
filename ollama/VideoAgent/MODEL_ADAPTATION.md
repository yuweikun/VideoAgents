# VideoAgent 方法概览（arXiv:2403.10517）
- 核心思想：LLM 作为中枢代理，面对长视频问答，先用少量均匀抽帧作答并自评置信度，再按需“计划→检索更关键帧→重答”，以少帧高效完成长视频理解。
- 工具链（论文版）：多模态工具（字幕/检索/描述等）供 LLM 调用获取更多视觉证据；本仓库简化为预存 caption + CLIP 相似度帧检索。
- 流程（对应代码简化版）：初次采样 5 帧→LLM 作答+自评→若置信度不足，按时间段生成候选描述并用 CLIP 检索补帧→迭代一到两次→最终输出答案（在我们改造的脚本中输出驾驶报告）。

# 我们的脚本（驾驶评估，Ollama）
- 入口：`run_videoagent_qwen_iter.py`（与原 `main.py` 的多选 QA 流程独立）。
- 功能：多轮生成驾驶评估报告（5 段），批判打分后若分数低则重写，最后强制格式化。
- 输入：本地/URL 视频（mp4）+ `sample_data` 中的车辆/注视信号 CSV + `labels.json` + 根目录 `template.py` 的 one_shot + INSTRUCTION。
- 模型：本地 Ollama（OpenAI 兼容），默认 `qwen3-vl:2b`。

# 如何运行
```bash
cd "C:\Program Files\code\ollama\VideoAgent"
python run_videoagent_qwen_iter.py --video-path ..\..\sample_data\01-1\start_at_min02sec03.mp4 --folder 01-1 --file-stem start_at_min02sec03 --output outputs\out.txt
```
参数：`--video-path`/`--video-url`，`--folder`，`--file-stem`，`--num-frames`，`--model`，`--base-url`，`--output`。

# 更换模型
- 推荐：直接用 `--model`/`--base-url` 替换。
- 如需固定默认值：修改 `DEFAULT_MODEL`/`DEFAULT_BASE_URL`。
- 调整生成：在 `qwen_chat` 中改 `temperature`/`max_tokens`。

# 修改输入
- `build_payload`：读取 CSV、labels、动作集；输入格式变动时在这里重写。
- 视频路径由 `--video-path`/`--video-url` 决定；不传则默认 `sample_data/{folder}/{file_stem}.mp4`。
- 采样帧数由 `--num-frames` 控制。

# 修改输出/提示词
- Prompt 位置：`build_prompt`（初稿）、`eval_reward_prompt_template` 调用处（批判）、`refine_prompt`（重写）、`format_prompt`（最终格式）。
- 输出结构：仅输出最终报告文本，无 `history` JSON。如需返回过程信息，修改 `run_iter` 返回值。
- 若模型不输出 JSON 评分，可放宽或调整 `parse_reward` 的解析逻辑。

# 原版多选 QA（仅参考）
- 入口 `main.py`：等距抽帧→CLIP 检索→LLM 作答，自评置信度，多轮检索。

# 最短改动清单
1) 换模型：`--model`/`--base-url` 或 `DEFAULT_MODEL`/`DEFAULT_BASE_URL`
2) 改输入：重写 `build_payload`，替换视频来源（`--video-path`/`--video-url`）
3) 改输出/模板：改 `build_prompt`/`format_prompt` 或 `run_iter` 返回值
4) 运行命令：`python run_videoagent_qwen_iter.py ...`，检查输出文本符合预期
