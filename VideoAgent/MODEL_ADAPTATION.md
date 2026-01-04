# VideoAgent 方法概览（arXiv:2403.10517）
- 核心思想：LLM 作为中枢代理，面对长视频问答，先用少量均匀抽帧作答并自评不确定性，再按需“计划→检索更关键帧→重答”，以少帧高效完成长视频理解。
- 工具链（论文版）：多模态工具（字幕/检测/描述等）供 LLM 调用获取更多视觉证据；本仓库简化为预存 caption + CLIP 相似度帧检索。
- 流程（对应代码简化版）：初次采样 5 帧 → LLM 作答+自评 → 若置信度不足，按时间段生成候选描述并用 CLIP 检索补帧 → 迭代一次或两次 → 最终输出答案（或在我们改造的脚本中输出驾驶报告）。

# 我们的脚本（驾驶评估，Qwen3-VL）
- 入口：`run_videoagent_qwen_iter.py`（与原 `main.py` 的多选 QA 流程独立）。
- 功能：多轮生成驾驶评估报告（5 段），批判打分后若分数低则重写，最后强制格式化。
- 输入：`sample_data` 中的视频（mp4）+ 车辆/注视信号 CSV + `labels.json`，以及根目录 `template.py` 的 one_shot + INSTRUCTION。
- 模型：DashScope `qwen3-vl-flash`（用 `QWEN_API_KEY` / `DASHSCOPE_API_KEY`）。

# 如何运行
```bash
$env:QWEN_API_KEY="your_key"
$env:DASHSCOPE_API_KEY=$env:QWEN_API_KEY
cd "C:\Program Files\code\VideoAgent"
python run_videoagent_qwen_iter.py --folder 01-1 --file-stem start_at_min02sec03 --output out.json
```
参数：`--reward-threshold` (默认 7.0)，`--max-iter` (默认 2)，`--video-dir` 如需自定义视频路径。

# 更换模型
- 在 `run_videoagent_qwen_iter.py` 的 `qwen_chat`/`DEFAULT_MODEL` 部分改为你的模型与 SDK；调整消息格式、鉴权方式、max_tokens 等。
- 环境变量按需更换（若不用 DashScope）。

# 修改输入
- `build_payload`：读取 CSV、labels、动作集；若输入格式变动，在此重写。
- 视频路径由 `--folder`、`--file-stem`、`--video-dir` 决定；如输入为其他模态，调整这里及 prompt 中的描述。

# 修改输出/提示词
- Prompt 位置：`build_prompt`（初稿）、`eval_reward_prompt_template` 调用处（批判），`refine_prompt`（重写），`format_prompt`（最终格式）。
- 输出结构：返回 `history`（各轮答案/评估/分数）和 `final_formatted`。如需保存其他字段，在 `run_iter` 返回值中添加。
- 若模型不输出纯 JSON 评分，可放宽或调整 `parse_reward` 的解析逻辑。

# 原版多选 QA（仅参考）
- 入口 `main.py`：等距抽帧+CLIP 检索+LLM 作答，自评置信度，多选索引输出。若无需此流程，可忽略。

# 最短改动清单
1) 换模型：改 `qwen_chat`/`DEFAULT_MODEL`，配置鉴权。
2) 改输入：重写 `build_payload`，设置视频/信号来源。
3) 改输出/模板：改各 prompt，调整返回字段。
4) 运行命令：`python run_videoagent_qwen_iter.py ...`，检查生成的 JSON 符合预期。 
