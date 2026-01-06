# Ollama 单轮生成（qwen3-vl-8b）说明

## 方法概述
- 使用本地 Ollama 的 OpenAI 兼容接口（默认 `http://localhost:11434/v1`，模型 `qwen3-vl-8b`）执行单轮文本生成。
- 提示来自根目录 `template.py`（one_shot + INSTRUCTION），仅使用文本（视频不上传），输入包含车辆/注视信号和推荐动作。
- 输出：JSON 包含 `payload` 与生成的 `report`。

## 脚本入口
- `ollama_single/run_ollama_single.py`
  - 读取 `sample_data` 下的信号 CSV 与 `labels.json`，采样信号，组合动作集。
  - 构造模板提示（可 `--extra-text`）。
  - 调用本地 Ollama（OpenAI 兼容）生成 5 段报告。
  - 默认保存到 `ollama_single/outputs/<file_stem>.json`。

## 运行示例
确保本地已 `ollama pull qwen3-vl-8b` 并 `ollama serve`：
```powershell
cd "C:\Program Files\code"
python ollama_single/run_ollama_single.py --folder 01-1 --file-stem start_at_min02sec03 --output ollama_single/outputs/output_ollama.json
```
可选参数：`--extra-text`（附加提示），`--output`（自定义保存路径）。

## 更换模型/接口
- 修改 `run_ollama_single.py` 的 `DEFAULT_BASE_URL`、`DEFAULT_MODEL`、`DEFAULT_API_KEY` 为你的 Ollama 配置或其他 OpenAI 兼容服务。
- 如需多模态上传，需改写 `ollama_call` 以发送图像/视频（当前为文本-only）。

## 修改输入
- 重写 `build_payload` 以适配你的数据（CSV/JSON/其他信号）。
- 文件定位由 `--folder`、`--file-stem` 决定，可在 payload 中增删字段并在 prompt 中引用。

## 修改输出/提示词
- 提示构造在 `build_prompt`（源自 `template.py`），可直接改模板或用 `--extra-text` 补充。
- 输出结构：`report` 字段为生成结果；需要更多字段可在返回字典中扩展。

## 注意
- 本脚本为单轮文本生成，不含批判/迭代或多模态上传；适用于快速离线实验。 
