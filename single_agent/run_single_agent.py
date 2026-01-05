"""Single-agent (LangChain) wrapper to generate a driving report from video + signals.

Inputs:
- sample_data videos (mp4) + vehicle_dynamics_data/*.csv + vehicle_gaze_data/*.csv + labels.json
- root-level template.py (one_shot + INSTRUCTION) for structured 5-section report.

Behavior:
- Builds a prompt from template and payload.
- Defines a LangChain Tool that calls Qwen3-VL (DashScope) for multimodal generation.
- Runs a zero-shot REACT agent to invoke the tool once and return the report.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List

import dashscope
from dashscope import MultiModalConversation
from langchain.agents import Tool, initialize_agent
from langchain.chat_models import ChatOpenAI

# root for template import
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in os.sys.path:
    os.sys.path.append(str(ROOT))

from template import INSTRUCTION, one_shot  # type: ignore

SAMPLE_ROOT = ROOT / "sample_data"

DEFAULT_API_KEY = os.getenv("QWEN_API_KEY", "sk-1ccb9f7d365747dca5560308f7854210")
DEFAULT_DASHSCOPE_KEY = os.getenv("DASHSCOPE_API_KEY", DEFAULT_API_KEY)
# text LLM for tool orchestration (DashScope OpenAI-compatible endpoint)
DEFAULT_TEXT_MODEL = "qwen-plus"  # adjust if needed
DEFAULT_VIDEO_MODEL = "qwen3-vl-flash"

DEFAULT_ACTION_POOL = [
    "Focus on traffic",
    "Continue",
    "Prepare takeover",
    "Check mirrors",
    "Decelerate",
    "Engage manual",
    "Observe pedestrians",
    "Reduce speed",
]


def load_labels() -> List[Dict[str, Any]]:
    with open(SAMPLE_ROOT / "labels.json", "r", encoding="utf-8") as f:
        return json.load(f)


def find_label(folder: str, file_stem: str) -> Dict[str, Any]:
    target = f"{file_stem}.txt"
    for item in load_labels():
        if item.get("folder_name") == folder and item.get("file_name") == target:
            return item
    raise FileNotFoundError(f"label not found for {folder}/{target}")


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def sample_series(series: List[float], target_len: int = 30) -> List[float]:
    if not series:
        return []
    if len(series) <= target_len:
        return [round(x, 3) for x in series]
    step = max(1, len(series) // target_len)
    return [round(series[i], 3) for i in range(0, len(series), step)][:target_len]


def build_payload(folder: str, file_stem: str) -> Dict[str, Any]:
    dyn_path = SAMPLE_ROOT / "vehicle_dynamics_data" / f"{folder}.csv"
    gaze_path = SAMPLE_ROOT / "vehicle_gaze_data" / f"{folder}.csv"
    label = find_label(folder, file_stem)

    dyn_rows = load_csv_rows(dyn_path)
    timestamps = [float(r["timestamp"]) for r in dyn_rows]
    speed = [float(r["speed"]) for r in dyn_rows]
    acceleration = [float(r["acceleration"]) for r in dyn_rows]
    steering_angle = [float(r["steering_angle"]) for r in dyn_rows]
    braking = [float(r["brake"]) for r in dyn_rows]

    duration = max(timestamps) - min(timestamps) if timestamps else 0.0
    interval = (
        (sum(b - a for a, b in zip(timestamps, timestamps[1:])) / (len(timestamps) - 1))
        if len(timestamps) > 1
        else 0.0
    )

    gaze_rows = load_csv_rows(gaze_path)
    gaze_series = [{"t": float(r["timestamp"]), "object": r.get("class", "unknown")} for r in gaze_rows]
    gaze_sampled = gaze_series[:: max(1, len(gaze_series) // 30)] if gaze_series else []

    action_set = list(dict.fromkeys(label.get("recommended_actions", []) + DEFAULT_ACTION_POOL))

    payload: Dict[str, Any] = {
        "duration": round(duration, 2),
        "interval": round(interval, 3),
        "speed": sample_series(speed),
        "acceleration": sample_series(acceleration),
        "steering_angle": sample_series(steering_angle),
        "braking": sample_series(braking),
        "object_fixations": gaze_sampled,
        "autonomous_mode": label.get("autonomous_mode", []),
        "action": action_set,
    }
    return payload


def build_prompt(payload: Dict[str, Any], extra_text: str = "") -> str:
    instruction = INSTRUCTION.format(**payload)
    prompt = f"{one_shot}\n\n{instruction}"
    if extra_text:
        prompt += f"\n\nAdditional notes:\n{extra_text}"
    return prompt


def qwen_video_report(prompt: str, video_uri: str, max_tokens: int = 900, temperature: float = 0.3) -> str:
    dashscope.api_key = DEFAULT_DASHSCOPE_KEY
    resp = MultiModalConversation.call(
        model=DEFAULT_VIDEO_MODEL,
        messages=[{"role": "user", "content": [{"text": prompt}, {"video": video_uri}]}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"DashScope call failed: {resp}")
    outputs = resp.output["choices"][0]["message"]["content"]
    if isinstance(outputs, list):
        for item in outputs:
            if isinstance(item, dict):
                if item.get("type") == "text" or ("text" in item and "type" not in item):
                    return str(item.get("text", "")).strip()
        return str(outputs)
    return str(outputs)


def run_agent(video_path: Path, payload: Dict[str, Any], extra_text: str = "") -> Dict[str, Any]:
    prompt = build_prompt(payload, extra_text=extra_text)
    video_uri = str(video_path.resolve())

    # Define tool that calls the multimodal model
    def _tool_func(_: str) -> str:
        return qwen_video_report(prompt, video_uri)

    tools = [
        Tool(
            name="GenerateDrivingReport",
            func=_tool_func,
            description="Generate a 5-section driving co-pilot report using the provided video and signals.",
        )
    ]

    # Text LLM for tool selection; uses DashScope OpenAI-compatible endpoint
    llm = ChatOpenAI(
        api_key=DEFAULT_DASHSCOPE_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model_name=DEFAULT_TEXT_MODEL,
        temperature=0,
    )

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent="zero-shot-react-description",
        verbose=False,
        max_iterations=2,
    )

    result_text = agent.run("Generate the driving report using GenerateDrivingReport.")
    return {"video_uri": video_uri, "payload": payload, "report": result_text}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-agent driving report via LangChain + Qwen3-VL.")
    parser.add_argument("--folder", default="01-1", help="Sample folder name.")
    parser.add_argument("--file-stem", default="start_at_min02sec03", help="File stem without extension.")
    parser.add_argument("--video-dir", default=str(SAMPLE_ROOT / "01-1"), help="Directory containing the video file.")
    parser.add_argument("--extra-text", default="", help="Optional extra text context appended to the prompt.")
    parser.add_argument(
        "--output",
        default="",
        help="Path to save JSON result (UTF-8). If empty, will save under single_agent/outputs/<file_stem>.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = Path(args.video_dir) / f"{args.file_stem}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    payload = build_payload(args.folder, args.file_stem)
    result = run_agent(video_path, payload, extra_text=args.extra_text)

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(__file__).resolve().parent / "outputs"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{args.file_stem}.json"

    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
