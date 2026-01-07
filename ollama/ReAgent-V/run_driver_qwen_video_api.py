"""Driver evaluation via Qwen3-VL-Flash API with direct video upload (no local weights).

Steps:
1) Load sample_data signals (dynamics + gaze + labels) to fill the template.
2) Build the prompt (one_shot + INSTRUCTION) with those signals.
3) Call Qwen3-VL-Flash using the OpenAI-compatible API, attaching the raw video.

Note: Requires network access to DashScope and a valid API key.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

from template import INSTRUCTION, one_shot

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_ROOT = ROOT / "sample_data"

DEFAULT_API_KEY = os.getenv("QWEN_API_KEY", "sk-1ccb9f7d365747dca5560308f7854210")
DEFAULT_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")
DEFAULT_MODEL = "qwen3-vl-flash"

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


def build_prompt(payload: Dict[str, Any]) -> str:
    instruction = INSTRUCTION.format(**payload)
    return f"{one_shot}\n\n{instruction}"


def encode_video_b64(path: Path) -> str:
    data = path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


def qwen_video_completion(prompt: str, video_b64: str, model: str = DEFAULT_MODEL) -> str:
    client = OpenAI(api_key=DEFAULT_API_KEY, base_url=DEFAULT_BASE_URL)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "input_video", "input_video": video_b64},
                ],
            }
        ],
        temperature=0.3,
        max_tokens=1200,
    )
    return resp.choices[0].message.content.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Driver evaluation via Qwen3-VL-Flash video API.")
    parser.add_argument("--folder", default="01-1", help="Sample folder name.")
    parser.add_argument("--file-stem", default="start_at_min02sec03", help="File stem without extension.")
    parser.add_argument("--video-dir", default=str(SAMPLE_ROOT / "01-1"), help="Directory containing the video file.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = Path(args.video_dir) / f"{args.file_stem}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    payload = build_payload(args.folder, args.file_stem)
    prompt = build_prompt(payload)
    video_b64 = encode_video_b64(video_path)
    result = qwen_video_completion(prompt, video_b64)
    print(result)
