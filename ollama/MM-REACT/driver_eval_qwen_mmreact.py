"""Single-pass driver evaluation using Qwen3-VL (similar to ReAgent/VideoAgent pipelines).

Inputs:
- sample_data videos (mp4) + vehicle_dynamics_data/*.csv + vehicle_gaze_data/*.csv + labels.json
- template.py one_shot + INSTRUCTION for the 5-section report.

Output:
- JSON with video_uri, payload, draft_report.

Note: Uses DashScope MultiModalConversation; requires QWEN_API_KEY or DASHSCOPE_API_KEY.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

import imageio.v3 as iio
import requests
from openai import OpenAI

# Allow importing template.py at repo root
ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
for p in (ROOT, REPO_ROOT):
    if str(p) not in os.sys.path:
        os.sys.path.append(str(p))

from template import INSTRUCTION, one_shot  # type: ignore

SAMPLE_ROOT = REPO_ROOT / "sample_data"
DEFAULT_MODEL = "qwen3-vl:2b"
DEFAULT_BASE_URL = "http://localhost:11434/v1"
DEFAULT_NUM_FRAMES = 4

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


def download_video(url: str, tmp_dir: Path) -> Path:
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    tmp_path = tmp_dir / f"video_{uuid.uuid4().hex}.mp4"
    with tmp_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return tmp_path


def _save_frame(frame, tmp_dir: Path, idx: int) -> Path:
    from PIL import Image
    import io

    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    path = tmp_dir / f"frame_{idx}.jpg"
    path.write_bytes(buf.getvalue())
    return path


def sample_frames(video_path: Path, tmp_dir: Path, num_frames: int) -> List[Path]:
    frames: List[Path] = []
    try:
        meta = iio.immeta(video_path, plugin="pyav")
        total = int(meta.get("n_frames") or 0)
    except Exception:
        total = 0
    if total <= 0:
        for idx, frame in enumerate(iio.imiter(video_path, plugin="pyav")):
            if idx >= num_frames:
                break
            frames.append(_save_frame(frame, tmp_dir, idx))
        return frames

    if num_frames <= 1:
        targets = {0}
    else:
        step = (total - 1) / (num_frames - 1)
        targets = {int(round(i * step)) for i in range(num_frames)}

    for idx, frame in enumerate(iio.imiter(video_path, plugin="pyav")):
        if idx in targets:
            frames.append(_save_frame(frame, tmp_dir, idx))
        if len(frames) >= num_frames:
            break
    return frames


def qwen_chat(
    client: OpenAI,
    model: str,
    prompt: str,
    images: List[Path],
    max_tokens: int = 900,
    temperature: float = 0.2,
) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(p) for p in images],
            }
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


def run_once(
    video_path: Path,
    payload: Dict[str, Any],
    model: str,
    base_url: str,
    num_frames: int,
) -> Dict[str, Any]:
    prompt = build_prompt(payload)
    video_uri = str(video_path.resolve())
    client = OpenAI(api_key="ollama", base_url=base_url)
    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        frames = sample_frames(video_path, tmp_dir, num_frames)
        if not frames:
            raise RuntimeError("No frames sampled; check video or imageio[pyav].")
        draft = qwen_chat(client, model, prompt, images=frames)
    return {"video_uri": video_uri, "payload": payload, "draft_report": draft}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Driver evaluation via local Ollama (single pass, MM-REACT adaptation).")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--video-path", help="Local video path (mp4).")
    src.add_argument("--video-url", help="Remote video URL (mp4).")
    parser.add_argument("--folder", default="01-1", help="Sample folder name.")
    parser.add_argument("--file-stem", default="start_at_min02sec03", help="File stem without extension.")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of sampled frames.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama OpenAI-compatible endpoint.")
    parser.add_argument("--output", default="", help="If set, save JSON result to this file (UTF-8).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    default_video = SAMPLE_ROOT / args.folder / f"{args.file_stem}.mp4"

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        if args.video_path:
            video_path = Path(args.video_path)
        elif args.video_url:
            video_path = download_video(args.video_url, tmp_dir)
        else:
            video_path = default_video

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        payload = build_payload(args.folder, args.file_stem)
        result = run_once(video_path, payload, model=args.model, base_url=args.base_url, num_frames=args.num_frames)
    if args.output:
        Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
