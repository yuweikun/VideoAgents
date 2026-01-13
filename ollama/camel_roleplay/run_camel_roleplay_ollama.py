"""
CAMEL RolePlaying (official multi-role) + finalizer using local Ollama.

Roles:
- Report Writer (assistant): drafts the report from template + frames.
- Safety Reviewer (user role): provides critique/requests.
- Final Editor (separate agent): consolidates draft + critique into final report.

Args match ollama_single:
  --video-path / --video-url / --folder / --file-stem / --num-frames / --model / --base-url / --output
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

# Make template importable from repo root
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
    # Modify prompt here if needed.
    instruction = INSTRUCTION.format(**payload)
    return f"{one_shot}\n\n{instruction}"


def load_frame_images(frame_paths: List[Path]):
    from PIL import Image

    images = []
    for path in frame_paths:
        images.append(Image.open(path))
    return images


def ollama_chat(
    client: OpenAI,
    model: str,
    prompt: str,
    images: List[Path],
    max_tokens: int = 900,
    temperature: float = 0.3,
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


def roleplay_once(prompt: str, frames: List[Path], model: str, base_url: str) -> Dict[str, str]:
    client = OpenAI(api_key="ollama", base_url=base_url)
    writer_prompt = (
        "You are Report Writer. Write a concise 5-section driving report using the provided template.\n\n"
        f"{prompt}"
    )
    draft = ollama_chat(client, model, writer_prompt, frames, max_tokens=900, temperature=0.3)

    reviewer_prompt = (
        "You are Safety Reviewer. Inspect the report and return concise notes for correction, "
        "but do not rewrite the report. Check: five headings present and Recommended Actions lists exactly three ranked items.\n\n"
        f"Report:\n{draft}"
    )
    review = ollama_chat(client, model, reviewer_prompt, frames, max_tokens=400, temperature=0.2)
    return {"draft": draft, "review": review}


def finalize_report(
    draft: str,
    review: str,
    frames: List[Path],
    model: str,
    base_url: str,
) -> str:
    client = OpenAI(api_key="ollama", base_url=base_url)
    final_prompt = (
        "You are Final Editor. Produce the final report in one pass. Keep exactly five headings "
        "(Scene Description, Driver's Attention, Human-Machine Interaction, Evaluation & Suggestions, Recommended Actions). "
        "Recommended Actions must list exactly three ranked items.\n\n"
        f"Reviewer notes:\n{review}\n\nDraft:\n{draft}"
    )
    return ollama_chat(client, model, final_prompt, frames, max_tokens=900, temperature=0.3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CAMEL RolePlaying with local Ollama.")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--video-path", help="Local video path (mp4).")
    src.add_argument("--video-url", help="Remote video URL (mp4).")
    parser.add_argument("--folder", default="01-1", help="Sample folder name.")
    parser.add_argument("--file-stem", default="start_at_min02sec03", help="File stem without extension.")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of sampled frames.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama OpenAI-compatible endpoint.")
    parser.add_argument("--output", default="", help="If set, save result to this file (UTF-8).")
    return parser.parse_args()


def main() -> None:
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
        prompt = build_prompt(payload)
        frames = sample_frames(video_path, tmp_dir, args.num_frames)
        if not frames:
            raise RuntimeError("No frames sampled; check video or imageio[pyav].")

        roleplay_result = roleplay_once(prompt, frames, model=args.model, base_url=args.base_url)
        final_text = finalize_report(
            draft=roleplay_result["draft"],
            review=roleplay_result["review"],
            frames=frames,
            model=args.model,
            base_url=args.base_url,
        )

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(final_text, encoding="utf-8")
        else:
            print(final_text)


if __name__ == "__main__":
    main()
