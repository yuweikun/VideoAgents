"""
本地/远程视频自动采样若干帧 + 固定模板提示（template.py），调用本地 Ollama（OpenAI 兼容，默认 qwen3-vl:2b）。
流程：
- 读取根目录 template.py 的 one_shot + INSTRUCTION，填入 sample_data 下的车辆信号。
- 视频来源优先本地 --video-path；否则 --video-url 下载；都未提供时默认 sample_data/{folder}/{file_stem}.mp4。
- 用 imageio[pyav] 采样若干帧保存为临时 jpg，作为 images 传给 Ollama。

示例：
  python ollama_single/run_ollama_single.py --video-path sample_data/01-1/start_at_min02sec03.mp4 --folder 01-1 --file-stem start_at_min02sec03 --output ollama_single/outputs/out.txt

前提：
- 本地已运行 `ollama serve`，并已 pull 对应模型（默认 qwen3-vl:2b）。
- 根目录存在 template.py，sample_data 下有对应的 csv/label。
"""

from __future__ import annotations

import argparse
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict, List

import imageio.v3 as iio
import requests
from openai import OpenAI

# 根目录导入模板
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in os.sys.path:
    import sys

    sys.path.append(str(ROOT))
from template import INSTRUCTION, one_shot  # type: ignore

SAMPLE_ROOT = ROOT / "sample_data"


def download_video(url: str, tmp_dir: Path) -> Path:
    resp = requests.get(url, stream=True, timeout=30)
    resp.raise_for_status()
    tmp_path = tmp_dir / f"video_{uuid.uuid4().hex}.mp4"
    with tmp_path.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return tmp_path


def save_frame(frame, tmp_dir: Path, idx: int) -> Path:
    from PIL import Image
    import io

    img = Image.fromarray(frame)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    path = tmp_dir / f"frame_{idx}.jpg"
    path.write_bytes(buf.getvalue())
    return path


def sample_frames(video_path: Path, tmp_dir: Path, num_frames: int = 4) -> List[Path]:
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
            frames.append(save_frame(frame, tmp_dir, idx))
    else:
        import numpy as np

        targets = set(np.linspace(0, total - 1, num_frames, dtype=int).tolist())
        for idx, frame in enumerate(iio.imiter(video_path, plugin="pyav")):
            if idx in targets:
                frames.append(save_frame(frame, tmp_dir, idx))
            if len(frames) >= num_frames:
                break
    return frames


def load_payload(folder: str, file_stem: str) -> Dict[str, Any]:
    import json
    import csv

    labels = json.load(open(SAMPLE_ROOT / "labels.json", "r", encoding="utf-8"))
    label = next(
        (item for item in labels if item.get("folder_name") == folder and item.get("file_name") == f"{file_stem}.txt"),
        None,
    )
    if label is None:
        raise FileNotFoundError(f"label not found for {folder}/{file_stem}.txt")

    def read_csv(path: Path):
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    dyn_rows = read_csv(SAMPLE_ROOT / "vehicle_dynamics_data" / f"{folder}.csv")
    gaze_rows = read_csv(SAMPLE_ROOT / "vehicle_gaze_data" / f"{folder}.csv")

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
    gaze_series = [{"t": float(r["timestamp"]), "object": r.get("class", "unknown")} for r in gaze_rows]
    gaze_sampled = gaze_series[:: max(1, len(gaze_series) // 30)] if gaze_series else []
    action_set = list(dict.fromkeys(label.get("recommended_actions", [])))

    def sample_series(series: List[float], target_len: int = 30) -> List[float]:
        if not series:
            return []
        if len(series) <= target_len:
            return [round(x, 3) for x in series]
        step = max(1, len(series) // target_len)
        return [round(series[i], 3) for i in range(0, len(series), step)][:target_len]

    return {
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


def main():
    parser = argparse.ArgumentParser(description="本地/远程视频采样 + 固定模板，调用本地 Ollama qwen3-vl:2b")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--video-path", help="本地视频路径 (mp4)，优先使用")
    src.add_argument("--video-url", help="可访问的远程视频URL (mp4)")
    parser.add_argument("--folder", default="01-1", help="sample_data 文件夹名（读取信号/labels）")
    parser.add_argument("--file-stem", default="start_at_min02sec03", help="文件名前缀（视频/信号/label）")
    parser.add_argument("--num-frames", type=int, default=4, help="采样帧数，默认4")
    parser.add_argument("--model", default="qwen3-vl:2b", help="Ollama 模型名")
    parser.add_argument("--base-url", default="http://localhost:11434/v1", help="Ollama OpenAI 兼容端点")
    parser.add_argument("--output", default="", help="如需保存结果到文件，指定路径")
    args = parser.parse_args()

    payload = load_payload(args.folder, args.file_stem)
    prompt = f"{one_shot}\n\n{INSTRUCTION.format(**payload)}"

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
            raise FileNotFoundError(f"未找到视频文件：{video_path}")

        frames = sample_frames(video_path, tmp_dir, num_frames=args.num_frames)
        if not frames:
            raise RuntimeError("未采样到帧，请检查视频或依赖（imageio[pyav]）")

        client = OpenAI(api_key="ollama", base_url=args.base_url)
        completion = client.chat.completions.create(
            model=args.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                    "images": [str(p) for p in frames],
                }
            ],
        )
        result = completion.choices[0].message.content
        print(result)
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(result, encoding="utf-8")


if __name__ == "__main__":
    main()
