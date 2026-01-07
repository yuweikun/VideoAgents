"""ReAgent-style critique/reflection loop using local Ollama with sampled frames.

Pipeline:
1) Load sample_data signals (dynamics + gaze + labels) and fill template prompt (one_shot + INSTRUCTION).
2) Sample frames from a local/URL video once; reuse frames for all calls.
3) Critique + three reflections + meta fusion; finalize formatted report.
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

# Ensure template / utils importable
ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = ROOT.parent
PKG_ROOT = Path(__file__).resolve().parent
for p in (ROOT, REPO_ROOT, PKG_ROOT, PKG_ROOT / "ReAgent-V", PKG_ROOT / "ReAgentV_utils"):
    if str(p) not in os.sys.path:
        os.sys.path.append(str(p))

from template import INSTRUCTION, one_shot
from ReAgentV_utils.prompt_builder.prompt import (
    eval_reward_prompt_template,
    conservative_template_str,
    neutral_template_str,
    aggressive_template_str,
    meta_agent_prompt_template,
)

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


def ollama_chat(
    client: OpenAI, model: str, prompt: str, images: List[Path], max_tokens: int, temperature: float
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


def run_pipeline(video_path: Path, payload: Dict[str, Any], model: str, base_url: str, num_frames: int) -> Dict[str, Any]:
    video_uri = str(video_path.resolve())
    question = build_prompt(payload)

    client = OpenAI(api_key="ollama", base_url=base_url)

    with tempfile.TemporaryDirectory() as td:
        tmp_dir = Path(td)
        frames = sample_frames(video_path, tmp_dir, num_frames)
        if not frames:
            raise RuntimeError("No frames sampled; check video or imageio[pyav].")

        draft = ollama_chat(client, model, question, frames, max_tokens=900, temperature=0.3)

        context_json = json.dumps(payload, ensure_ascii=False, indent=2)
        eval_prompt = eval_reward_prompt_template.format(
            question=question,
            context=context_json,
            initial_answer=draft,
        )
        eval_report = ollama_chat(client, model, eval_prompt, frames, max_tokens=700, temperature=0.3)

        conservative_prompt = conservative_template_str.replace("$text", question).replace("$answer", draft).replace(
            "$eval_report", eval_report
        )
        neutral_prompt = neutral_template_str.replace("$text", question).replace("$answer", draft).replace(
            "$eval_report", eval_report
        )
        aggressive_prompt = aggressive_template_str.replace("$text", question).replace("$answer", draft).replace(
            "$eval_report", eval_report
        )

        cons_res = ollama_chat(client, model, conservative_prompt, frames, max_tokens=400, temperature=0.3)
        neut_res = ollama_chat(client, model, neutral_prompt, frames, max_tokens=400, temperature=0.3)
        aggr_res = ollama_chat(client, model, aggressive_prompt, frames, max_tokens=400, temperature=0.3)

        try:
            cons = json.loads(cons_res)
            ans_cons = cons.get("final_answer")
            conf_cons = float(cons.get("confidence", 0.0))
        except Exception:
            ans_cons, conf_cons = cons_res, 0.0

        try:
            neut = json.loads(neut_res)
            ans_neut = neut.get("final_answer")
            conf_neut = float(neut.get("confidence", 0.0))
        except Exception:
            ans_neut, conf_neut = neut_res, 0.0

        try:
            aggr = json.loads(aggr_res)
            ans_aggr = aggr.get("final_answer")
            conf_aggr = float(aggr.get("confidence", 0.0))
        except Exception:
            ans_aggr, conf_aggr = aggr_res, 0.0

        from string import Template

        meta_prompt = Template(meta_agent_prompt_template).substitute(
            answer_conservative=ans_cons,
            conf_conservative=conf_cons,
            answer_neutral=ans_neut,
            conf_neutral=conf_neut,
            answer_aggressive=ans_aggr,
            conf_aggressive=conf_aggr,
            text=question,
            initial_answer=draft,
        )
        final_answer = ollama_chat(client, model, meta_prompt, frames, max_tokens=400, temperature=0.3)

        action_set = payload.get("action", [])
        format_prompt = (
            "Rewrite the final driver evaluation so it strictly follows the template with five headings "
            "(Scene Description, Driver's Attention, Human-Machine Interaction, Evaluation & Suggestions, "
            "Recommended Actions). Use only the provided action set and pick the top three actions, ranked. "
            "Keep concise, avoid lists outside the actions section.\n"
            f"Action set: {action_set}\n"
            f"Template draft to fix:\n{final_answer}\n"
            "If any section is missing, add it based on available information."
        )
        formatted_answer = ollama_chat(client, model, format_prompt, frames, max_tokens=600, temperature=0.3)

        return {
            "video_uri": video_uri,
            "draft": draft,
            "eval_report": eval_report,
            "conservative": cons_res,
            "neutral": neut_res,
            "aggressive": aggr_res,
            "final_answer": final_answer,
            "final_formatted": formatted_answer,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Driver evaluation via local Ollama (ReAgent-style).")
    src = parser.add_mutually_exclusive_group(required=False)
    src.add_argument("--video-path", help="Local video path (mp4).")
    src.add_argument("--video-url", help="Remote video URL (mp4).")
    parser.add_argument("--folder", default="01-1", help="Sample folder name.")
    parser.add_argument("--file-stem", default="start_at_min02sec03", help="File stem without extension.")
    parser.add_argument("--num-frames", type=int, default=DEFAULT_NUM_FRAMES, help="Number of sampled frames.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model name.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="Ollama OpenAI-compatible endpoint.")
    parser.add_argument("--output", default="", help="If set, save final report to this file (UTF-8).")
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
        result = run_pipeline(video_path, payload, model=args.model, base_url=args.base_url, num_frames=args.num_frames)
        final_text = result["final_formatted"]
        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(final_text, encoding="utf-8")
        else:
            print(final_text)
