"""
LangGraph multi-agent driver evaluation with local Ollama (OpenAI compatible).

Agents:
- Drafter: writes the initial report from template prompt + sampled frames.
- Critic: audits structure/actions and returns concise notes (no rewriting).
- Synthesizer: produces the final report using draft + notes (single pass).

Inputs/args match ollama_single:
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
from typing import Any, Dict, List, TypedDict

import imageio.v3 as iio
import requests
from openai import OpenAI
from langgraph.graph import END, StateGraph

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


def ollama_chat(client: OpenAI, model: str, prompt: str, images: List[Path], max_tokens: int = 900) -> str:
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [str(p) for p in images],
            }
        ],
        temperature=0.3,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


class GraphState(TypedDict):
    payload: Dict[str, Any]
    prompt: str
    frames: List[Path]
    draft: str
    critique: str
    final: str


def drafter(state: GraphState) -> GraphState:
    client = state["payload"]["_client"]
    model = state["payload"]["_model"]
    prompt = state["prompt"]
    frames = state["frames"]
    draft = ollama_chat(client, model, prompt, frames, max_tokens=900)
    state["draft"] = draft
    return state


def critic(state: GraphState) -> GraphState:
    client = state["payload"]["_client"]
    model = state["payload"]["_model"]
    frames = state["frames"]
    action_set = state["payload"].get("action", [])
    critique_prompt = (
        "You are a strict reviewer. Inspect the report and return concise notes for correction, "
        "but do not rewrite the report. Check: five required headings "
        "(Scene Description, Driver's Attention, Human-Machine Interaction, Evaluation & Suggestions, Recommended Actions); "
        "Recommended Actions must list exactly three ranked items from this set only: "
        f"{action_set}. Return 3-6 short bullet notes; if everything is correct, return 'PASS'.\n\n"
        f"Report:\n{state['draft']}"
    )
    critique = ollama_chat(client, model, critique_prompt, frames, max_tokens=400)
    state["critique"] = critique
    return state


def synthesizer(state: GraphState) -> GraphState:
    client = state["payload"]["_client"]
    model = state["payload"]["_model"]
    frames = state["frames"]
    action_set = state["payload"].get("action", [])
    fix_prompt = (
        "Produce the final report in one pass. Use the draft and the reviewer notes to fix issues. "
        "Keep exactly five headings (Scene Description, Driver's Attention, Human-Machine Interaction, "
        "Evaluation & Suggestions, Recommended Actions). Recommended Actions must list exactly three ranked items "
        f"from this set only: {action_set}. Keep concise.\n\n"
        f"Reviewer notes:\n{state['critique']}\n\n"
        f"Draft:\n{state['draft']}"
    )
    final = ollama_chat(client, model, fix_prompt, frames, max_tokens=900)
    state["final"] = final
    return state


def build_graph() -> StateGraph:
    graph = StateGraph(GraphState)
    graph.add_node("draft", drafter)
    graph.add_node("critique", critic)
    graph.add_node("finalize", synthesizer)
    graph.set_entry_point("draft")
    graph.add_edge("draft", "critique")
    graph.add_edge("critique", "finalize")
    graph.add_edge("finalize", END)
    return graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LangGraph multi-agent driver evaluation with Ollama.")
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

        client = OpenAI(api_key="ollama", base_url=args.base_url)
        payload["_client"] = client
        payload["_model"] = args.model

        graph = build_graph().compile()
        state: GraphState = {
            "payload": payload,
            "prompt": prompt,
            "frames": frames,
            "draft": "",
            "critique": "",
            "final": "",
        }
        result = graph.invoke(state)
        final = result["final"]

        if args.output:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(final, encoding="utf-8")
        else:
            print(final)


if __name__ == "__main__":
    main()
