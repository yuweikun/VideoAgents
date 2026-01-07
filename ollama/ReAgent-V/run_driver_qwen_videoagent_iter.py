"""VideoAgent-style iterative loop using the same Qwen3-VL-Flash model/keys as ReAgent.

Workflow:
- Load video + driving signals/gaze from sample_data (same loader as ReAgent demo).
- Round 0: generate a full 5-section report with template prompt (one_shot + INSTRUCTION).
- Critic: score the report with reward prompt (visual/temporal/etc.) -> scalar_reward.
- Iteration: if reward < threshold, send a refine prompt that cites critic feedback and rewrites the report.
- Repeat up to max_iter; stop early once reward >= threshold.
- Final step: enforce strict template format and top-3 actions from provided action set.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from http import HTTPStatus
from pathlib import Path
from typing import Any, Dict, List, Tuple

import dashscope
from dashscope import MultiModalConversation

# Import templates
ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = Path(__file__).resolve().parent
for p in (ROOT, PKG_ROOT, PKG_ROOT / "ReAgent-V", PKG_ROOT / "ReAgent-V" / "ReAgentV_utils"):
    if str(p) not in os.sys.path:
        os.sys.path.append(str(p))

from template import INSTRUCTION, one_shot  # type: ignore
from ReAgentV_utils.prompt_builder.prompt import (  # type: ignore
    eval_reward_prompt_template,
)

SAMPLE_ROOT = ROOT / "sample_data"

DEFAULT_API_KEY = os.getenv("QWEN_API_KEY", "sk-1ccb9f7d365747dca5560308f7854210")
DEFAULT_DASHSCOPE_KEY = os.getenv("DASHSCOPE_API_KEY", DEFAULT_API_KEY)
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


def to_file_path(path: Path) -> str:
    return str(path.resolve())


def qwen_chat(prompt: str, video_uri: str, max_tokens: int = 1200, temperature: float = 0.3) -> str:
    dashscope.api_key = DEFAULT_DASHSCOPE_KEY
    resp = MultiModalConversation.call(
        model=DEFAULT_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"text": prompt},
                    {"video": video_uri},
                ],
            }
        ],
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


def parse_reward(eval_report: str) -> float:
    # Strict JSON first
    try:
        data = json.loads(eval_report)
        val = data.get("scalar_reward")
        if val is None:
            val = data.get("total_score")
        return float(val)
    except Exception:
        pass
    # Regex fallback: look for scalar_reward / total_score numbers anywhere in text
    for key in ("scalar_reward", "total_score"):
        m = re.search(rf'"{key}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', eval_report)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                continue
    return 0.0


def run_iterative(video_path: Path, payload: Dict[str, Any], reward_threshold: float, max_iter: int) -> Dict[str, Any]:
    video_uri = to_file_path(video_path)
    question = build_prompt(payload)

    history: List[Dict[str, Any]] = []

    draft = qwen_chat(question, video_uri=video_uri, max_tokens=900)
    eval_prompt = (
        eval_reward_prompt_template.format(
            question=question,
            context=json.dumps(payload, ensure_ascii=False, indent=2),
            initial_answer=draft,
        )
        + "\nReturn ONLY the JSON object, no extra text."
    )
    eval_report = qwen_chat(eval_prompt, video_uri=video_uri, max_tokens=700)
    reward = parse_reward(eval_report)
    history.append({"answer": draft, "eval_report": eval_report, "reward": reward})

    current_answer = draft
    current_eval = eval_report
    current_reward = reward

    for step in range(1, max_iter + 1):
        if current_reward >= reward_threshold:
            break
        refine_prompt = (
            "You are an agent improving your previous driving evaluation report.\n"
            "Inputs:\n"
            f"- Original instruction + examples:\n{question}\n"
            f"- Previous report:\n{current_answer}\n"
            f"- Critic feedback (with scores):\n{current_eval}\n\n"
            "Task: Rewrite the report to fix the issues highlighted by the critic. "
            "Keep exactly five headings (Scene Description, Driver's Attention, Human-Machine Interaction, "
            "Evaluation & Suggestions, Recommended Actions). "
            "Recommended Actions must list the top 3 items chosen from this set only: "
            f"{payload.get('action', [])}. "
            "Be concise and evidence-grounded."
        )
        improved = qwen_chat(refine_prompt, video_uri=video_uri, max_tokens=800, temperature=0.2)

    eval_prompt = (
        eval_reward_prompt_template.format(
            question=question,
            context=json.dumps(payload, ensure_ascii=False, indent=2),
            initial_answer=improved,
        )
        + "\nReturn ONLY the JSON object, no extra text."
    )
        eval_report = qwen_chat(eval_prompt, video_uri=video_uri, max_tokens=700)
        reward = parse_reward(eval_report)
        history.append({"answer": improved, "eval_report": eval_report, "reward": reward})

        current_answer, current_eval, current_reward = improved, eval_report, reward

    # Final format enforcement
    format_prompt = (
        "Rewrite the final driver evaluation so it strictly follows the template with five headings "
        "(Scene Description, Driver's Attention, Human-Machine Interaction, Evaluation & Suggestions, "
        "Recommended Actions). Use only the provided action set and pick the top three actions, ranked. "
        "Keep concise, avoid extra lists outside the actions section.\n"
        f"Action set: {payload.get('action', [])}\n"
        f"Draft to fix:\n{current_answer}\n"
        "If any section is missing, add it based on available information."
    )
    formatted_answer = qwen_chat(format_prompt, video_uri=video_uri, max_tokens=600, temperature=0.2)

    return {
        "video_uri": video_uri,
        "history": history,
        "final_formatted": formatted_answer,
        "reward_threshold": reward_threshold,
        "max_iter": max_iter,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VideoAgent-style iterative refinement with Qwen3-VL-Flash.")
    parser.add_argument("--folder", default="01-1", help="Sample folder name.")
    parser.add_argument("--file-stem", default="start_at_min02sec03", help="File stem without extension.")
    parser.add_argument("--video-dir", default=str(SAMPLE_ROOT / "01-1"), help="Directory containing the video file.")
    parser.add_argument("--reward-threshold", type=float, default=7.0, help="Scalar reward threshold to stop refinement.")
    parser.add_argument("--max-iter", type=int, default=2, help="Max refinement iterations after the first draft.")
    parser.add_argument("--output", default="", help="If set, save JSON result to this file (UTF-8).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = Path(args.video_dir) / f"{args.file_stem}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    payload = build_payload(args.folder, args.file_stem)
    result = run_iterative(video_path, payload, reward_threshold=args.reward_threshold, max_iter=args.max_iter)
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
