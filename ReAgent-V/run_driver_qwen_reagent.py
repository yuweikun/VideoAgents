"""ReAgent-style critique/reflection loop using DashScope SDK with local video path input.

Pipeline:
1) Load sample_data signals (dynamics + gaze + labels) and fill template prompt (one_shot + INSTRUCTION).
2) Call Qwen3-VL-Flash via DashScope MultiModalConversation, passing local video path (file://).
3) Critique + three reflections + meta fusion, all using the same video input.
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

# Ensure template / utils importable
ROOT = Path(__file__).resolve().parent.parent
PKG_ROOT = Path(__file__).resolve().parent
for p in (ROOT, PKG_ROOT, PKG_ROOT / "ReAgent-V", PKG_ROOT / "ReAgentV_utils"):
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
    # Use absolute local path for DashScope SDK
    return str(path.resolve())


def qwen_chat(prompt: str, video_uri: str, max_tokens: int = 1200) -> str:
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
        temperature=0.3,
        max_tokens=max_tokens,
    )
    if resp.status_code != HTTPStatus.OK:
        raise RuntimeError(f"DashScope call failed: {resp}")
    # Response content is a list of {type,text} dicts; take first text
    outputs = resp.output["choices"][0]["message"]["content"]
    # Some SDK variants return items without explicit "type" key
    if isinstance(outputs, list):
        for item in outputs:
            if isinstance(item, dict):
                if item.get("type") == "text" or ("text" in item and "type" not in item):
                    return str(item.get("text", "")).strip()
        return str(outputs)
    return str(outputs)


def run_pipeline(video_path: Path, payload: Dict[str, Any]) -> Dict[str, Any]:
    video_uri = to_file_path(video_path)
    question = build_prompt(payload)

    draft = qwen_chat(question, video_uri=video_uri, max_tokens=900)

    context_json = json.dumps(payload, ensure_ascii=False, indent=2)
    eval_prompt = eval_reward_prompt_template.format(
        question=question,
        context=context_json,
        initial_answer=draft,
    )
    eval_report = qwen_chat(eval_prompt, video_uri=video_uri, max_tokens=700)

    conservative_prompt = conservative_template_str.replace("$text", question).replace("$answer", draft).replace(
        "$eval_report", eval_report
    )
    neutral_prompt = neutral_template_str.replace("$text", question).replace("$answer", draft).replace(
        "$eval_report", eval_report
    )
    aggressive_prompt = aggressive_template_str.replace("$text", question).replace("$answer", draft).replace(
        "$eval_report", eval_report
    )

    cons_res = qwen_chat(conservative_prompt, video_uri=video_uri, max_tokens=400)
    neut_res = qwen_chat(neutral_prompt, video_uri=video_uri, max_tokens=400)
    aggr_res = qwen_chat(aggressive_prompt, video_uri=video_uri, max_tokens=400)

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
    final_answer = qwen_chat(meta_prompt, video_uri=video_uri, max_tokens=400)

    # Enforce template formatting with top-3 actions from provided action set
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
    formatted_answer = qwen_chat(format_prompt, video_uri=video_uri, max_tokens=600)

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
    parser = argparse.ArgumentParser(description="Driver evaluation via Qwen3-VL-Flash with ReAgent-style reflection (local file URI).")
    parser.add_argument("--folder", default="01-1", help="Sample folder name.")
    parser.add_argument("--file-stem", default="start_at_min02sec03", help="File stem without extension.")
    parser.add_argument("--video-dir", default=str(SAMPLE_ROOT / "01-1"), help="Directory containing the video file.")
    parser.add_argument("--output", default="", help="If set, save JSON result to this file (UTF-8).")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    video_path = Path(args.video_dir) / f"{args.file_stem}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    payload = build_payload(args.folder, args.file_stem)
    result = run_pipeline(video_path, payload)
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2))
