"""Driver behavior evaluation pipeline built on a lightweight ReAgent-style loop.

This script ingests sample_data signals, formats them with template.py, and
runs a draft→critique→rewrite loop with Qwen3-VL-Flash."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI

# Ensure template.py (at repo root) is importable
ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from template import INSTRUCTION, one_shot  # noqa: E402


DEFAULT_MODEL = "qwen3-vl-flash"
DEFAULT_API_KEY = os.getenv("QWEN_API_KEY", "sk-1ccb9f7d365747dca5560308f7854210")
DEFAULT_BASE_URL = os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")


@dataclass
class QwenClientConfig:
    api_key: str = DEFAULT_API_KEY
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    temperature: float = 0.3
    max_tokens: int = 900


class QwenClient:
    def __init__(self, config: QwenClientConfig | None = None):
        self.config = config or QwenClientConfig()
        self._client = OpenAI(api_key=self.config.api_key, base_url=self.config.base_url)

    def complete(self, prompt: str, temperature: float | None = None) -> str:
        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature if temperature is not None else self.config.temperature,
            max_tokens=self.config.max_tokens,
        )
        return response.choices[0].message.content.strip()


class DriverReAgent:
    """Minimal critique-and-rewrite loop tailored to the driver evaluation task."""

    def __init__(self, llm: QwenClient | None = None):
        self.llm = llm or QwenClient()

    @staticmethod
    def build_payload_prompt(payload: Dict[str, Any]) -> str:
        instruction = INSTRUCTION.format(**payload)
        return f"{one_shot}\n\n{instruction}"

    @staticmethod
    def build_critique_prompt(payload: Dict[str, Any], draft: str) -> str:
        return (
            "You are a ReAgent critic. Validate the draft report against the template.\n"
            f"Action set: {payload['action']}\n"
            f"Inputs JSON: {json.dumps(payload, ensure_ascii=False)}\n"
            "Check: 1) all five sections present; 2) action choices come from the action set "
            "and include exactly three ranked items; 3) language is concise and structured.\n"
            "Return JSON with fields: "
            "{" "\"pass\": bool, \"issues\": [\"...\"], \"missing_sections\": [\"...\"]" "}.\n"
            f"Draft:\n{draft}"
        )

    @staticmethod
    def build_revision_prompt(
        payload: Dict[str, Any], draft: str, critique: str
    ) -> str:
        return (
            "Rewrite the report so it strictly follows the driver-evaluation template. "
            "Use the original inputs, the draft, and the critique feedback to fix issues. "
            "Keep the exact five headings and rank the top three actions from the provided set.\n"
            f"Inputs JSON: {json.dumps(payload, ensure_ascii=False)}\n"
            f"Critique JSON: {critique}\n"
            f"Draft:\n{draft}\n"
            "Now produce the final corrected report."
        )

    def run(self, payload: Dict[str, Any]) -> Dict[str, str]:
        draft = self.llm.complete(self.build_payload_prompt(payload))
        critique = self.llm.complete(self.build_critique_prompt(payload, draft))
        final_answer = self.llm.complete(self.build_revision_prompt(payload, draft, critique))
        return {"draft": draft, "critique": critique, "final_answer": final_answer}


# --------- Data loading helpers ----------
SAMPLE_ROOT = REPO_ROOT / "sample_data"
_DEFAULT_ACTION_POOL = [
    "Focus on traffic",
    "Continue",
    "Prepare takeover",
    "Check mirrors",
    "Decelerate",
    "Engage manual",
    "Observe pedestrians",
    "Reduce speed",
]


def _load_labels() -> List[Dict[str, Any]]:
    with open(SAMPLE_ROOT / "labels.json", "r", encoding="utf-8") as f:
        return json.load(f)


def _find_label(folder: str, file_stem: str) -> Dict[str, Any]:
    target_name = f"{file_stem}.txt"
    for item in _load_labels():
        if item.get("folder_name") == folder and item.get("file_name") == target_name:
            return item
    raise FileNotFoundError(f"label not found for {folder}/{target_name}")


def _load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _sample_series(series: List[float], target_len: int = 30) -> List[float]:
    if not series:
        return []
    if len(series) <= target_len:
        return [round(x, 3) for x in series]
    step = max(1, len(series) // target_len)
    return [round(series[i], 3) for i in range(0, len(series), step)][:target_len]


def build_payload_from_sample(folder: str, file_stem: str) -> Dict[str, Any]:
    dynamics_path = SAMPLE_ROOT / "vehicle_dynamics_data" / f"{folder}.csv"
    gaze_path = SAMPLE_ROOT / "vehicle_gaze_data" / f"{folder}.csv"
    label = _find_label(folder, file_stem)

    dyn_rows = _load_csv_rows(dynamics_path)
    timestamps = [float(r["timestamp"]) for r in dyn_rows]
    speed = [float(r["speed"]) for r in dyn_rows]
    acceleration = [float(r["acceleration"]) for r in dyn_rows]
    steering_angle = [float(r["steering_angle"]) for r in dyn_rows]
    braking = [float(r["brake"]) for r in dyn_rows]

    duration = max(timestamps) - min(timestamps) if timestamps else 0.0
    interval = statistics.median(
        [b - a for a, b in zip(timestamps, timestamps[1:])]
    ) if len(timestamps) > 1 else 0.0

    gaze_rows = _load_csv_rows(gaze_path)
    gaze_series = [
        {"t": float(r["timestamp"]), "object": r.get("class", "unknown")}
        for r in gaze_rows
    ]
    gaze_sampled = gaze_series[:: max(1, len(gaze_series) // 30)] if gaze_series else []

    action_set = list(dict.fromkeys(label.get("recommended_actions", []) + _DEFAULT_ACTION_POOL))

    payload: Dict[str, Any] = {
        "duration": round(duration, 2),
        "interval": round(interval, 3),
        "speed": _sample_series(speed),
        "acceleration": _sample_series(acceleration),
        "steering_angle": _sample_series(steering_angle),
        "braking": _sample_series(braking),
        "object_fixations": gaze_sampled,
        "autonomous_mode": label.get("autonomous_mode", []),
        "action": action_set,
    }
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run driver evaluation with sample_data.")
    parser.add_argument(
        "--folder",
        default="01-1",
        help="Sample folder name (matches labels folder_name).",
    )
    parser.add_argument(
        "--file-stem",
        default="start_at_min02sec03",
        help="Sample file stem (without extension, matches labels file_name).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    payload = build_payload_from_sample(args.folder, args.file_stem)
    agent = DriverReAgent()
    result = agent.run(payload)
    print(json.dumps(result, indent=2, ensure_ascii=False))
