"""End-to-end driver evaluation with video + signals using the original ReAgent-V pipeline.

This script:
1) Loads sample_data video and structured signals (dynamics + gaze + labels).
2) Builds the driver-evaluation prompt (one_shot + INSTRUCTION) with those signals.
3) Runs the ReAgent-V flow: frame selection (ECRS) -> tool retrieval -> draft answer ->
   critical questions -> re-retrieval -> evaluation report -> reflective final answer.

Note: ReAgent-V expects model weights (CLIP/Whisper/LLaVA) available locally as configured
in path_dict. Without those weights, the script will not complete inference.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List

from ReAgentV import ReAgentV
from ReAgentV_utils.model_inference.model_inference import llava_inference
from ReAgentV_utils.prompt_builder.prompt import tool_retrieval_prompt_template
from template import INSTRUCTION, one_shot

# ---------- Helpers ----------

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_ROOT = ROOT / "sample_data"

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


def build_question(payload: Dict[str, Any]) -> str:
    instruction = INSTRUCTION.format(**payload)
    return f"{one_shot}\n\n{instruction}"


# ---------- Main pipeline using ReAgent-V ----------


def run_with_video(video_path: Path, payload: Dict[str, Any], path_dict: Dict[str, str]) -> Dict[str, Any]:
    qa_system = ReAgentV.load_default(path_dict)

    question = build_question(payload)

    frames, key_frames, key_indices, max_frames_num, raw_video, video_tensor = qa_system.load_and_sample_video(
        question=question, video_path=str(video_path)
    )

    modal_info, det_top_idx, USE_OCR, USE_ASR, USE_DET = qa_system.retrieve_modal_info(
        video_path=str(video_path),
        question=question,
        frames=key_frames,
        raw_video=raw_video,
        clip_model=qa_system.clip_model,
        clip_processor=qa_system.clip_processor,
    )

    qs = qa_system.build_multimodal_prompt(
        question=question,
        modal_info=modal_info,
        det_top_idx=det_top_idx,
        max_frames_num=max_frames_num,
        USE_DET=USE_DET,
        USE_ASR=USE_ASR,
        USE_OCR=USE_OCR,
    )

    initial_answer = llava_inference(qs, video_tensor)

    critique_questions = qa_system.generate_critical_questions(question, initial_answer, modal_info, video_tensor)

    updated_infos = {}
    for cq in critique_questions:
        tool_selection_prompt = tool_retrieval_prompt_template.format(question=cq)
        response_list = llava_inference(tool_selection_prompt, video=None)

        use_asr = "ASR" in response_list
        use_ocr = "OCR" in response_list
        use_det = "Scene Graph" in response_list

        new_modal_info, new_det_top_idx, _, _, _ = qa_system.retrieve_modal_info(
            video_path=str(video_path),
            question=cq,
            frames=key_frames,
            raw_video=raw_video,
            clip_model=qa_system.clip_model,
            clip_processor=qa_system.clip_processor,
        )
        updated_infos[cq] = new_modal_info

    context_infos = {question: modal_info, **updated_infos}
    context_str = json.dumps(context_infos[question], indent=2)

    eval_report = qa_system.generate_eval_report(
        question=question,
        context_info=context_str,
        initial_answer=initial_answer,
        video=video_tensor,
    )

    final_answer = qa_system.get_reflective_final_answer(
        question=question, initial_answer=initial_answer, eval_report=eval_report, video=video_tensor
    )

    return {
        "prompt": qs,
        "initial_answer": initial_answer,
        "critique_questions": critique_questions,
        "context_infos": context_infos,
        "eval_report": eval_report,
        "final_answer": final_answer,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Driver evaluation with video + signals using ReAgent-V.")
    parser.add_argument("--folder", default="01-1", help="Sample folder name.")
    parser.add_argument("--file-stem", default="start_at_min02sec03", help="File stem in the folder (without extension).")
    parser.add_argument("--video-dir", default=str(SAMPLE_ROOT / "01-1"), help="Directory containing the video file.")
    parser.add_argument(
        "--model-root",
        default="models",
        help="Root where model weights are stored (CLIP/Whisper/LLaVA). Update to your local paths.",
    )
    return parser.parse_args()


def build_path_dict(model_root: Path) -> Dict[str, str]:
    # Update these paths to point to your local model snapshots.
    return {
        "clip_model_path": str(model_root / "clip-vit-large-patch14-336" / "snapshots" / "ce19dc912ca5cd21c8a653c79e251e808ccabcd1"),
        "clip_cache_dir": str(model_root / "clip-vit-large-patch14-336"),
        "whisper_model_path": str(model_root / "whisper-large" / "snapshots" / "4ef9b41f0d4fe232daafdb5f76bb1dd8b23e01d7"),
        "whisper_cache_dir": str(model_root / "whisper-large"),
        "llava_model_path": str(model_root / "llava-video-7b-qwen2" / "snapshots" / "013210b3aff822f1558b166d39c1046dd109520f"),
        "llava_cache_dir": str(model_root / "llava-video-7b-qwen2"),
    }


if __name__ == "__main__":
    args = parse_args()
    payload = build_payload(args.folder, args.file_stem)

    video_path = Path(args.video_dir) / f"{args.file_stem}.mp4"
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    path_dict = build_path_dict(Path(args.model_root))
    result = run_with_video(video_path, payload, path_dict)
    print(json.dumps(result, indent=2, ensure_ascii=False))
