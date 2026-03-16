"""
Script to detect robotic actions in egocentric video frames.

This script supports two modes:
1) Refine-only mode:
   - Reads an existing processed JSON.
2) End-to-end mode:
   - Runs first-pass processing, applies refinement, and optionally uploads improved NDJSON.

In both modes, it writes Labelbox-safe NDJSON with strict constraints:
  - 1-indexed frames only (no frame 0),
  - end >= start,
  - no overlapping frame ranges.

It also optionally emits boundary refinement candidates for a targeted second pass.
"""

import base64
import gc
import io
import json
import os
import re
import random
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import labelbox as lb
import requests as http_requests
from PIL import Image
from requests.adapters import HTTPAdapter

try:
    from google import genai
except Exception:  # pragma: no cover - optional backend dependency
    genai = None


LABELBOX_LITELLM_URL = ""
DEFAULT_LITELLM_MODEL = "vertex_ai/gemini-3-flash-preview"
DEFAULT_LITELLM_PROJECT_TAG = ""
DEFAULT_GEMINI_MODEL = "gemini-3-flash-preview"

ACTION_DETECTION_PROMPT = """You are analyzing egocentric video frames from a robotic data collection setup with MULTIPLE CAMERA VIEWS stitched together.

IMPORTANT TEMPORAL CONTEXT:
- Consecutive frames may skip over short transitions.
- Prefer continuous action segments over splitting into many short segments.
- Prefer OVER-SEGMENTATION to UNDER-SEGMENTATION when uncertain.

CAMERA LAYOUT (each frame has 3 views):
┌─────────────────────────────────┐
│      TOP CENTER VIEW            │  ← Overhead bird's-eye view showing full scene,
│   (scene context, objects)      │     object positions, and spatial layout
├───────────────┬─────────────────┤
│  LEFT ARM     │   RIGHT ARM     │  ← Wrist-mounted cameras showing
│  VIEW         │   VIEW          │     detailed hand/object interactions
│  (left hand)  │   (right hand)  │
└───────────────┴─────────────────┘

ANALYSIS STRATEGY:
1. TOP VIEW: Identify WHAT objects are present and WHERE they are located
2. ARM VIEWS: Identify WHAT ACTION the hands are performing (this is PRIMARY for action detection)
3. Combine both: The action should describe what the hands are doing to which object

IMPORTANT: Actions are TEMPORAL EVENTS that span multiple frames. Identify ACTION SEGMENTS with start and end frame indices.

SEGMENTATION RULES:
- Start a NEW action segment when:
  • the manipulated object changes
  • the verb/action changes
  • the hand releases one object and reaches for another
- Short actions lasting only 1–2 frames are valid and should still be segmented.
- Do NOT merge actions involving different objects into one segment.
- Skip idle or purely transitional motion unless it includes object interaction.

Guidelines:
- Use "verb + object" format (e.g., "wipe mirror with pink cloth", "fold sheet", "place blanket")
- Focus on the ARM VIEWS for action detection - they show the actual hand manipulation
- Use the TOP VIEW for object identification and context
- Use concrete verbs: wipe, grab, place, fold, pull, push, hold, release, adjust, lift, turn, open, close, stuff, smooth, spray, scrub, etc.
- Be specific about objects: "white roll", "white tape", "blue blanket"
- An action STARTS when the movement begins and ENDS when the movement completes

Respond in this exact JSON format with ACTION SEGMENTS (not per-frame):
{
    "action_segments": [
        {"start_frame": 0, "end_frame": 12, "action": "wipe mirror with pink cloth"},
        {"start_frame": 14, "end_frame": 18, "action": "grab spray bottle"},
        {"start_frame": 20, "end_frame": 35, "action": "spray mirror"}
    ]
}

NOTE:
- start_frame and end_frame are indices (0-based) into the frames provided below.
- end_frame must be >= start_frame.

Analyze these frames in sequence:"""

SECOND_PASS_BOUNDARY_PROMPT_TEMPLATE = """You are doing a SECOND PASS boundary refinement on egocentric frames.

Context hypothesis from first pass:
- Left-side action near boundary: "{left_action}"
- Right-side action near boundary: "{right_action}"

Your goal:
1) Refine start/end boundaries precisely in this local window.
2) Merge segments only when they are truly the same action on the same object.
3) Keep separate segments when object or verb changes.
4) Prefer precision over aggressive merging.

Respond ONLY in JSON:
{{
  "action_segments": [
    {{"start_frame": 0, "end_frame": 4, "action": "grab blue pen from holder"}},
    {{"start_frame": 5, "end_frame": 12, "action": "write on white paper"}}
  ]
}}
"""

@dataclass
class Segment:
    """Internal canonical segment representation in source-frame coordinates (0-indexed)."""

    start_frame_src: int
    end_frame_src: int
    action_text_raw: str
    action_text_norm: str
    action_id: str
    confidence: Optional[float] = None


@dataclass
class FrameData:
    frame_number: int
    timestamp: float
    frame: Image.Image


class StandaloneFirstPassProcessor:
    """Self-contained first-pass processor (no dependency on egocentric_video_processor.py)."""

    def __init__(
        self,
        api_key: str,
        inference_backend: str = "litellm",
        gemini_model: str = DEFAULT_GEMINI_MODEL,
        litellm_model: Optional[str] = None,
        litellm_project_tag: Optional[str] = None,
        fps_sample: float = 3.0,
        max_frames_per_batch: int = 10,
        max_workers: int = 10,
        retry_attempts: int = 3,
        retry_delay: float = 2.0,
        request_timeout: int = 120,
    ) -> None:
        self.api_key = api_key
        self.inference_backend = inference_backend
        self.gemini_model = gemini_model
        self.litellm_model = litellm_model or DEFAULT_LITELLM_MODEL
        self.litellm_project_tag = litellm_project_tag or DEFAULT_LITELLM_PROJECT_TAG
        self.fps_sample = fps_sample
        self.max_frames_per_batch = max_frames_per_batch
        self.max_workers = max_workers
        self.retry_attempts = retry_attempts
        self.retry_delay = retry_delay
        self.request_timeout = request_timeout
        self.video_info: Dict = {}

        if inference_backend == "litellm":
            self.client = None
            self.session = http_requests.Session()
            adapter = HTTPAdapter(pool_connections=100, pool_maxsize=100)
            self.session.mount("https://", adapter)
            self.session.mount("http://", adapter)
            self.session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "x-labelbox-context": json.dumps({"tag": self.litellm_project_tag}),
                }
            )
        else:
            if genai is None:
                raise ImportError("google-genai is required for inference_backend='gemini'.")
            self.client = genai.Client(api_key=self.api_key)
            self.session = None

    @staticmethod
    def _encode_frame_b64(pil_image: Image.Image, max_size: int = 1024) -> str:
        w, h = pil_image.size
        if max(w, h) > max_size:
            ratio = max_size / max(w, h)
            pil_image = pil_image.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=80)
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    @staticmethod
    def _parse_json_response(response_text: str) -> Dict:
        text = (response_text or "").strip()
        if not text:
            return {}

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        md_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if md_match:
            try:
                return json.loads(md_match.group(1))
            except json.JSONDecodeError:
                pass

        first_brace = text.find("{")
        last_brace = text.rfind("}")
        if first_brace != -1 and last_brace > first_brace:
            try:
                return json.loads(text[first_brace : last_brace + 1])
            except json.JSONDecodeError:
                pass

        return {}

    def _build_litellm_messages(self, prompt: str, frames_batch: List[FrameData]) -> List[Dict]:
        content_parts = [{"type": "text", "text": prompt}]
        for i, frame_data in enumerate(frames_batch):
            content_parts.append({"type": "text", "text": f"\n[Frame {i} - timestamp {frame_data.timestamp}s]:"})
            content_parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{self._encode_frame_b64(frame_data.frame)}"},
                }
            )
        return [{"role": "user", "content": content_parts}]

    def _litellm_api_call(self, messages: List[Dict]) -> str:
        payload = {
            "model": self.litellm_model,
            "messages": messages,
            "temperature": 0.2,
            "max_tokens": 65536,
        }
        resp = self.session.post(
            LABELBOX_LITELLM_URL,
            json=payload,
            timeout=self.request_timeout,
        )
        if resp.status_code in (429, 503):
            raise RuntimeError(f"Rate limited ({resp.status_code})")
        resp.raise_for_status()
        data = resp.json()
        try:
            choice = data.get("choices", [{}])[0]
            msg = choice.get("message", {})
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
        except Exception:
            pass
        return ""

    def _api_call_with_retry(self, prompt: str, frames_batch: Optional[List[FrameData]] = None) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(self.retry_attempts):
            try:
                if self.inference_backend == "litellm":
                    if frames_batch:
                        messages = self._build_litellm_messages(prompt, frames_batch)
                    else:
                        messages = [{"role": "user", "content": prompt}]
                    return self._litellm_api_call(messages)

                content = [prompt]
                if frames_batch:
                    for i, frame_data in enumerate(frames_batch):
                        content.append(f"\n[Frame {i} - timestamp {frame_data.timestamp}s]:")
                        content.append(frame_data.frame)
                response = self.client.models.generate_content(
                    model=self.gemini_model,
                    contents=content,
                )
                if hasattr(response, "text") and response.text:
                    return response.text
                text_parts = []
                if response.candidates and response.candidates[0].content.parts:
                    for part in response.candidates[0].content.parts:
                        if hasattr(part, "text") and part.text:
                            text_parts.append(part.text)
                return "".join(text_parts)
            except Exception as exc:
                last_error = exc
                if attempt < self.retry_attempts - 1:
                    wait = self.retry_delay * (attempt + 1) + random.uniform(0, 1)
                    time.sleep(wait)
        raise RuntimeError(f"Inference failed after retries: {last_error}")

    def extract_frames(self, video_path: str) -> Tuple[List[FrameData], Dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if original_fps <= 0:
            original_fps = 30.0
        duration = total_frames / original_fps if total_frames > 0 else 0.0
        frame_interval = max(1, int(original_fps / self.fps_sample))

        info = {
            "original_fps": original_fps,
            "total_frames": total_frames,
            "duration": duration,
            "sampled_fps": self.fps_sample,
            "frame_interval": frame_interval,
        }
        self.video_info = info

        frames: List[FrameData] = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % frame_interval == 0:
                timestamp = frame_count / original_fps
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(
                    FrameData(
                        frame_number=frame_count,
                        timestamp=round(timestamp, 3),
                        frame=Image.fromarray(frame_rgb),
                    )
                )
            frame_count += 1
        cap.release()
        return frames, info

    def detect_actions_batch(self, frames_batch: List[FrameData], prompt: Optional[str] = None) -> List[Dict]:
        if not frames_batch:
            return []
        try:
            active_prompt = prompt or ACTION_DETECTION_PROMPT
            response_text = self._api_call_with_retry(active_prompt, frames_batch=frames_batch)
            parsed = self._parse_json_response(response_text)
            segments = parsed.get("action_segments", [])
            out: List[Dict] = []
            for item in segments:
                if not isinstance(item, dict):
                    continue
                action = str(item.get("action", "")).strip()
                if not action:
                    continue
                start_idx = int(item.get("start_frame", -1))
                end_idx = int(item.get("end_frame", start_idx))
                if not (0 <= start_idx < len(frames_batch)):
                    continue
                end_idx = min(max(start_idx, end_idx), len(frames_batch) - 1)
                out.append(
                    {
                        "start_frame": frames_batch[start_idx].frame_number,
                        "end_frame": frames_batch[end_idx].frame_number,
                        "start_timestamp": frames_batch[start_idx].timestamp,
                        "end_timestamp": frames_batch[end_idx].timestamp,
                        "action": action,
                    }
                )
            return out
        except Exception:
            return []

    @staticmethod
    def _merge_adjacent_actions(actions: List[Dict], frame_gap_tolerance: int = 30) -> List[Dict]:
        if not actions:
            return []
        sorted_actions = sorted(actions, key=lambda x: x["start_frame"])
        merged = [sorted_actions[0]]
        for action in sorted_actions[1:]:
            prev = merged[-1]
            same = action["action"].strip().lower() == prev["action"].strip().lower()
            gap = action["start_frame"] - prev["end_frame"]
            if same and gap <= frame_gap_tolerance:
                prev["end_frame"] = max(prev["end_frame"], action["end_frame"])
                prev["end_timestamp"] = max(prev.get("end_timestamp", 0), action.get("end_timestamp", 0))
            else:
                merged.append(action)
        return merged

    @staticmethod
    def _remove_overlaps(actions: List[Dict]) -> List[Dict]:
        if not actions:
            return []
        sorted_actions = sorted(actions, key=lambda x: x["start_frame"])
        result: List[Dict] = []
        for action in sorted_actions:
            if not result:
                result.append(action)
                continue
            prev = result[-1]
            if action["start_frame"] <= prev["end_frame"]:
                new_prev_end = action["start_frame"] - 1
                if new_prev_end >= prev["start_frame"]:
                    prev["end_frame"] = new_prev_end
                else:
                    result.pop()
            result.append(action)
        return [a for a in result if a["end_frame"] >= a["start_frame"]]

    def detect_all_actions(self, frames: List[FrameData]) -> List[Dict]:
        batches = [frames[i : i + self.max_frames_per_batch] for i in range(0, len(frames), self.max_frames_per_batch)]
        results: Dict[int, List[Dict]] = {}

        with ThreadPoolExecutor(max_workers=max(1, self.max_workers)) as executor:
            futures = {executor.submit(self.detect_actions_batch, batch): idx for idx, batch in enumerate(batches)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    results[idx] = future.result()
                except Exception:
                    results[idx] = []

        all_actions: List[Dict] = []
        for idx in range(len(batches)):
            all_actions.extend(results.get(idx, []))
        all_actions = self._merge_adjacent_actions(all_actions)
        all_actions = self._remove_overlaps(all_actions)
        return all_actions

    def generate_video_summary(self, frame_actions: List[Dict]) -> Dict:
        actions_text = ", ".join(a.get("action", "") for a in frame_actions[:50])
        if len(frame_actions) > 50:
            actions_text += f"... ({len(frame_actions) - 50} more actions)"

        prompt = (
            "You are analyzing an egocentric video showing first-person perspective.\n\n"
            f"The following atomic actions were detected in sequence:\n{actions_text}\n\n"
            "Provide a 1-2 sentence summary describing what happens in the video and the goal.\n"
            "Respond in JSON format: {\"summary\": \"Your summary here\"}"
        )
        try:
            summary_text = self._api_call_with_retry(prompt, frames_batch=None)
            parsed = self._parse_json_response(summary_text)
            summary = parsed.get("summary", "No summary available")
        except Exception:
            summary = "Unable to generate summary"

        return {
            "summary": summary,
            "original_fps": self.video_info.get("original_fps", 0),
            "total_frames": self.video_info.get("total_frames", 0),
            "duration": self.video_info.get("duration", 0),
            "sampled_fps": self.video_info.get("sampled_fps", self.fps_sample),
        }

    def process_video(self, video_path: str) -> Dict:
        frames, video_info = self.extract_frames(video_path)
        if not frames:
            raise ValueError("No frames extracted from video.")

        frame_actions = self.detect_all_actions(frames)
        for frame_data in frames:
            frame_data.frame = None
        frames.clear()
        gc.collect()

        video_metadata = self.generate_video_summary(frame_actions)
        return {
            "video_path": str(video_path),
            "video_info": video_info,
            "video_metadata": video_metadata,
            "frame_actions": frame_actions,
        }

def _clean_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _replace_multiword(text: str, mapping: Dict[str, str]) -> str:
    # Replace longer keys first to avoid partial collisions.
    for src in sorted(mapping.keys(), key=len, reverse=True):
        dst = mapping[src]
        text = re.sub(rf"\b{re.escape(src)}\b", dst, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def canonicalize_action(
    action_text: str,
    verb_map: Dict[str, str],
    object_map: Dict[str, str],
) -> Tuple[str, str]:
    """
    Returns:
      - normalized_action_text (human readable)
      - action_id (stable canonical key, used for merging)
    """
    cleaned = _clean_text(action_text)
    if not cleaned:
        return "", "UNKNOWN"

    # Normalize multi-word and token synonyms.
    normalized = _replace_multiword(cleaned, verb_map)
    normalized = _replace_multiword(normalized, object_map)

    tokens = normalized.split()
    if not tokens:
        return "", "UNKNOWN"

    # Assume first token is the canonical verb after normalization.
    verb = tokens[0]
    obj = " ".join(tokens[1:]).strip()

    if obj:
        norm_text = f"{verb} {obj}"
        action_id = f"{verb}_{obj}".upper().replace(" ", "_")
    else:
        norm_text = verb
        action_id = verb.upper()

    return norm_text, action_id


def get_canonical_maps() -> Tuple[Dict[str, str], Dict[str, str]]:
    verb_map = {
        "pick up": "grab",
        "pickup": "grab",
        "lift": "grab",
        "take": "grab",
        "set down": "place",
        "put down": "place",
        "drop": "release",
        "hold onto": "hold",
        "turn on": "open",
        "turn off": "close",
    }
    object_map = {
        "rag": "cloth",
        "towel": "cloth",
        "sprayer": "spray bottle",
        "bottle sprayer": "spray bottle",
    }
    return verb_map, object_map


def build_second_pass_prompt(left_action: str, right_action: str) -> str:
    return SECOND_PASS_BOUNDARY_PROMPT_TEMPLATE.format(
        left_action=left_action or "unknown",
        right_action=right_action or "unknown",
    )


def _segments_signature(segments: List[Segment]) -> List[Tuple[int, int, str]]:
    return sorted((s.start_frame_src, s.end_frame_src, s.action_id) for s in segments)


def apply_llm_second_pass(
    segments: List[Segment],
    sampled_frames: List[FrameData],
    candidates: List[Dict],
    processor: StandaloneFirstPassProcessor,
    max_gap_src: int,
    min_window_sampled_frames: int = 4,
    max_windows: Optional[int] = None,
) -> Tuple[List[Segment], Dict, List[Dict]]:
    if not segments or not sampled_frames or not candidates:
        stats = {
            "enabled": True,
            "candidate_count": len(candidates),
            "processed_candidates": 0,
            "applied_candidates": 0,
            "skipped_no_frames": 0,
            "skipped_empty_response": 0,
            "skipped_unchanged": 0,
            "errors": 0,
        }
        return segments, stats, []

    verb_map, object_map = get_canonical_maps()
    updated_segments = list(segments)
    candidate_results: List[Dict] = []

    if max_windows is None or max_windows <= 0:
        active_candidates = candidates
    else:
        active_candidates = candidates[:max_windows]

    processed_candidates = 0
    applied_candidates = 0
    skipped_no_frames = 0
    skipped_empty_response = 0
    skipped_unchanged = 0
    errors = 0

    for idx, cand in enumerate(active_candidates):
        processed_candidates += 1
        ws = int(cand.get("window_start_src", 0))
        we = int(cand.get("window_end_src", ws))
        if we < ws:
            ws, we = we, ws

        window_frames = [fd for fd in sampled_frames if ws <= fd.frame_number <= we]
        if len(window_frames) < min_window_sampled_frames:
            skipped_no_frames += 1
            candidate_results.append(
                {
                    "candidate_idx": idx,
                    "status": "skipped_no_frames",
                    "window_start_src": ws,
                    "window_end_src": we,
                    "window_sampled_frames": len(window_frames),
                }
            )
            continue

        left_action = str(cand.get("left_action", "")).strip()
        right_action = str(cand.get("right_action", "")).strip()
        prompt = build_second_pass_prompt(left_action, right_action)

        try:
            rerun_actions = processor.detect_actions_batch(window_frames, prompt=prompt)
        except Exception as exc:
            errors += 1
            candidate_results.append(
                {
                    "candidate_idx": idx,
                    "status": "error",
                    "window_start_src": ws,
                    "window_end_src": we,
                    "error": str(exc),
                }
            )
            continue

        if not rerun_actions:
            skipped_empty_response += 1
            candidate_results.append(
                {
                    "candidate_idx": idx,
                    "status": "skipped_empty_response",
                    "window_start_src": ws,
                    "window_end_src": we,
                }
            )
            continue

        rerun_segments = to_segments(rerun_actions, verb_map=verb_map, object_map=object_map)
        rerun_segments = conservative_merge(rerun_segments, max_gap_src_frames=max_gap_src)
        rerun_segments = enforce_non_overlap_src(rerun_segments)
        if not rerun_segments:
            skipped_empty_response += 1
            candidate_results.append(
                {
                    "candidate_idx": idx,
                    "status": "skipped_empty_after_validation",
                    "window_start_src": ws,
                    "window_end_src": we,
                }
            )
            continue

        old_window_segments = [
            s for s in updated_segments if not (s.end_frame_src < ws or s.start_frame_src > we)
        ]
        if _segments_signature(old_window_segments) == _segments_signature(rerun_segments):
            skipped_unchanged += 1
            candidate_results.append(
                {
                    "candidate_idx": idx,
                    "status": "skipped_unchanged",
                    "window_start_src": ws,
                    "window_end_src": we,
                }
            )
            continue

        outside_segments = [
            s for s in updated_segments if s.end_frame_src < ws or s.start_frame_src > we
        ]
        updated_segments = outside_segments + rerun_segments
        updated_segments.sort(key=lambda s: (s.start_frame_src, s.end_frame_src))
        updated_segments = conservative_merge(updated_segments, max_gap_src_frames=max_gap_src)
        updated_segments = enforce_non_overlap_src(updated_segments)

        applied_candidates += 1
        candidate_results.append(
            {
                "candidate_idx": idx,
                "status": "applied",
                "window_start_src": ws,
                "window_end_src": we,
                "old_segments_in_window": len(old_window_segments),
                "new_segments_in_window": len(rerun_segments),
            }
        )

    stats = {
        "enabled": True,
        "candidate_count": len(candidates),
        "processed_candidates": processed_candidates,
        "applied_candidates": applied_candidates,
        "skipped_no_frames": skipped_no_frames,
        "skipped_empty_response": skipped_empty_response,
        "skipped_unchanged": skipped_unchanged,
        "errors": errors,
    }
    return updated_segments, stats, candidate_results


def to_segments(
    frame_actions: List[Dict],
    verb_map: Dict[str, str],
    object_map: Dict[str, str],
) -> List[Segment]:
    segments: List[Segment] = []
    for row in frame_actions:
        raw = str(row.get("action", "")).strip()
        if not raw:
            continue

        start_src = int(row.get("start_frame", -1))
        end_src = int(row.get("end_frame", start_src))
        if start_src < 0:
            continue
        if end_src < start_src:
            end_src = start_src

        norm_text, action_id = canonicalize_action(raw, verb_map, object_map)
        if not norm_text:
            continue

        segments.append(
            Segment(
                start_frame_src=start_src,
                end_frame_src=end_src,
                action_text_raw=raw,
                action_text_norm=norm_text,
                action_id=action_id,
                confidence=row.get("confidence"),
            )
        )

    segments.sort(key=lambda s: (s.start_frame_src, s.end_frame_src))
    return segments


def conservative_merge(
    segments: List[Segment],
    max_gap_src_frames: int,
) -> List[Segment]:
    """Precision-first merge: only same action_id and small gap."""
    if not segments:
        return []

    merged: List[Segment] = [segments[0]]
    for seg in segments[1:]:
        prev = merged[-1]
        gap = seg.start_frame_src - prev.end_frame_src
        same_action = seg.action_id == prev.action_id
        if same_action and gap <= max_gap_src_frames:
            merged[-1] = Segment(
                start_frame_src=prev.start_frame_src,
                end_frame_src=max(prev.end_frame_src, seg.end_frame_src),
                action_text_raw=prev.action_text_raw,
                action_text_norm=prev.action_text_norm,
                action_id=prev.action_id,
                confidence=prev.confidence if prev.confidence is not None else seg.confidence,
            )
        else:
            merged.append(seg)
    return merged


def enforce_non_overlap_src(segments: List[Segment]) -> List[Segment]:
    """
    Deterministic no-overlap enforcement in source coordinates.
    If overlap occurs:
      - same action_id => merge
      - different action_id => truncate previous to current.start - 1
    """
    if not segments:
        return []

    fixed: List[Segment] = []
    for seg in sorted(segments, key=lambda s: (s.start_frame_src, s.end_frame_src)):
        if not fixed:
            fixed.append(seg)
            continue

        prev = fixed[-1]
        if seg.start_frame_src <= prev.end_frame_src:
            if seg.action_id == prev.action_id:
                fixed[-1] = Segment(
                    start_frame_src=prev.start_frame_src,
                    end_frame_src=max(prev.end_frame_src, seg.end_frame_src),
                    action_text_raw=prev.action_text_raw,
                    action_text_norm=prev.action_text_norm,
                    action_id=prev.action_id,
                    confidence=prev.confidence if prev.confidence is not None else seg.confidence,
                )
                continue

            new_prev_end = seg.start_frame_src - 1
            if new_prev_end >= prev.start_frame_src:
                fixed[-1] = Segment(
                    start_frame_src=prev.start_frame_src,
                    end_frame_src=new_prev_end,
                    action_text_raw=prev.action_text_raw,
                    action_text_norm=prev.action_text_norm,
                    action_id=prev.action_id,
                    confidence=prev.confidence,
                )
            else:
                fixed.pop()

        fixed.append(seg)

    return [s for s in fixed if s.end_frame_src >= s.start_frame_src]


def _boundary_uncertainty(left_seg: Segment, right_seg: Segment, tiny_len_src: int) -> float:
    """Heuristic uncertainty score in [0, 1]."""
    score = 0.0

    gap = right_seg.start_frame_src - left_seg.end_frame_src
    if 0 <= gap <= 3:
        score += 0.35
    elif gap < 0:
        score += 0.35

    if left_seg.action_id != right_seg.action_id:
        sim = SequenceMatcher(None, left_seg.action_text_norm, right_seg.action_text_norm).ratio()
        if sim >= 0.75:
            score += 0.40

    left_len = left_seg.end_frame_src - left_seg.start_frame_src + 1
    right_len = right_seg.end_frame_src - right_seg.start_frame_src + 1
    if left_len <= tiny_len_src or right_len <= tiny_len_src:
        score += 0.25

    return min(score, 1.0)


def build_boundary_candidates(
    segments: List[Segment],
    threshold: float,
    max_ratio: float,
    context_window_src: int,
) -> List[Dict]:
    """
    Build limited candidate list for optional second-pass boundary refinement.
    """
    raw_candidates: List[Dict] = []
    for i in range(len(segments) - 1):
        left_seg = segments[i]
        right_seg = segments[i + 1]
        score = _boundary_uncertainty(left_seg, right_seg, tiny_len_src=2)
        if score < threshold:
            continue

        center = max(left_seg.end_frame_src, right_seg.start_frame_src)
        raw_candidates.append(
            {
                "boundary_index": i,
                "score": round(score, 3),
                "window_start_src": max(0, center - context_window_src),
                "window_end_src": center + context_window_src,
                "left_action": left_seg.action_text_norm,
                "right_action": right_seg.action_text_norm,
                "left_action_id": left_seg.action_id,
                "right_action_id": right_seg.action_id,
            }
        )

    raw_candidates.sort(key=lambda x: x["score"], reverse=True)
    cap = max(1, int(len(segments) * max_ratio)) if segments else 0
    return raw_candidates[:cap]


def to_labelbox_answer_segments(segments_src: List[Segment]) -> List[Dict]:
    """
    Convert source 0-indexed segments to Labelbox answer segments:
    [{"value": "...", "frames": [{"start": 1, "end": 3}]}]
    Enforces:
      - start >= 1
      - end >= start
      - no overlaps
    """
    answer_items: List[Dict] = []
    for seg in sorted(segments_src, key=lambda s: (s.start_frame_src, s.end_frame_src)):
        start_lb = max(1, seg.start_frame_src + 1)
        end_lb = max(1, seg.end_frame_src + 1)
        if end_lb < start_lb:
            end_lb = start_lb

        answer_items.append(
            {
                "value": seg.action_text_norm,
                "frames": [{"start": int(start_lb), "end": int(end_lb)}],
                "action_id": seg.action_id,
            }
        )

    # Enforce no overlap in Labelbox coordinates.
    fixed: List[Dict] = []
    for item in answer_items:
        if not fixed:
            fixed.append(item)
            continue

        prev = fixed[-1]
        prev_start = prev["frames"][0]["start"]
        prev_end = prev["frames"][0]["end"]
        curr_start = item["frames"][0]["start"]
        curr_end = item["frames"][0]["end"]

        if curr_start <= prev_end:
            if prev.get("action_id") == item.get("action_id"):
                prev["frames"][0]["end"] = max(prev_end, curr_end)
                continue

            new_prev_end = curr_start - 1
            if new_prev_end >= prev_start:
                prev["frames"][0]["end"] = new_prev_end
            else:
                fixed.pop()

        fixed.append(item)

    # Strip helper field before export.
    export_items: List[Dict] = []
    for item in fixed:
        if item["frames"][0]["end"] >= item["frames"][0]["start"]:
            export_items.append(
                {"value": item["value"], "frames": item["frames"]}
            )
    return export_items


def build_ndjson_entries(
    global_key: str,
    action_feature_name: str,
    summary_feature_name: str,
    summary_text: str,
    answer_segments: List[Dict],
) -> List[Dict]:
    entries: List[Dict] = []
    if answer_segments:
        entries.append(
            {
                "name": action_feature_name,
                "answer": answer_segments,
                "dataRow": {"globalKey": str(global_key)},
            }
        )
    if summary_text:
        entries.append(
            {
                "name": summary_feature_name,
                "answer": str(summary_text),
                "dataRow": {"globalKey": str(global_key)},
            }
        )
    return entries


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def dump_ndjson(path: Path, entries: List[Dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in entries:
            f.write(json.dumps(row) + "\n")


def resolve_api_key(explicit_key: Optional[str], env_name: str) -> Optional[str]:
    return explicit_key or os.getenv(env_name)


def download_video_with_retry(
    video_url: str,
    dest_path: str,
    max_retries: int = 3,
    connect_timeout: int = 30,
    read_timeout: int = 300,
) -> None:
    for attempt in range(max_retries):
        try:
            with http_requests.get(video_url, stream=True, timeout=(connect_timeout, read_timeout)) as resp:
                resp.raise_for_status()
                with open(dest_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
                return
            raise IOError("Downloaded file is empty.")
        except Exception as exc:
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if attempt < max_retries - 1:
                wait = (attempt + 1) * 3 + random.uniform(0, 2)
                print(f"download retry {attempt + 1}/{max_retries}: {type(exc).__name__}: {exc}")
                time.sleep(wait)
            else:
                raise IOError(f"Failed to download after {max_retries} attempts: {exc}") from exc


def get_video_url_by_global_key(labelbox_api_key: str, global_key: str) -> str:
    client = lb.Client(api_key=labelbox_api_key)
    data_row = client.get_data_row_by_global_key(global_key)
    return data_row.row_data


def run_refinement_pipeline(
    data: Dict,
    global_key: str,
    output_ndjson_path: Path,
    output_json_path: Optional[Path],
    candidates_path: Optional[Path],
    action_feature_name: str,
    summary_feature_name: str,
    max_gap_src: int,
    candidate_threshold: float,
    candidate_max_ratio: float,
    candidate_context_window_src: int,
    second_pass_enabled: bool = False,
    second_pass_processor: Optional[StandaloneFirstPassProcessor] = None,
    sampled_frames: Optional[List[FrameData]] = None,
    second_pass_max_windows: Optional[int] = None,
    second_pass_min_window_sampled_frames: int = 4,
) -> Dict:
    frame_actions = data.get("frame_actions", [])
    summary_text = data.get("video_metadata", {}).get("summary", "")

    verb_map, object_map = get_canonical_maps()

    segments = to_segments(frame_actions, verb_map=verb_map, object_map=object_map)
    merged = conservative_merge(segments, max_gap_src_frames=max_gap_src)
    fixed_src = enforce_non_overlap_src(merged)

    initial_candidates = build_boundary_candidates(
        fixed_src,
        threshold=candidate_threshold,
        max_ratio=candidate_max_ratio,
        context_window_src=candidate_context_window_src,
    )
    second_pass_results: List[Dict] = []
    second_pass_stats = {
        "enabled": bool(second_pass_enabled),
        "candidate_count": len(initial_candidates),
        "processed_candidates": 0,
        "applied_candidates": 0,
        "skipped_no_frames": 0,
        "skipped_empty_response": 0,
        "skipped_unchanged": 0,
        "errors": 0,
    }

    if second_pass_enabled:
        if second_pass_processor is None or sampled_frames is None:
            raise ValueError(
                "second_pass_enabled=True requires second_pass_processor and sampled_frames."
            )
        fixed_src, second_pass_stats, second_pass_results = apply_llm_second_pass(
            segments=fixed_src,
            sampled_frames=sampled_frames,
            candidates=initial_candidates,
            processor=second_pass_processor,
            max_gap_src=max_gap_src,
            min_window_sampled_frames=second_pass_min_window_sampled_frames,
            max_windows=second_pass_max_windows,
        )

    final_candidates = build_boundary_candidates(
        fixed_src,
        threshold=candidate_threshold,
        max_ratio=candidate_max_ratio,
        context_window_src=candidate_context_window_src,
    )

    answer_segments = to_labelbox_answer_segments(fixed_src)
    ndjson_entries = build_ndjson_entries(
        global_key=global_key,
        action_feature_name=action_feature_name,
        summary_feature_name=summary_feature_name,
        summary_text=summary_text,
        answer_segments=answer_segments,
    )
    dump_ndjson(output_ndjson_path, ndjson_entries)

    diagnostics = {
        "global_key": global_key,
        "counts": {
            "input_segments": len(frame_actions),
            "canonical_segments": len(segments),
            "merged_segments": len(merged),
            "final_segments": len(fixed_src),
            "labelbox_answer_segments": len(answer_segments),
            "initial_second_pass_candidates": len(initial_candidates),
            "remaining_second_pass_candidates": len(final_candidates),
        },
        "final_segments_src": [
            {
                "start_frame_src": s.start_frame_src,
                "end_frame_src": s.end_frame_src,
                "action_text_raw": s.action_text_raw,
                "action_text_norm": s.action_text_norm,
                "action_id": s.action_id,
            }
            for s in fixed_src
        ],
        "labelbox_answer_segments": answer_segments,
        "second_pass": second_pass_stats,
        "second_pass_results": second_pass_results,
        "initial_candidates": initial_candidates,
        "final_candidates": final_candidates,
    }

    if output_json_path:
        dump_json(output_json_path, diagnostics)

    if candidates_path:
        dump_json(
            candidates_path,
            {
                "global_key": global_key,
                "initial_candidate_count": len(initial_candidates),
                "final_candidate_count": len(final_candidates),
                "second_pass": second_pass_stats,
                "initial_candidates": initial_candidates,
                "second_pass_results": second_pass_results,
                "final_candidates": final_candidates,
            },
        )

    return diagnostics


def run_first_pass(
    global_key: str,
    output_dir: Path,
    inference_backend: str,
    fps_sample: float,
    max_workers: int,
    max_frames_per_batch: int,
    retry_attempts: int,
    video_url: Optional[str],
    labelbox_api_key: Optional[str],
    gemini_api_key: Optional[str],
    litellm_model: Optional[str],
    litellm_project_tag: Optional[str],
    download_retries: int,
    download_connect_timeout: int,
    download_read_timeout: int,
) -> Tuple[Dict, Path, List[FrameData], StandaloneFirstPassProcessor]:
    if inference_backend == "litellm":
        processor_api_key = labelbox_api_key
        if not processor_api_key:
            raise ValueError("LABELBOX_API_KEY is required for litellm backend.")
    else:
        processor_api_key = gemini_api_key
        if not processor_api_key:
            raise ValueError("GEMINI_API_KEY is required for gemini backend.")

    resolved_video_url = video_url
    if not resolved_video_url:
        if not labelbox_api_key:
            raise ValueError("Provide video_url or LABELBOX_API_KEY to fetch URL by global key.")
        resolved_video_url = get_video_url_by_global_key(labelbox_api_key, global_key)

    processor = StandaloneFirstPassProcessor(
        api_key=processor_api_key,
        fps_sample=fps_sample,
        max_frames_per_batch=max_frames_per_batch,
        retry_attempts=retry_attempts,
        max_workers=max_workers,
        inference_backend=inference_backend,
        litellm_model=litellm_model,
        litellm_project_tag=litellm_project_tag,
        gemini_model=DEFAULT_GEMINI_MODEL,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_video_path = Path(tmp.name)

    sampled_frames: List[FrameData] = []
    try:
        t0 = time.time()
        download_video_with_retry(
            resolved_video_url,
            str(temp_video_path),
            max_retries=download_retries,
            connect_timeout=download_connect_timeout,
            read_timeout=download_read_timeout,
        )
        print(f"[{global_key}] download complete in {time.time() - t0:.1f}s")

        t1 = time.time()
        sampled_frames, video_info = processor.extract_frames(str(temp_video_path))
        if not sampled_frames:
            raise ValueError("No frames extracted from video.")
        frame_actions = processor.detect_all_actions(sampled_frames)
        video_metadata = processor.generate_video_summary(frame_actions)
        result = {
            "video_path": str(temp_video_path),
            "video_info": video_info,
            "video_metadata": video_metadata,
            "frame_actions": frame_actions,
        }
        print(f"[{global_key}] first pass complete in {time.time() - t1:.1f}s")

    finally:
        if temp_video_path.exists():
            temp_video_path.unlink()

    first_pass_json = output_dir / f"{global_key}_processed.json"
    dump_json(first_pass_json, result)
    return result, first_pass_json, sampled_frames, processor


def prepare_second_pass_context(
    global_key: str,
    inference_backend: str,
    fps_sample: float,
    max_workers: int,
    max_frames_per_batch: int,
    retry_attempts: int,
    video_url: Optional[str],
    labelbox_api_key: Optional[str],
    gemini_api_key: Optional[str],
    litellm_model: Optional[str],
    litellm_project_tag: Optional[str],
    download_retries: int,
    download_connect_timeout: int,
    download_read_timeout: int,
) -> Tuple[List[FrameData], StandaloneFirstPassProcessor]:
    if inference_backend == "litellm":
        processor_api_key = labelbox_api_key
        if not processor_api_key:
            raise ValueError("LABELBOX_API_KEY is required for litellm backend.")
    else:
        processor_api_key = gemini_api_key
        if not processor_api_key:
            raise ValueError("GEMINI_API_KEY is required for gemini backend.")

    resolved_video_url = video_url
    if not resolved_video_url:
        if not labelbox_api_key:
            raise ValueError("Provide video_url or LABELBOX_API_KEY to fetch URL by global key.")
        resolved_video_url = get_video_url_by_global_key(labelbox_api_key, global_key)

    processor = StandaloneFirstPassProcessor(
        api_key=processor_api_key,
        fps_sample=fps_sample,
        max_frames_per_batch=max_frames_per_batch,
        retry_attempts=retry_attempts,
        max_workers=max_workers,
        inference_backend=inference_backend,
        litellm_model=litellm_model,
        litellm_project_tag=litellm_project_tag,
        gemini_model=DEFAULT_GEMINI_MODEL,
    )

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        temp_video_path = Path(tmp.name)

    try:
        download_video_with_retry(
            resolved_video_url,
            str(temp_video_path),
            max_retries=download_retries,
            connect_timeout=download_connect_timeout,
            read_timeout=download_read_timeout,
        )
        sampled_frames, _ = processor.extract_frames(str(temp_video_path))
    finally:
        if temp_video_path.exists():
            temp_video_path.unlink()

    if not sampled_frames:
        raise ValueError("No sampled frames extracted for second pass context.")
    return sampled_frames, processor


def upload_improved_ndjson(
    labelbox_api_key: str,
    project_id: str,
    global_key: str,
    ndjson_path: Path,
    job_name: Optional[str],
) -> Dict:
    client = lb.Client(api_key=labelbox_api_key)
    project = client.get_project(project_id)
    effective_job_name = job_name or f"mal_import_{global_key}_improved"

    upload_job = lb.MALPredictionImport.create_from_file(
        client=client,
        project_id=project.uid,
        name=effective_job_name,
        path=str(ndjson_path),
    )
    upload_job.wait_until_done()
    return {
        "job_name": effective_job_name,
        "errors": upload_job.errors,
        "statuses": upload_job.statuses,
    }


def main(
    global_key: str,
    input_json: Optional[str] = None,
    video_url: Optional[str] = None,
    output_dir: str = "output",
    output_ndjson: Optional[str] = None,
    output_json: Optional[str] = None,
    boundary_candidates_json: Optional[str] = None,
    pipeline_report_json: Optional[str] = None,
    action_feature_name: str = "Action",
    summary_feature_name: str = "Global summary",
    max_gap_src: int = 15,
    candidate_threshold: float = 0.60,
    candidate_max_ratio: float = 0.15,
    candidate_context_window_src: int = 30,
    second_pass_enabled: bool = False,
    second_pass_max_windows: Optional[int] = None,
    second_pass_min_window_sampled_frames: int = 4,
    inference_backend: str = "litellm",
    fps_sample: float = 3.0,
    max_workers: int = 10,
    max_frames_per_batch: int = 10,
    retry_attempts: int = 3,
    download_retries: int = 3,
    download_connect_timeout: int = 30,
    download_read_timeout: int = 300,
    labelbox_api_key: Optional[str] = None,
    gemini_api_key: Optional[str] = None,
    project_id: Optional[str] = None,
    litellm_model: Optional[str] = None,
    litellm_project_tag: Optional[str] = None,
    upload: bool = False,
    job_name: Optional[str] = None,
) -> Dict:
    if inference_backend not in {"gemini", "litellm"}:
        raise ValueError("inference_backend must be 'gemini' or 'litellm'.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_ndjson_path = Path(output_ndjson) if output_ndjson else output_dir / f"{global_key}_improved.ndjson"
    output_json_path = Path(output_json) if output_json else output_dir / f"{global_key}_improved_debug.json"
    candidates_path = Path(boundary_candidates_json) if boundary_candidates_json else output_dir / f"{global_key}_boundary_candidates.json"
    pipeline_report_path = Path(pipeline_report_json) if pipeline_report_json else output_dir / f"{global_key}_pipeline_report.json"

    labelbox_api_key = resolve_api_key(labelbox_api_key, "LABELBOX_API_KEY")
    gemini_api_key = resolve_api_key(gemini_api_key, "GEMINI_API_KEY")

    first_pass_source = "input_json"
    first_pass_json_path: Optional[Path] = None
    sampled_frames: Optional[List[FrameData]] = None
    second_pass_processor: Optional[StandaloneFirstPassProcessor] = None

    if input_json:
        input_path = Path(input_json)
        data = load_json(input_path)
        first_pass_json_path = input_path
        print(f"[{global_key}] refine-only mode using {input_path}")
        if second_pass_enabled:
            sampled_frames, second_pass_processor = prepare_second_pass_context(
                global_key=global_key,
                inference_backend=inference_backend,
                fps_sample=fps_sample,
                max_workers=max_workers,
                max_frames_per_batch=max_frames_per_batch,
                retry_attempts=retry_attempts,
                video_url=video_url,
                labelbox_api_key=labelbox_api_key,
                gemini_api_key=gemini_api_key,
                litellm_model=litellm_model,
                litellm_project_tag=litellm_project_tag,
                download_retries=download_retries,
                download_connect_timeout=download_connect_timeout,
                download_read_timeout=download_read_timeout,
            )
    else:
        first_pass_source = "generated"
        data, first_pass_json_path, sampled_frames, second_pass_processor = run_first_pass(
            global_key=global_key,
            output_dir=output_dir,
            inference_backend=inference_backend,
            fps_sample=fps_sample,
            max_workers=max_workers,
            max_frames_per_batch=max_frames_per_batch,
            retry_attempts=retry_attempts,
            video_url=video_url,
            labelbox_api_key=labelbox_api_key,
            gemini_api_key=gemini_api_key,
            litellm_model=litellm_model,
            litellm_project_tag=litellm_project_tag,
            download_retries=download_retries,
            download_connect_timeout=download_connect_timeout,
            download_read_timeout=download_read_timeout,
        )

    diagnostics = run_refinement_pipeline(
        data=data,
        global_key=global_key,
        output_ndjson_path=output_ndjson_path,
        output_json_path=output_json_path,
        candidates_path=candidates_path,
        action_feature_name=action_feature_name,
        summary_feature_name=summary_feature_name,
        max_gap_src=max_gap_src,
        candidate_threshold=candidate_threshold,
        candidate_max_ratio=candidate_max_ratio,
        candidate_context_window_src=candidate_context_window_src,
        second_pass_enabled=second_pass_enabled,
        second_pass_processor=second_pass_processor,
        sampled_frames=sampled_frames,
        second_pass_max_windows=second_pass_max_windows,
        second_pass_min_window_sampled_frames=second_pass_min_window_sampled_frames,
    )

    if sampled_frames is not None:
        for frame_data in sampled_frames:
            frame_data.frame = None
        sampled_frames.clear()
    if second_pass_processor is not None and getattr(second_pass_processor, "session", None) is not None:
        second_pass_processor.session.close()
    gc.collect()

    upload_result = None
    dry_run = not upload
    if upload:
        if not labelbox_api_key:
            raise ValueError("Upload requires LABELBOX_API_KEY (env or LABELBOX_API_KEY variable).")
        if not project_id:
            raise ValueError("Upload requires project_id.")

        print(f"[{global_key}] uploading improved NDJSON...")
        upload_result = upload_improved_ndjson(
            labelbox_api_key=labelbox_api_key,
            project_id=project_id,
            global_key=global_key,
            ndjson_path=output_ndjson_path,
            job_name=job_name,
        )
        print(f"[{global_key}] upload complete.")

    report = {
        "global_key": global_key,
        "mode": first_pass_source,
        "first_pass_json": str(first_pass_json_path) if first_pass_json_path else None,
        "improved_ndjson": str(output_ndjson_path),
        "improved_debug_json": str(output_json_path),
        "boundary_candidates_json": str(candidates_path),
        "dry_run": dry_run,
        "uploaded": bool(upload),
        "upload_result": upload_result,
        "counts": diagnostics.get("counts", {}),
        "second_pass": diagnostics.get("second_pass", {}),
        "labelbox_constraints_applied": {
            "start_min_1": True,
            "end_gte_start": True,
            "non_overlapping_segments": True,
        },
    }
    dump_json(pipeline_report_path, report)

    print(f"Wrote improved NDJSON: {output_ndjson_path}")
    print(f"Wrote diagnostics JSON: {output_json_path}")
    print(f"Wrote boundary candidates: {candidates_path}")
    print(f"Wrote pipeline report: {pipeline_report_path}")
    if dry_run:
        print("Dry-run mode: no Labelbox upload was performed.")
    return report


if __name__ == "__main__":

    # Required
    GLOBAL_KEY = "replace_with_global_key"

    # Choose one:
    # - Refine-only mode: set INPUT_JSON to an existing *_processed.json
    # - End-to-end mode: set INPUT_JSON = None and provide VIDEO_URL or LABELBOX_API_KEY
    INPUT_JSON = None
    VIDEO_URL = None

    # Output paths
    OUTPUT_DIR = "output"
    OUTPUT_NDJSON = None
    OUTPUT_JSON = None
    BOUNDARY_CANDIDATES_JSON = None
    PIPELINE_REPORT_JSON = None

    # Ontology names (case-sensitive in Labelbox)
    ACTION_FEATURE_NAME = "Action"
    SUMMARY_FEATURE_NAME = "Global summary"

    # Refinement tuning
    MAX_GAP_SRC = 15
    CANDIDATE_THRESHOLD = 0.60
    CANDIDATE_MAX_RATIO = 0.15
    CANDIDATE_CONTEXT_WINDOW_SRC = 30
    SECOND_PASS_ENABLED = True
    SECOND_PASS_MAX_WINDOWS = None
    SECOND_PASS_MIN_WINDOW_SAMPLED_FRAMES = 4

    # First-pass settings (used only when INPUT_JSON is None)
    INFERENCE_BACKEND = "litellm"  # "litellm" or "gemini"
    FPS_SAMPLE = 3.0
    MAX_WORKERS = 10
    MAX_FRAMES_PER_BATCH = 10
    RETRY_ATTEMPTS = 3
    DOWNLOAD_RETRIES = 3
    DOWNLOAD_CONNECT_TIMEOUT = 30
    DOWNLOAD_READ_TIMEOUT = 300

    # Auth and upload
    LABELBOX_API_KEY = None  # if None, uses env LABELBOX_API_KEY
    GEMINI_API_KEY = None    # if None, uses env GEMINI_API_KEY
    PROJECT_ID = None        # required only when UPLOAD = True
    LITELLM_MODEL = None
    LITELLM_PROJECT_TAG = None
    UPLOAD = False           # True => upload improved NDJSON to Labelbox
    JOB_NAME = None

    if GLOBAL_KEY == "replace_with_global_key":
        raise ValueError("Set GLOBAL_KEY in __main__ before running this script.")

    main(
        global_key=GLOBAL_KEY,
        input_json=INPUT_JSON,
        video_url=VIDEO_URL,
        output_dir=OUTPUT_DIR,
        output_ndjson=OUTPUT_NDJSON,
        output_json=OUTPUT_JSON,
        boundary_candidates_json=BOUNDARY_CANDIDATES_JSON,
        pipeline_report_json=PIPELINE_REPORT_JSON,
        action_feature_name=ACTION_FEATURE_NAME,
        summary_feature_name=SUMMARY_FEATURE_NAME,
        max_gap_src=MAX_GAP_SRC,
        candidate_threshold=CANDIDATE_THRESHOLD,
        candidate_max_ratio=CANDIDATE_MAX_RATIO,
        candidate_context_window_src=CANDIDATE_CONTEXT_WINDOW_SRC,
        second_pass_enabled=SECOND_PASS_ENABLED,
        second_pass_max_windows=SECOND_PASS_MAX_WINDOWS,
        second_pass_min_window_sampled_frames=SECOND_PASS_MIN_WINDOW_SAMPLED_FRAMES,
        inference_backend=INFERENCE_BACKEND,
        fps_sample=FPS_SAMPLE,
        max_workers=MAX_WORKERS,
        max_frames_per_batch=MAX_FRAMES_PER_BATCH,
        retry_attempts=RETRY_ATTEMPTS,
        download_retries=DOWNLOAD_RETRIES,
        download_connect_timeout=DOWNLOAD_CONNECT_TIMEOUT,
        download_read_timeout=DOWNLOAD_READ_TIMEOUT,
        labelbox_api_key=LABELBOX_API_KEY,
        gemini_api_key=GEMINI_API_KEY,
        project_id=PROJECT_ID,
        litellm_model=LITELLM_MODEL,
        litellm_project_tag=LITELLM_PROJECT_TAG,
        upload=UPLOAD,
        job_name=JOB_NAME,
    )
