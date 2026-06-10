import os
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"

import argparse
import logging
import math
import mimetypes
import os
import shutil
import subprocess
import threading
import time
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import numpy as np
from PIL import Image
import requests
import torch

os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TF", "0")

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from transformers import WhisperModel

from musetalk.utils.audio_processor import AudioProcessor
from musetalk.utils.blending import get_image_blending, get_image_prepare_material
from musetalk.utils.face_parsing import FaceParsing
from musetalk.utils.utils import load_all_model


PROJECT_DIR = Path(__file__).resolve().parent
RESULT_ROOT = PROJECT_DIR / "results" / "api"
INPUT_ROOT = RESULT_ROOT / "inputs"
OUTPUT_ROOT = RESULT_ROOT / "outputs"

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".flv", ".webm"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".ogg", ".flac"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for directory in (INPUT_ROOT, OUTPUT_ROOT):
    directory.mkdir(parents=True, exist_ok=True)


@dataclass
class Settings:
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "6006"))
    ffmpeg_path: str = os.getenv("FFMPEG_PATH", "./ffmpeg-4.4-amd64-static/")
    gpu_id: int = int(os.getenv("MUSETALK_GPU_ID", "0"))
    version: str = os.getenv("MUSETALK_VERSION", "v15")
    vae_type: str = os.getenv("MUSETALK_VAE_TYPE", "")
    unet_config: str = os.getenv("MUSETALK_UNET_CONFIG", "./models/musetalkV15/musetalk.json")
    unet_model_path: str = os.getenv("MUSETALK_UNET_MODEL", "./models/musetalkV15/unet.pth")
    whisper_dir: str = os.getenv("MUSETALK_WHISPER_DIR", "./models/whisper")
    use_float16: bool = os.getenv("MUSETALK_USE_FLOAT16", "0").lower() in {"1", "true", "yes"}
    face_confidence: float = float(os.getenv("MUSETALK_FACE_CONFIDENCE", "0.30"))
    max_download_bytes: int = int(os.getenv("API_MAX_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))
    download_retries: int = int(os.getenv("API_DOWNLOAD_RETRIES", "2"))
    download_retry_backoff_seconds: float = float(os.getenv("API_DOWNLOAD_RETRY_BACKOFF_SECONDS", "1.0"))
    face_embedding_backend: str = os.getenv("MUSETALK_FACE_EMBEDDING_BACKEND", "auto").lower()
    face_embedding_model: str = os.getenv("MUSETALK_FACE_EMBEDDING_MODEL", "buffalo_l")
    face_embedding_root: str = os.getenv(
        "MUSETALK_FACE_EMBEDDING_ROOT",
        str(PROJECT_DIR / "models" / "insightface"),
    )
    face_embedding_det_size: int = int(os.getenv("MUSETALK_FACE_EMBEDDING_DET_SIZE", "640"))
    progress_enabled: bool = os.getenv("API_PROGRESS", "1").lower() not in {"0", "false", "no", "off"}
    # CodeFormer face-restoration postprocess.
    codeformer_checkpoint_path: str = os.getenv(
        "MUSETALK_CODEFORMER_CKPT",
        str(PROJECT_DIR / "models" / "codeformer" / "codeformer.pth"),
    )
    codeformer_preload: bool = os.getenv("MUSETALK_CODEFORMER_PRELOAD", "0").lower() in {"1", "true", "yes"}
    codeformer_batch_size: int = int(os.getenv("MUSETALK_CODEFORMER_BATCH_SIZE", "8"))
    codeformer_required: bool = os.getenv("MUSETALK_CODEFORMER_REQUIRED", "0").lower() in {"1", "true", "yes"}


settings = Settings()
logger = logging.getLogger("musetalk.api")


def _resolve_vae_type(vae_type: str) -> str:
    if vae_type:
        candidate_path = Path(vae_type)
        if candidate_path.is_absolute() and (candidate_path / "config.json").is_file():
            return str(candidate_path)
        model_path = PROJECT_DIR / "models" / vae_type
        if (model_path / "config.json").is_file():
            return str(model_path)
        return vae_type
    for candidate in ("sd-vae", "sd-vae-ft-mse"):
        model_path = PROJECT_DIR / "models" / candidate
        if (model_path / "config.json").is_file():
            return str(model_path)
    return "sd-vae-ft-mse"


def _resolve_model_file(path_value: str, candidates: List[str]) -> str:
    if path_value:
        path = Path(path_value)
        if path.is_absolute() and path.is_file():
            return str(path)
        repo_path = PROJECT_DIR / path_value
        if repo_path.is_file():
            return str(repo_path)

    for candidate in candidates:
        candidate_path = PROJECT_DIR / candidate
        if candidate_path.is_file():
            return str(candidate_path)

    return str(PROJECT_DIR / candidates[0])


def _resolve_musetalk_model_paths(unet_config: str, unet_model_path: str) -> Tuple[str, str]:
    model_layouts = [
        ("models/musetalkV15/musetalk.json", "models/musetalkV15/unet.pth"),
        ("models/musetalk/musetalk.json", "models/musetalk/pytorch_model.bin"),
    ]
    for config_candidate, model_candidate in model_layouts:
        config_path = PROJECT_DIR / config_candidate
        model_path = PROJECT_DIR / model_candidate
        if config_path.is_file() and model_path.is_file():
            return str(config_path), str(model_path)

    return (
        _resolve_model_file(
            unet_config,
            ["models/musetalkV15/musetalk.json", "models/musetalk/musetalk.json"],
        ),
        _resolve_model_file(
            unet_model_path,
            ["models/musetalkV15/unet.pth", "models/musetalk/pytorch_model.bin"],
        ),
    )


def _resolve_model_dir(path_value: str, candidates: List[str], required_files: List[str]) -> str:
    candidate_dirs = []
    if path_value:
        path = Path(path_value)
        candidate_dirs.append(path if path.is_absolute() else PROJECT_DIR / path_value)
    candidate_dirs.extend(PROJECT_DIR / candidate for candidate in candidates)

    for candidate_dir in candidate_dirs:
        if all((candidate_dir / filename).is_file() for filename in required_files):
            return str(candidate_dir)

    fallback_dir = candidate_dirs[0]
    missing = [filename for filename in required_files if not (fallback_dir / filename).is_file()]
    if missing:
        logger.warning("Model directory %s is missing required files: %s", fallback_dir, ", ".join(missing))
    return str(fallback_dir)


def _require_files(directory: str, required_files: List[str], label: str) -> None:
    directory_path = Path(directory)
    missing = [filename for filename in required_files if not (directory_path / filename).is_file()]
    if missing:
        raise FileNotFoundError(
            f"{label} directory {directory_path} is missing required files: {', '.join(missing)}"
        )


settings.vae_type = _resolve_vae_type(settings.vae_type)
settings.unet_config, settings.unet_model_path = _resolve_musetalk_model_paths(
    settings.unet_config,
    settings.unet_model_path,
)
settings.whisper_dir = _resolve_model_dir(
    settings.whisper_dir,
    ["models/whisper"],
    ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
)


class LipSyncRequest(BaseModel):
    # Field order is kept identical to LatentSync api.py:94-285 for
    # cross-repository client compatibility (the frontend at
    # ~/Downloads/dub can switch backends without re-validating).
    video_url: str = Field(..., description="Source video URL")
    avatar_url: Optional[str] = Field(None, description="Reference avatar image URL")
    audio_url: str = Field(..., description="Driving audio URL")
    similarity_threshold: float = Field(0.35, ge=0.0, le=1.0)
    identity_margin: float = Field(0.03, ge=0.0, le=1.0)
    identity_cluster_threshold: float = Field(0.78, ge=0.0, le=1.0)
    default_identity_min_coverage: float = Field(0.5, ge=0.0, le=1.0)
    require_face_embedding: bool = True
    allow_crop_embedding_fallback: bool = True
    crop_embedding_min_detection_score: float = Field(0.0, ge=0.0, le=1.0)
    temporal_tracking_weight: float = Field(0.08, ge=0.0, le=0.5)
    target_fill_max_gap_seconds: float = Field(0.8, ge=0.0, le=3.0)
    target_fill_window_seconds: float = Field(2.0, ge=0.1, le=10.0)
    target_fill_min_match_ratio: float = Field(0.30, ge=0.0, le=1.0)
    target_fill_max_center_shift: float = Field(0.8, ge=0.0, le=5.0)
    target_motion_gate_enabled: bool = True
    target_motion_max_center_shift: float = Field(1.30, ge=0.0, le=5.0)
    target_motion_max_scale_change: float = Field(0.70, ge=0.0, le=2.0)
    target_fast_motion_gate_enabled: bool = True
    target_fast_motion_max_center_shift_per_frame: float = Field(0.35, ge=0.0, le=2.0)
    target_fast_motion_max_scale_change_per_frame: float = Field(0.20, ge=0.0, le=2.0)
    target_fast_motion_min_run_frames: int = Field(2, ge=1, le=120)
    lipsync_continuity_max_gap_seconds: float = Field(1.20, ge=0.0, le=2.0)
    lipsync_continuity_max_center_shift: float = Field(1.00, ge=0.0, le=5.0)
    lipsync_continuity_max_scale_change: float = Field(0.70, ge=0.0, le=2.0)
    # Independent knobs for the second-pass continuity fill so it
    # does not silently inherit the initial gap-fill's window /
    # match-ratio. The second pass runs after motion + fast-motion +
    # mouth-diff filters, so it usually wants a slightly larger
    # window and a lower match ratio to recover bridged segments.
    lipsync_continuity_window_seconds: float = Field(2.0, ge=0.1, le=10.0)
    lipsync_continuity_min_match_ratio: float = Field(0.20, ge=0.0, le=1.0)
    # Mouth-region pixel diff break: complementary to the embedding
    # similarity check. When the mouth region mean abs diff between
    # consecutive aligned face crops exceeds this fraction, treat the
    # next frame as a continuity break -- catches face switches the
    # embedding check misses (similar-looking people, side faces).
    # 0 disables. Default 0.60 sits well above both same-person
    # speech motion (open/close transitions reach ~0.30-0.50) and
    # typical cross-person diff (0.10-0.30), so it only fires on
    # hard face switches; lower to ~0.50 to be more aggressive,
    # raise or set to 0 to effectively disable.
    lipsync_mouth_diff_break_threshold: float = Field(
        0.60, ge=0.0, le=1.0,
        description="Mouth-region mean abs diff break threshold; 0 disables.",
    )
    target_bbox_smoothing_window: int = Field(7, ge=1, le=15)
    target_bbox_smoothing_max_center_shift: float = Field(0.85, ge=0.0, le=5.0)
    identity_scan_interval: int = Field(0, ge=0, le=300, description="0 means scan about 2 frames per second")
    identity_scan_max_frames: int = Field(0, ge=0, description="0 means scan all sampled identity frames")
    identity_scan_require_landmark_match: bool = False
    min_detection_score: float = Field(0.30, ge=0.0, le=1.0)
    require_landmark_match: bool = True
    min_landmark_points: int = Field(8, ge=1, le=68)
    min_landmark_overlap: float = Field(0.08, ge=0.0, le=1.0)
    lipsync_min_segment_frames: int = Field(1, ge=1, le=300)
    lipsync_min_face_area_ratio: float = Field(0.005, ge=0.0, le=1.0)
    bbox_shift: int = 0
    extra_margin: int = Field(10, ge=0, le=100)
    parsing_mode: str = "jaw"
    blend_upper_boundary_ratio: float = Field(0.58, ge=0.0, le=1.0)
    blend_mask_blur_ratio: float = Field(0.015, ge=0.0, le=0.2)
    color_match_strength: float = Field(0.70, ge=0.0, le=1.0)
    mouth_detail_strength: float = Field(0.90, ge=0.0, le=1.0)
    mouth_sharpen_strength: float = Field(0.30, ge=0.0, le=1.0)
    mouth_temporal_stabilization_strength: float = Field(0.08, ge=0.0, le=0.6)
    mouth_temporal_stabilization_max_delta: float = Field(0.12, ge=0.0, le=2.0)
    # Inpaint mask override. None = use the server-side default.
    # MuseTalk does not consume this field (encoder-decoder pipeline,
    # not diffusion inpainting); it is accepted for API compatibility
    # with LatentSync and logged when non-default.
    mask_image_path: Optional[str] = Field(
        None,
        description="Override the inpaint mask path. None = use server default.",
    )
    # Quality gate. Disabled by default (matching LatentSync's
    # conservative posture -- a difficult frame falls back to the
    # source, never returns smeared lips).
    quality_gate_enabled: bool = False
    quality_min_laplacian: float = Field(0.02, ge=0.0, le=2000.0)
    quality_min_sharpness_ratio: float = Field(0.03, ge=0.0, le=1.0)
    quality_ref_min_laplacian: float = Field(
        0.50,
        ge=0.0,
        le=2000.0,
        description="Only apply generated/reference sharpness-ratio fallback when the source mouth ROI is at least this sharp.",
    )
    quality_max_fallback_ratio: float = Field(
        0.80,
        ge=0.0,
        le=1.0,
        description="Disable quality fallback for this run if it would skip more than this fraction of non-prefiltered frames.",
    )
    # Mouth-region postfilter. Catches a single large blurry patch
    # in the generated mouth (CodeFormer failure, VAE collapse)
    # that the whole-image quality gate misses because the
    # surrounding face still looks fine. When the mouth-ROI
    # Laplacian variance drops below this threshold, the frame
    # falls back to the source. 0 disables. Reasonable values
    # for 256x256 face crops sit in the 1.0-5.0 range; tune
    # upward if too many false-positive fallbacks.
    # Mouth-region sharpness floor. Catches a single large
    # BLURRY patch in the generated mouth (CodeFormer failure,
    # VAE collapse, model color block) where the color is
    # similar to the surrounding face but the local sharpness
    # has dropped. Complements the color histogram check above
    # which only catches color shifts, not blurriness. 0
    # disables. Default 0 (OFF): the Laplacian check tends to
    # over-fire on normal lip motion (a 32x32 mouth tile that
    # moved 20px between frames already pushes the variance
    # below the floor), which collapses effective throughput to
    # near-zero. Enable per-request only when you have a
    # specific badcase (e.g. CodeFormer failure producing
    # blurred faces).
    quality_mouth_min_laplacian: float = Field(
        0.0,
        ge=0.0,
        le=2000.0,
        description="Mouth-region Laplacian floor. 0 (default) disables. Raise to 5-20 to catch blurred mouths at the cost of rejecting normal lip motion.",
    )
    # Light color-block check. Compares the COLOR DISTRIBUTION
    # (per-channel histogram) of the upper face (y < 55%) between
    # the post-processed crop and the reference. Returns the sum
    # of per-channel CHISQR distances (0 = identical, higher =
    # more different). Unlike the MSE checks above, this metric
    # is more about "did the color shift significantly" than
    # "did any single pixel change" -- so it tolerates normal
    # per-pixel variation from lip motion and only fires on a
    # hard color block. 0 disables. Default 0.5 is a LIGHT
    # check (catches obvious color blocks, passes clean
    # lipsync). Lower to 0.2 for stricter, raise to 1.0+ for
    # more permissive.
    quality_max_face_color_histogram_distance: float = Field(
        0.5,
        ge=0.0,
        le=10.0,
        description="Upper-face color histogram CHISQR distance ceiling. 0 disables. Default 0.5 = light check.",
    )
    # Drift fallback. Compares the post-processed face crop to
    # the reference, excluding the deep mouth ROI (y 55-80%, x
    # 30-70%). The "non-mouth" region covers the upper face, the
    # lip-border band right around the lips, the jaw and the
    # chin. When the mean squared error exceeds this threshold,
    # the frame falls back to the source (no lip-sync paste).
    # Catches post-processing over-shifts that produce a visible
    # color band around the lips, or generations that drifted to
    # the wrong identity. 0 disables (default). MSE is a coarse
    # metric that cannot reliably distinguish a MuseTalk color
    # block from legitimate lip-motion variation, so the
    # default is OFF and clients opt in by setting a value.
    # 500-1000 catches obvious drift; 100-300 is strict and
    # will start catching clean generations.
    quality_max_face_outside_mouth_mse: float = Field(
        0.0,
        ge=0.0,
        le=10000.0,
        description="Face-crop MSE ceiling outside the mouth ROI. 0 (default) disables.",
    )
    # Localized color-block fallback. Divides the face crop into
    # tiles (default 32x32) and computes per-tile MSE vs the
    # reference. If any single tile exceeds this threshold, the
    # frame falls back to source. 0 disables (default). The
    # 32x32 tile size is too small to tolerate normal lip
    # motion: a tile that shifted 20px between frames already
    # pushes the max-tile MSE well past 500, so enabling this
    # in default requests collapses effective throughput to
    # near-zero. Either lower the tile size to make this
    # smoother (e.g. pass 64 in code), or run it with a much
    # higher ceiling (5000+) per-request. The intended use
    # case is a one-shot badcase, not a default safety net.
    quality_max_face_tile_mse: float = Field(
        0.0,
        ge=0.0,
        le=10000.0,
        description="Per-tile MSE ceiling on the face crop. 0 (default) disables. Enable per-request with a high ceiling to catch obvious color blocks.",
    )
    # Source-face motion-blur prefilter. Catches frames where the
    # detected face is too blurry (camera motion, fast head turn)
    # to produce useful lip-sync output. When the face-crop
    # Laplacian variance is below this threshold, the frame is
    # treated as having no target and falls through to passthrough.
    # 0 disables. Reasonable values for ~200x200 face crops sit
    # in the 30-100 range; raise if too aggressive.
    prefilter_min_face_laplacian: float = Field(
        0.0,
        ge=0.0,
        le=2000.0,
        description="Source-face Laplacian floor. 0 disables. Frame is skipped (passthrough) if the source face-crop variance is below this.",
    )
    # Side-face (near-profile) prefilter. When the detected face
    # bbox's width/height aspect ratio exceeds this value, the
    # face is considered near-profile and the frame is skipped
    # (passthrough). MuseTalk is trained mostly on front/3-quarter
    # views and produces poor lipsync output (color blocks,
    # wrong mouth shape) for near-profile faces; skipping avoids
    # the artifact. 0 disables. Default 1.3 catches obvious
    # side profiles while leaving front/3-quarter views alone.
    prefilter_side_face_aspect_ratio: float = Field(
        1.3,
        ge=0.0,
        le=5.0,
        description="Source-face bbox aspect ratio (w/h) above which the face is considered side-profile and the frame is skipped. 0 disables. Default 1.3.",
    )
    # Cross-frame face lock (no-avatar path). When > 0, the per-
    # frame mouth-openness picker filters candidates to those
    # whose bbox has IoU >= this value against the previous
    # frame's accepted bbox. Prevents the lipsync target from
    # switching to a different person when the camera pans
    # between two faces (the "face switching" artifact). 0
    # disables (free picking each frame). IoU is sensitive to
    # bbox overlap and breaks under fast camera/head motion;
    # prefer the center-shift ratio below for motion-heavy
    # videos.
    target_track_min_iou: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Cross-frame face lock IoU floor (no-avatar path). 0 disables. Strict bbox-overlap lock; prefer the center-shift ratio below for motion-heavy videos.",
    )
    # Cross-frame face lock via bbox center distance (no-avatar
    # path). Filters candidates to those whose bbox center is
    # within ``this value * (previous face width)`` of the
    # previous frame's bbox center. More robust than IoU for
    # motion-heavy videos (camera pan / head turn) because it
    # only depends on center distance, not on bbox shape or
    # overlap. 0 disables. Per-request 1.5 = strict (lock only
    # when face stays within 1.5 face widths), 3.0 = loose
    # (tolerate more motion), 5.0+ = essentially disabled
    # without setting 0.
    target_track_max_center_shift_ratio: float = Field(
        2.0,
        ge=0.0,
        le=20.0,
        description="Cross-frame face lock via bbox center distance, normalized by previous face width. 0 disables. Default 2.0 = face within 2 face widths of last frame.",
    )
    # Output-level temporal blend. After all post-processing,
    # mixes the current face crop with the previous frame's face
    # crop. Cures the per-frame content jitter that bbox
    # smoothing alone cannot fix: the bbox position is stable,
    # but the generated mouth shape / texture still shakes
    # between frames. 0 disables (current per-frame behavior).
    # Per-request 0.2-0.3 is a light smooth, 0.4-0.5 is heavy
    # and may ghost on fast motion.
    # Output-level temporal blend. After all post-processing,
    # mixes the current face crop with the previous frame's face
    # crop. 0 disables (default). Bbox smoothing already locks
    # position, and the previous default (0.20) was layered on
    # top -- in practice this mainly smeared the lipsync output
    # against the source frame, costing more than it saved. Use
    # per-request 0.20-0.30 only if the source itself is
    # jittering for reasons the bbox gate cannot see.
    # Output-level temporal blend. After all post-processing,
    # mixes the current face crop with the previous frame's face
    # crop. 0 disables. Default 0.12 = light cross-frame blend
    # to smooth the per-frame content jitter that bbox
    # smoothing alone cannot fix (bbox position is stable, but
    # the generated mouth shape / texture still shakes between
    # frames). At 0.12 the previous frame contributes just
    # enough to damp the worst per-frame flickers without
    # ghosting on fast motion. Raise to 0.20-0.30 for heavier
    # smoothing; lower toward 0 if you see trailing.
    output_temporal_blend: float = Field(
        0.12,
        ge=0.0,
        le=0.9,
        description="Output-level temporal blend with the previous frame. 0.12 (default) for light smoothing.",
    )
    # Side-face / fast-turn prefilters (diffusion-only). MuseTalk does
    # not currently implement yaw-based skipping; values are accepted
    # for API compatibility and logged when non-default.
    yaw_skip_threshold: float = Field(45.0, ge=0.0, le=90.0)
    yaw_rate_skip_threshold: float = Field(28.0, ge=0.0, le=45.0)
    side_face_episode_pre_pad: int = Field(0, ge=0, le=30)
    side_face_episode_post_pad: int = Field(0, ge=0, le=30)
    yaw_warn_threshold_ratio: float = Field(0.75, ge=0.0, le=1.0)
    side_face_warn_min_run_frames: int = Field(
        0,
        ge=0,
        le=120,
        description="Skip sustained near-profile runs above the yaw warn threshold; 0 disables.",
    )
    # Per-request inference overrides. None = use server-side setting
    # (LatentSync uses these for a 质量预设 group fast/balanced/quality).
    # MuseTalk is single-step, so guidance/inference_steps/deepcache
    # have no effect; values are accepted for schema compatibility and
    # logged when non-default. seed affects torch/numpy RNG for the
    # inference path when set.
    guidance_scale_override: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="Classifier-free guidance scale. None = use server default. (Diffusion-only; no effect in MuseTalk.)"
    )
    inference_steps_override: Optional[int] = Field(
        None, ge=1, le=100, description="DDIM inference steps. None = use server default. (Diffusion-only; no effect in MuseTalk.)"
    )
    seed_override: Optional[int] = Field(
        None, description="RNG seed (-1 for random). None = use server default."
    )
    enable_deepcache_override: Optional[bool] = Field(
        None, description="Hint for DeepCache enable. None = use server default. (Diffusion-only; no effect in MuseTalk.)"
    )
    # Mouth-occlusion prefilter. MuseTalk does not currently implement
    # an occlusion detector; value is accepted for schema compatibility.
    mouth_occlusion_skip_threshold: float = Field(1.0, ge=0.0, le=1.0)
    # Motion-blur input filter. Not currently implemented in MuseTalk;
    # value is accepted for schema compatibility.
    motion_blur_skip_threshold: float = Field(0.08, ge=0.0, le=10.0)
    face_jump_center_threshold: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Skip frames whose landmark center jumps by more than this fraction of face size; 0 disables.",
    )
    face_jump_scale_threshold: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description="Skip frames whose landmark face scale changes abruptly by more than this fraction; 0 disables.",
    )
    left_cheek_width: int = Field(75, ge=1, le=240)
    right_cheek_width: int = Field(75, ge=1, le=240)
    batch_size: int = Field(8, ge=1, le=64)
    audio_padding_length_left: int = Field(2, ge=0, le=10)
    audio_padding_length_right: int = Field(2, ge=0, le=10)
    audio_sync_offset_seconds: float = Field(0.0, ge=-0.5, le=0.5)
    audio_feature_fps: float = Field(
        0.0,
        ge=0.0,
        le=120.0,
        description="0 follows source fps, otherwise use this fps for Whisper audio features",
    )
    max_audio_feature_fps: float = Field(
        25.0,
        ge=0.0,
        le=120.0,
        description="0 disables capping; high-fps videos default to 25fps audio features",
    )
    # Speech gate is intentionally disabled by default to mirror
    # LatentSync's conservative posture -- the per-frame RMS gate can
    # mask out soft speech on noisy audio. The MuseTalk impl remains
    # available; clients that want it can opt in.
    speech_gate_enabled: bool = False
    speech_gate_relative_db: float = Field(-38.0, ge=-80.0, le=0.0)
    speech_gate_min_rms: float = Field(0.0005, ge=0.0, le=1.0)
    speech_gate_window_seconds: float = Field(0.12, ge=0.02, le=0.5)
    speech_gate_pre_roll_seconds: float = Field(0.04, ge=0.0, le=1.0)
    speech_gate_post_roll_seconds: float = Field(0.12, ge=0.0, le=1.0)
    speech_gate_fill_gap_seconds: float = Field(0.16, ge=0.0, le=1.0)
    # --- CodeFormer face-restoration postprocess ---
    codeformer_enabled: bool = Field(
        False,
        description="Run CodeFormer face restoration on the aligned face crops before paste-back.",
    )
    codeformer_fidelity_weight: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="CodeFormer fidelity weight. 0 = sharpest (most codebook-driven, identity drift), "
                    "1 = closest to input. Default 0.7 keeps the restored face close to the "
                    "inpainter's output so per-frame variation stays low (the source of the "
                    "flicker CodeFormer 0.5 produced). The remaining color block fix is carried "
                    "by the per-frame post-process (skin-only detail restore, mouth->skin color "
                    "match, mouth CLAHE) so the fidelity bump does not lose the cleanup. "
                    "Lower (0.4-0.6) for stronger restoration if you also enable "
                    "codeformer_temporal_alpha to tame the flicker. Higher (0.85-0.95) almost "
                    "disables CodeFormer while keeping its safeguard path.",
    )
    codeformer_temporal_alpha: float = Field(
        0.7,
        ge=0.0,
        le=1.0,
        description="Cross-frame EMA on the CodeFormer-restored face crops. After the model "
                    "restores each face, blend it with the previous frame's restored face at "
                    "this ratio (alpha = current, 1-alpha = previous). 1.0 disables (raw "
                    "CodeFormer output, may flicker). 0.5-0.7 dampens the per-frame "
                    "variation that causes flicker without lagging lipsync too much. Set 0 "
                    "to skip the EMA (legacy behavior).",
    )
    codeformer_adain: bool = Field(
        True,
        description="Apply adaptive instance normalization so restored face color matches input.",
    )
    codeformer_required: bool = Field(
        settings.codeformer_required,
        description="If True and codeformer_enabled=True, fail the request when the CodeFormer checkpoint is missing.",
    )


class FaceListRequest(BaseModel):
    # Defaults mirror LatentSync api.py:288-299 (favor recall -- lower
    # thresholds and a per-frame scan so we surface as many distinct
    # faces as possible for the user to pick from).
    video_url: str = Field(..., description="Source video URL")
    similarity_threshold: float = Field(0.78, ge=0.0, le=1.0)
    frame_sample_interval: int = Field(1, ge=0, le=300, description="0 means sample about 2 frames per second; 1 scans every frame")
    max_frames: int = Field(0, ge=0, description="0 means scan all sampled frames")
    min_face_area: int = Field(100, ge=1)
    min_detection_score: float = Field(0.35, ge=0.0, le=1.0)
    require_face_embedding: bool = False
    require_landmark_match: bool = False
    min_landmark_points: int = Field(8, ge=1, le=68)
    min_landmark_overlap: float = Field(0.08, ge=0.0, le=1.0)
    crop_padding: float = Field(0.8, ge=0.0, le=1.5)


app = FastAPI(title="MuseTalk lip-sync API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_ROOT)), name="outputs")


@app.exception_handler(HTTPException)
async def log_http_exception(request: Request, exc: HTTPException):
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(
        "HTTP error while handling %s %s: %s\n%s",
        request.method,
        request.url,
        exc.detail,
        stack,
    )
    response = await http_exception_handler(request, exc)
    if isinstance(response, JSONResponse):
        return JSONResponse(
            status_code=exc.status_code,
            content=jsonable_encoder({"detail": exc.detail, "traceback": stack}),
            headers=exc.headers,
        )
    return response


@app.exception_handler(RequestValidationError)
async def log_validation_exception(request: Request, exc: RequestValidationError):
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.error(
        "Validation error while handling %s %s: %s\n%s",
        request.method,
        request.url,
        exc.errors(),
        stack,
    )
    return JSONResponse(
        status_code=422,
        content=jsonable_encoder({"detail": exc.errors(), "traceback": stack}),
    )


@app.exception_handler(Exception)
async def log_unhandled_exception(request: Request, exc: Exception):
    stack = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    logger.exception("Unhandled error while handling %s %s", request.method, request.url)
    return JSONResponse(
        status_code=500,
        content=jsonable_encoder({
            "detail": str(exc),
            "error_type": type(exc).__name__,
            "traceback": stack,
        }),
    )


def _check_ffmpeg() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


def _ensure_ffmpeg() -> None:
    if _check_ffmpeg():
        return
    path_separator = ";" if os.name == "nt" else ":"
    os.environ["PATH"] = f"{settings.ffmpeg_path}{path_separator}{os.environ.get('PATH', '')}"
    if not _check_ffmpeg():
        raise RuntimeError("ffmpeg was not found. Install ffmpeg or set FFMPEG_PATH.")


def _validate_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail=f"Invalid URL: {url}")


def _describe_target_identity_source(
    avatar_descriptor: Optional[Dict[str, object]],
    target_identity: Optional[Dict[str, object]],
) -> str:
    """Return a human-readable label for the response field
    `target_identity_source`.

    Possible values:
      - "avatar": avatar provided AND a matching cluster was found.
      - "avatar_fallback:<most_frequent_face|first_largest_face>":
        avatar was provided but no cluster cleared the similarity
        threshold, so we fell back to the no-avatar heuristic.
      - "largest_face_per_frame": kept for backward compatibility;
        older runs may still report this string.
      - "most_open_mouth_per_frame": no avatar provided; per-frame
        loop picks the face with the most open mouth in each
        frame, without cross-frame identity matching.
      - "<most_frequent_face|first_largest_face>": no avatar
        provided; heuristic pick from the source video (legacy
        no-avatar path; not used by the default flow anymore).
      - "none": no face detected in the source video (passthrough).
    """
    if target_identity is None:
        return "none"
    selection_source = target_identity.get("selection_source")
    if selection_source == "most_open_mouth_per_frame":
        return "most_open_mouth_per_frame"
    if selection_source == "largest_face_per_frame":
        return "largest_face_per_frame"
    if avatar_descriptor is not None and selection_source in (
        "most_frequent_face",
        "first_largest_face",
    ):
        return f"avatar_fallback:{selection_source}"
    if avatar_descriptor is not None:
        return "avatar"
    return str(selection_source or "most_frequent_face")


class _RetryableDownloadError(Exception):
    pass


def _download_attempt_count() -> int:
    return max(1, settings.download_retries + 1)


def _is_retryable_download_status(status_code: int) -> bool:
    return status_code in {408, 425, 429} or 500 <= status_code < 600


def _is_retryable_download_error(exc: Exception) -> bool:
    if isinstance(exc, _RetryableDownloadError):
        return True
    if isinstance(exc, requests.HTTPError):
        response = getattr(exc, "response", None)
        status_code = response.status_code if response is not None else None
        return status_code is None or _is_retryable_download_status(status_code)
    return isinstance(exc, requests.RequestException)


def _download_retry_delay(attempt_index: int) -> float:
    return max(0.0, settings.download_retry_backoff_seconds) * (2 ** attempt_index)


def _get_download_response_once(url: str) -> requests.Response:
    response = requests.get(url, stream=True, timeout=(10, 120))
    if _is_retryable_download_status(response.status_code):
        status_code = response.status_code
        response.close()
        raise _RetryableDownloadError(f"HTTP {status_code}")
    response.raise_for_status()
    return response


def _get_download_response(url: str, label: str) -> requests.Response:
    attempts = _download_attempt_count()
    last_error = None
    attempts_made = 0
    for attempt_index in range(attempts):
        attempts_made = attempt_index + 1
        try:
            return _get_download_response_once(url)
        except _RetryableDownloadError as exc:
            last_error = exc
        except requests.RequestException as exc:
            last_error = exc

        if not _is_retryable_download_error(last_error):
            break
        if attempt_index >= attempts - 1:
            break
        logger.warning(
            "Failed to download %s on attempt %s/%s: %s; retrying",
            label,
            attempt_index + 1,
            attempts,
            last_error,
        )
        time.sleep(_download_retry_delay(attempt_index))

    raise HTTPException(
        status_code=400,
        detail=f"Failed to download {label} after {attempts_made} attempts: {last_error}",
    )


def _guess_suffix(url: str, content_type: str, allowed: set, fallback: str) -> str:
    suffix = Path(unquote(urlparse(url).path)).suffix.lower()
    if suffix in allowed:
        return suffix
    guessed = mimetypes.guess_extension((content_type or "").split(";")[0].strip())
    if guessed and guessed.lower() in allowed:
        return guessed.lower()
    return fallback


def _download_to_file(url: str, dest_dir: Path, prefix: str, allowed: set, fallback: str) -> Path:
    local_path = _local_output_from_url(url)
    if local_path is not None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        suffix = local_path.suffix.lower()
        if suffix not in allowed:
            suffix = fallback
        output_path = dest_dir / f"{prefix}{suffix}"
        shutil.copyfile(local_path, output_path)
        return output_path

    _validate_url(url)
    dest_dir.mkdir(parents=True, exist_ok=True)
    attempts = _download_attempt_count()
    last_error = None
    attempts_made = 0
    for attempt_index in range(attempts):
        attempts_made = attempt_index + 1
        response = None
        temp_path = None
        try:
            response = _get_download_response_once(url)
            content_length = response.headers.get("content-length")
            if content_length and int(content_length) > settings.max_download_bytes:
                raise HTTPException(status_code=413, detail=f"{prefix} is larger than API_MAX_DOWNLOAD_BYTES")

            suffix = _guess_suffix(url, response.headers.get("content-type", ""), allowed, fallback)
            output_path = dest_dir / f"{prefix}{suffix}"
            temp_path = dest_dir / f"{prefix}{suffix}.part"
            downloaded = 0
            with temp_path.open("wb") as file_obj:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    downloaded += len(chunk)
                    if downloaded > settings.max_download_bytes:
                        raise HTTPException(status_code=413, detail=f"{prefix} is larger than API_MAX_DOWNLOAD_BYTES")
                    file_obj.write(chunk)
            if downloaded == 0:
                # Server returned 2xx but no bytes -- treat as retryable
                # so a transient CDN/TCP glitch gets a second chance
                # instead of silently producing a 0-byte input file.
                raise _RetryableDownloadError("Downloaded 0 bytes")
            logger.info(
                "[Download] %s -> %s (%d bytes)",
                prefix, output_path, downloaded,
            )
            temp_path.replace(output_path)
            return output_path
        except HTTPException:
            raise
        except _RetryableDownloadError as exc:
            last_error = exc
        except (requests.RequestException, OSError) as exc:
            last_error = exc
        finally:
            if response is not None:
                response.close()
            if temp_path is not None and temp_path.exists():
                temp_path.unlink(missing_ok=True)

        if not _is_retryable_download_error(last_error):
            break
        if attempt_index >= attempts - 1:
            break
        logger.warning(
            "Failed to download %s on attempt %s/%s: %s; retrying",
            prefix,
            attempt_index + 1,
            attempts,
            last_error,
        )
        time.sleep(_download_retry_delay(attempt_index))

    raise HTTPException(
        status_code=400,
        detail=f"Failed to download {prefix} after {attempts_made} attempts: {last_error}",
    )


def _read_video_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    if not video_path.exists():
        raise RuntimeError(f"Video file does not exist: {video_path}")
    file_size = video_path.stat().st_size
    if file_size == 0:
        raise RuntimeError(
            f"Video file is empty (0 bytes): {video_path} -- "
            "the upstream download likely failed silently; check the URL and API_MAX_DOWNLOAD_BYTES."
        )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video with OpenCV: {video_path} "
            f"(size={file_size} bytes) -- the file is likely truncated, "
            "or its codec/container is not supported by this OpenCV build "
            "(check ffmpeg linkage and the file with `ffprobe` / `file`)."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 1:
        fps = 25.0

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        raise RuntimeError(
            f"No frames were decoded from video: {video_path} "
            f"(size={file_size} bytes) -- file is likely truncated or uses an unsupported codec."
        )
    return frames, fps


def _clip_box(bbox: Tuple[int, int, int, int], frame_shape: Tuple[int, int, int]) -> Optional[Tuple[int, int, int, int]]:
    height, width = frame_shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(0, min(width, x2))
    y2 = max(0, min(height, y2))
    if x2 - x1 < 8 or y2 - y1 < 8:
        return None
    return x1, y1, x2, y2


def _box_area(bbox: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def _box_center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _box_iou(left: Tuple[int, int, int, int], right: Tuple[int, int, int, int]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    inter_x1 = max(lx1, rx1)
    inter_y1 = max(ly1, ry1)
    inter_x2 = min(lx2, rx2)
    inter_y2 = min(ly2, ry2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    if inter_area == 0:
        return 0.0
    union_area = _box_area(left) + _box_area(right) - inter_area
    return inter_area / union_area if union_area else 0.0


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm < 1e-8:
        return vector.astype(np.float32)
    return (vector / norm).astype(np.float32)


def _progress(iterable, desc: str, total: Optional[int] = None, unit: str = "it"):
    if not settings.progress_enabled or tqdm is None:
        return iterable
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        unit=unit,
        dynamic_ncols=True,
        leave=True,
    )


class MuseTalkApiRuntime:
    def __init__(self) -> None:
        self.loaded = False
        self.detectors_loaded = False
        self.load_lock = threading.RLock()
        self.run_lock = threading.Lock()
        self.face_parser_cache: Dict[Tuple[int, int], FaceParsing] = {}
        self.face_embedder = None
        self.face_recognition_session = None
        self.face_recognition_input_name = ""
        self.face_recognition_output_name = ""
        self.face_embedding_loaded = False
        self.face_embedding_error = ""
        self.codeformer_restorer = None
        self.codeformer_load_attempted = False
        self.codeformer_load_error = ""

    def load_detectors(self) -> None:
        if self.detectors_loaded:
            return
        with self.load_lock:
            if self.detectors_loaded:
                return

            _ensure_ffmpeg()

            from musetalk.utils import preprocessing as preprocessing_module

            self.preprocessing = preprocessing_module
            self.face_alignment = preprocessing_module.fa
            self.pose_model = preprocessing_module.model
            self.coord_placeholder = preprocessing_module.coord_placeholder
            self._load_face_embedder()
            self.detectors_loaded = True

    def _load_face_embedder(self) -> None:
        if self.face_embedding_loaded:
            return
        backend = settings.face_embedding_backend
        if backend in {"0", "false", "no", "off", "none"}:
            self.face_embedding_loaded = True
            return

        try:
            from insightface.app import FaceAnalysis

            model_dir = Path(settings.face_embedding_root) / "models" / settings.face_embedding_model
            if not model_dir.exists():
                logger.warning(
                    "InsightFace model directory %s was not found. FaceAnalysis may try to download it.",
                    model_dir,
                )
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if torch.cuda.is_available()
                else ["CPUExecutionProvider"]
            )
            self.face_embedder = FaceAnalysis(
                name=settings.face_embedding_model,
                root=settings.face_embedding_root,
                providers=providers,
            )
            ctx_id = settings.gpu_id if torch.cuda.is_available() else -1
            det_size = (settings.face_embedding_det_size, settings.face_embedding_det_size)
            self.face_embedder.prepare(ctx_id=ctx_id, det_size=det_size)
            self._load_crop_face_embedder(providers)
        except Exception as exc:
            self.face_embedding_error = str(exc)
            self.face_embedder = None
            if backend in {"insightface", "required"}:
                raise RuntimeError(f"Failed to load InsightFace embedding backend: {exc}") from exc
        finally:
            self.face_embedding_loaded = True

    def _load_crop_face_embedder(self, providers: List[str]) -> None:
        model_path = (
            Path(settings.face_embedding_root)
            / "models"
            / settings.face_embedding_model
            / "w600k_r50.onnx"
        )
        if not model_path.is_file():
            return
        try:
            import onnxruntime as ort

            available_providers = set(ort.get_available_providers())
            session_providers = [provider for provider in providers if provider in available_providers]
            if not session_providers:
                session_providers = ["CPUExecutionProvider"]
            self.face_recognition_session = ort.InferenceSession(
                str(model_path),
                providers=session_providers,
            )
            self.face_recognition_input_name = self.face_recognition_session.get_inputs()[0].name
            self.face_recognition_output_name = self.face_recognition_session.get_outputs()[0].name
        except Exception as exc:
            self.face_embedding_error = str(exc)
            self.face_recognition_session = None

    def _get_codeformer_restorer(self):
        """Return the singleton CodeFormerRestorer, building it on first call."""
        if self.codeformer_restorer is not None:
            return self.codeformer_restorer, ""
        if self.codeformer_load_attempted and self.codeformer_load_error:
            # Retry if the checkpoint appeared on disk after the first attempt.
            if os.path.isfile(settings.codeformer_checkpoint_path):
                self.codeformer_load_error = ""
                self.codeformer_load_attempted = False
            else:
                return None, self.codeformer_load_error
        self.codeformer_load_attempted = True
        try:
            from musetalk.utils.codeformer_restorer import CodeFormerRestorer
        except Exception as exc:
            self.codeformer_load_error = f"failed to import musetalk.utils.codeformer_restorer: {exc}"
            logger.exception("[CodeFormer] %s", self.codeformer_load_error)
            return None, self.codeformer_load_error
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.codeformer_restorer = CodeFormerRestorer(
            checkpoint_path=settings.codeformer_checkpoint_path,
            device=device,
            batch_size=settings.codeformer_batch_size,
        )
        # Eagerly probe the load so we surface errors early.
        if not settings.codeformer_checkpoint_path or not os.path.isfile(settings.codeformer_checkpoint_path):
            self.codeformer_load_error = (
                f"CodeFormer checkpoint not found at {settings.codeformer_checkpoint_path!r}"
            )
            logger.warning("[CodeFormer] %s", self.codeformer_load_error)
        else:
            try:
                self.codeformer_restorer._ensure_loaded()
                if self.codeformer_restorer.is_loaded:
                    logger.info("[CodeFormer] Preloaded successfully")
                else:
                    self.codeformer_load_error = self.codeformer_restorer.load_error
            except Exception as exc:
                self.codeformer_load_error = f"CodeFormer load failed: {exc}"
                logger.exception("[CodeFormer] %s", self.codeformer_load_error)
        return self.codeformer_restorer, self.codeformer_load_error

    def load(self) -> None:
        if self.loaded:
            return
        self.load_detectors()
        with self.load_lock:
            if self.loaded:
                return

            self.device = torch.device(
                f"cuda:{settings.gpu_id}" if torch.cuda.is_available() else "cpu"
            )
            self.vae, self.unet, self.pe = load_all_model(
                unet_model_path=settings.unet_model_path,
                vae_type=settings.vae_type,
                unet_config=settings.unet_config,
                device=self.device,
            )
            self.timesteps = torch.tensor([0], device=self.device)

            if settings.use_float16:
                self.pe = self.pe.half()
                self.vae.vae = self.vae.vae.half()
                self.unet.model = self.unet.model.half()

            self.pe = self.pe.to(self.device)
            self.vae.vae = self.vae.vae.to(self.device)
            self.unet.model = self.unet.model.to(self.device)
            self.weight_dtype = self.unet.model.dtype

            _require_files(
                settings.whisper_dir,
                ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
                "Whisper",
            )
            self.audio_processor = AudioProcessor(feature_extractor_path=settings.whisper_dir)
            self.whisper = WhisperModel.from_pretrained(settings.whisper_dir)
            self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
            self.whisper.requires_grad_(False)

            if settings.codeformer_preload:
                logger.info("[CodeFormer] Preloading CodeFormer model as requested...")
                self._get_codeformer_restorer()

            self.loaded = True

    def _get_face_parser(self, left_cheek_width: int, right_cheek_width: int) -> FaceParsing:
        key = (left_cheek_width, right_cheek_width)
        if key not in self.face_parser_cache:
            if settings.version == "v15":
                self.face_parser_cache[key] = FaceParsing(
                    left_cheek_width=left_cheek_width,
                    right_cheek_width=right_cheek_width,
                )
            else:
                self.face_parser_cache[key] = FaceParsing()
        return self.face_parser_cache[key]

    def _detect_face_boxes(self, frame: np.ndarray) -> List[Tuple[Tuple[int, int, int, int], float]]:
        rgb_frame = frame[..., ::-1]
        try:
            detected = self.face_alignment.face_detector.detect_from_batch(np.asarray([rgb_frame]))
        except Exception:
            return []

        if not detected or len(detected[0]) == 0:
            return []

        boxes = []
        for item in detected[0]:
            if len(item) < 4:
                continue
            score = float(item[4]) if len(item) > 4 else 1.0
            if score < settings.face_confidence:
                continue
            clipped = _clip_box(tuple(item[:4]), frame.shape)
            if clipped is None:
                continue
            boxes.append((clipped, score))

        boxes.sort(key=lambda item: item[1] * max(1, _box_area(item[0])), reverse=True)
        return boxes

    def _pose_face_landmarks(self, frame: np.ndarray) -> List[np.ndarray]:
        try:
            results = self.preprocessing.inference_topdown(self.pose_model, frame)
            results = self.preprocessing.merge_data_samples(results)
            keypoints = getattr(results.pred_instances, "keypoints", None)
        except Exception:
            return []

        if keypoints is None or len(keypoints) == 0:
            return []
        if torch.is_tensor(keypoints):
            keypoints = keypoints.detach().cpu().numpy()

        landmarks = []
        for person_keypoints in keypoints:
            if len(person_keypoints) >= 91:
                landmarks.append(person_keypoints[23:91].astype(np.int32))
        return landmarks

    def _landmark_bbox_for_face(
        self,
        landmarks: List[np.ndarray],
        detector_bbox: Tuple[int, int, int, int],
        bbox_shift: int,
        frame_shape: Tuple[int, int, int],
    ) -> Tuple[int, int, int, int]:
        if not landmarks:
            return detector_bbox

        face_cx, face_cy = _box_center(detector_bbox)
        face_scale = max(1.0, math.sqrt(_box_area(detector_bbox)))
        best_landmark = None
        best_distance = float("inf")

        for landmark in landmarks:
            landmark_bbox = (
                int(np.min(landmark[:, 0])),
                int(np.min(landmark[:, 1])),
                int(np.max(landmark[:, 0])),
                int(np.max(landmark[:, 1])),
            )
            landmark_cx, landmark_cy = _box_center(landmark_bbox)
            distance = math.hypot(landmark_cx - face_cx, landmark_cy - face_cy) / face_scale
            if distance < best_distance:
                best_distance = distance
                best_landmark = landmark

        if best_landmark is None or best_distance > 1.5:
            return detector_bbox

        half_face_coord = best_landmark[29].copy()
        if bbox_shift != 0:
            half_face_coord[1] = bbox_shift + half_face_coord[1]
        half_face_dist = np.max(best_landmark[:, 1]) - half_face_coord[1]
        upper_bond = max(0, half_face_coord[1] - half_face_dist)
        landmark_bbox = (
            int(np.min(best_landmark[:, 0])),
            int(upper_bond),
            int(np.max(best_landmark[:, 0])),
            int(np.max(best_landmark[:, 1])),
        )
        clipped = _clip_box(landmark_bbox, frame_shape)
        return clipped if clipped is not None else detector_bbox

    def _face_embedding(
        self,
        image: np.ndarray,
        reference_bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Optional[np.ndarray]:
        if self.face_embedder is None or image.size == 0:
            return None
        try:
            faces = self.face_embedder.get(image)
        except Exception as exc:
            self.face_embedding_error = str(exc)
            return None
        if not faces:
            return None

        def face_weight(face) -> float:
            bbox = getattr(face, "bbox", None)
            if bbox is None:
                area = 1.0
                overlap = 0.0
            else:
                x1, y1, x2, y2 = bbox
                area = max(1.0, float((x2 - x1) * (y2 - y1)))
                overlap = _box_iou(tuple(bbox), reference_bbox) if reference_bbox else 0.0
            return (overlap * 1000.0 + 1.0) * float(getattr(face, "det_score", 1.0)) * area

        face = max(faces, key=face_weight)
        if reference_bbox is not None:
            bbox = getattr(face, "bbox", None)
            if bbox is None or _box_iou(tuple(bbox), reference_bbox) <= 0.0:
                return None
        embedding = getattr(face, "normed_embedding", None)
        if embedding is None:
            return None
        return _normalize_vector(np.asarray(embedding, dtype=np.float32))

    def _crop_face_embedding(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if self.face_recognition_session is None or crop.size == 0:
            return None
        try:
            resized = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
            blob = cv2.dnn.blobFromImage(
                resized,
                scalefactor=1.0 / 127.5,
                size=(112, 112),
                mean=(127.5, 127.5, 127.5),
                swapRB=True,
            )
            embedding = self.face_recognition_session.run(
                [self.face_recognition_output_name],
                {self.face_recognition_input_name: blob},
            )[0]
        except Exception as exc:
            self.face_embedding_error = str(exc)
            return None
        return _normalize_vector(np.asarray(embedding, dtype=np.float32).reshape(-1))

    def _spatial_histogram(self, hsv: np.ndarray, grid_size: int = 3) -> np.ndarray:
        height, width = hsv.shape[:2]
        features = []
        for row in range(grid_size):
            y1 = row * height // grid_size
            y2 = (row + 1) * height // grid_size
            for col in range(grid_size):
                x1 = col * width // grid_size
                x2 = (col + 1) * width // grid_size
                cell = hsv[y1:y2, x1:x2]
                hist = cv2.calcHist([cell], [0, 1], None, [8, 6], [0, 180, 0, 256])
                hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
                features.append(hist)
        return _normalize_vector(np.concatenate(features))

    def _lbp_histogram(self, gray: np.ndarray) -> np.ndarray:
        center = gray[1:-1, 1:-1]
        code = np.zeros(center.shape, dtype=np.uint8)
        neighbors = [
            gray[:-2, :-2],
            gray[:-2, 1:-1],
            gray[:-2, 2:],
            gray[1:-1, 2:],
            gray[2:, 2:],
            gray[2:, 1:-1],
            gray[2:, :-2],
            gray[1:-1, :-2],
        ]
        for bit, neighbor in enumerate(neighbors):
            code |= ((neighbor >= center).astype(np.uint8) << bit)
        hist = np.bincount(code.ravel(), minlength=256).astype(np.float32)
        total = float(np.sum(hist))
        return hist / total if total > 0 else hist

    def _gradient_histogram(self, gray: np.ndarray, grid_size: int = 4, bins: int = 8) -> np.ndarray:
        gray_f = gray.astype(np.float32) / 255.0
        grad_x = cv2.Sobel(gray_f, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray_f, cv2.CV_32F, 0, 1, ksize=3)
        magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=False)
        bin_index = np.floor(angle * bins / (2.0 * math.pi)).astype(np.int32) % bins

        height, width = gray.shape[:2]
        features = []
        for row in range(grid_size):
            y1 = row * height // grid_size
            y2 = (row + 1) * height // grid_size
            for col in range(grid_size):
                x1 = col * width // grid_size
                x2 = (col + 1) * width // grid_size
                cell_bins = bin_index[y1:y2, x1:x2].ravel()
                cell_weights = magnitude[y1:y2, x1:x2].ravel()
                hist = np.bincount(cell_bins, weights=cell_weights, minlength=bins).astype(np.float32)
                features.append(hist)
        return _normalize_vector(np.concatenate(features))

    def _descriptor_embedding(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        crop: np.ndarray,
        allow_crop_embedding_fallback: bool,
    ) -> Optional[np.ndarray]:
        embedding_crop = self._crop_face(frame, bbox, 0.25)
        if embedding_crop is None:
            embedding_crop = crop

        embedding = self._face_embedding(embedding_crop)
        if embedding is None and allow_crop_embedding_fallback:
            embedding = self._crop_face_embedding(embedding_crop)
        if embedding is None:
            embedding = self._face_embedding(frame, bbox)
        return embedding

    def _face_descriptor(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        embedding_only: bool = False,
        allow_crop_embedding_fallback: bool = True,
    ) -> Optional[Dict[str, np.ndarray]]:
        clipped = _clip_box(bbox, frame.shape)
        if clipped is None:
            return None
        x1, y1, x2, y2 = clipped
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        embedding = self._descriptor_embedding(
            frame,
            clipped,
            crop,
            allow_crop_embedding_fallback,
        )
        if embedding_only:
            return {"embedding": embedding} if embedding is not None else None

        crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 4], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
        spatial_hist = self._spatial_histogram(hsv)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        small = small - float(np.mean(small))
        dct = cv2.dct(small)[:12, :12].flatten()
        lbp = self._lbp_histogram(gray)
        grad = self._gradient_histogram(gray)

        descriptor = {
            "hist": hist,
            "spatial_hist": spatial_hist,
            "dct": _normalize_vector(dct),
            "lbp": lbp,
            "grad": grad,
        }
        if embedding is not None:
            descriptor["embedding"] = embedding
        elif allow_crop_embedding_fallback:
            embedding = self._crop_face_embedding(crop)
            if embedding is not None:
                descriptor["embedding"] = embedding
        return descriptor

    def _is_reasonable_face_box(
        self,
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
    ) -> bool:
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= 0 or height <= 0:
            return False
        aspect_ratio = width / height
        area_ratio = _box_area(bbox) / max(1, frame_shape[0] * frame_shape[1])
        if aspect_ratio < 0.45 or aspect_ratio > 1.8:
            return False
        if area_ratio > 0.85:
            return False
        return True

    def _face_box_matches_landmarks(
        self,
        landmarks: List[np.ndarray],
        bbox: Tuple[int, int, int, int],
        frame_shape: Tuple[int, int, int],
        min_points: int,
        min_overlap: float,
    ) -> bool:
        if not landmarks:
            return False

        frame_height, frame_width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * 0.35)
        pad_y = int(height * 0.35)
        expanded = _clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), frame_shape)
        if expanded is None:
            return False

        ex1, ey1, ex2, ey2 = expanded
        face_center = np.array(_box_center(bbox), dtype=np.float32)
        face_scale = max(1.0, float(max(width, height)))

        for landmark in landmarks:
            points = np.asarray(landmark, dtype=np.float32).reshape(-1, 2)
            valid_mask = (
                (points[:, 0] > 1)
                & (points[:, 0] < frame_width)
                & (points[:, 1] > 1)
                & (points[:, 1] < frame_height)
            )
            valid_points = points[valid_mask]
            if len(valid_points) < min_points:
                continue

            inside_mask = (
                (valid_points[:, 0] >= ex1)
                & (valid_points[:, 0] <= ex2)
                & (valid_points[:, 1] >= ey1)
                & (valid_points[:, 1] <= ey2)
            )
            inside_points = valid_points[inside_mask]
            if len(inside_points) < min_points:
                continue

            landmark_center = np.mean(inside_points, axis=0)
            center_distance = np.linalg.norm(landmark_center - face_center) / face_scale
            if center_distance > 0.75:
                continue

            landmark_bbox = (
                int(np.min(inside_points[:, 0])),
                int(np.min(inside_points[:, 1])),
                int(np.max(inside_points[:, 0])),
                int(np.max(inside_points[:, 1])),
            )
            overlap = _box_iou(bbox, landmark_bbox)
            point_ratio = len(inside_points) / max(1, len(valid_points))
            if overlap >= min_overlap or point_ratio >= 0.45:
                return True

        return False

    def _descriptor_similarity(self, left: Dict[str, np.ndarray], right: Dict[str, np.ndarray]) -> float:
        if "embedding" in left and "embedding" in right:
            embedding_score = float(np.dot(left["embedding"], right["embedding"]))
            return float(np.clip((embedding_score + 1.0) / 2.0, 0.0, 1.0))

        hist_score = cv2.compareHist(left["hist"], right["hist"], cv2.HISTCMP_CORREL)
        hist_score = float(np.clip((hist_score + 1.0) / 2.0, 0.0, 1.0))
        spatial_score = float(np.dot(left["spatial_hist"], right["spatial_hist"]))
        spatial_score = float(np.clip((spatial_score + 1.0) / 2.0, 0.0, 1.0))
        dct_score = float(np.dot(left["dct"], right["dct"]))
        dct_score = float(np.clip((dct_score + 1.0) / 2.0, 0.0, 1.0))
        lbp_score = cv2.compareHist(
            left["lbp"].astype(np.float32),
            right["lbp"].astype(np.float32),
            cv2.HISTCMP_CORREL,
        )
        lbp_score = float(np.clip((lbp_score + 1.0) / 2.0, 0.0, 1.0))
        grad_score = float(np.dot(left["grad"], right["grad"]))
        grad_score = float(np.clip((grad_score + 1.0) / 2.0, 0.0, 1.0))
        return (
            0.15 * hist_score
            + 0.25 * spatial_score
            + 0.25 * dct_score
            + 0.15 * lbp_score
            + 0.20 * grad_score
        )

    def _avatar_descriptor(self, avatar_path: Path) -> Dict[str, np.ndarray]:
        avatar = cv2.imread(str(avatar_path))
        if avatar is None:
            raise RuntimeError(f"Could not read avatar image: {avatar_path}")

        descriptor = self._avatar_descriptor_from_image(avatar)
        if descriptor is None:
            raise RuntimeError("No face was detected in the avatar image.")
        return descriptor

    def _avatar_descriptor_from_image(self, avatar: np.ndarray) -> Optional[Dict[str, np.ndarray]]:
        boxes = self._detect_face_boxes(avatar)
        fallback_descriptor = None
        for bbox, _ in boxes:
            descriptor = self._face_descriptor(avatar, bbox)
            if descriptor is None:
                continue
            if "embedding" in descriptor:
                return descriptor
            if fallback_descriptor is None:
                fallback_descriptor = descriptor

        height, width = avatar.shape[:2]
        descriptor = self._face_descriptor(avatar, (0, 0, width, height))
        if descriptor is None:
            return fallback_descriptor
        if "embedding" in descriptor:
            return descriptor
        return fallback_descriptor or descriptor

    def _avatar_ready_face_crop(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        crop_padding: float,
        require_embedding: bool,
    ) -> Optional[np.ndarray]:
        paddings = []
        for padding in (crop_padding, 0.8, 1.0, 1.25):
            if padding not in paddings:
                paddings.append(padding)

        for padding in paddings:
            crop = self._crop_face(frame, bbox, padding)
            if crop is None:
                continue
            descriptor = self._avatar_descriptor_from_image(crop)
            if descriptor is None:
                continue
            if require_embedding and "embedding" not in descriptor:
                continue
            return crop
        return None

    def _passes_face_filters(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        detection_score: float,
        min_detection_score: float,
        landmarks: List[np.ndarray],
        require_landmark_match: bool,
        min_landmark_points: int,
        min_landmark_overlap: float,
        min_face_area: int = 0,
    ) -> bool:
        if detection_score < min_detection_score:
            return False
        if _box_area(bbox) < min_face_area:
            return False
        if not self._is_reasonable_face_box(bbox, frame.shape):
            return False
        if require_landmark_match and not self._face_box_matches_landmarks(
            landmarks,
            bbox,
            frame.shape,
            min_landmark_points,
            min_landmark_overlap,
        ):
            return False
        return True

    def _crop_face(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        padding: float,
    ) -> Optional[np.ndarray]:
        x1, y1, x2, y2 = bbox
        width = x2 - x1
        height = y2 - y1
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        clipped = _clip_box((x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y), frame.shape)
        if clipped is None:
            return None
        cx1, cy1, cx2, cy2 = clipped
        crop = frame[cy1:cy2, cx1:cx2]
        return crop if crop.size else None

    def _cluster_similarity(self, cluster: Dict[str, object], descriptor: Dict[str, np.ndarray]) -> float:
        descriptors = cluster.get("descriptors") or []
        if not descriptors:
            return 0.0
        return max(self._descriptor_similarity(existing, descriptor) for existing in descriptors)

    def _add_face_to_clusters(
        self,
        clusters: List[Dict[str, object]],
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        descriptor: Dict[str, np.ndarray],
        frame_index: int,
        detection_score: float,
        similarity_threshold: float,
        crop_padding: float,
    ) -> None:
        area = _box_area(bbox)
        crop = self._crop_face(frame, bbox, crop_padding)
        if crop is None:
            return

        best_cluster = None
        best_score = -1.0
        for cluster in clusters:
            score = self._cluster_similarity(cluster, descriptor)
            if score > best_score:
                best_score = score
                best_cluster = cluster

        if best_cluster is None or best_score < similarity_threshold:
            clusters.append(
                {
                    "descriptors": [descriptor],
                    "max_area": area,
                    "best_crop": crop,
                    "best_bbox": bbox,
                    "best_frame_index": frame_index,
                    "best_detection_score": detection_score,
                    "first_frame_index": frame_index,
                    "first_area": area,
                    "count": 1,
                }
            )
            return

        best_cluster["count"] = int(best_cluster["count"]) + 1
        descriptors = best_cluster["descriptors"]
        if len(descriptors) < 12:
            descriptors.append(descriptor)
        if area > int(best_cluster["max_area"]):
            best_cluster["max_area"] = area
            best_cluster["best_crop"] = crop
            best_cluster["best_bbox"] = bbox
            best_cluster["best_frame_index"] = frame_index
            best_cluster["best_detection_score"] = detection_score

    def extract_distinct_faces(
        self,
        video_path: Path,
        output_dir: Path,
        payload: FaceListRequest,
    ) -> Dict[str, object]:
        self.load_detectors()
        with self.run_lock:
            frames, fps = _read_video_frames(video_path)
            sample_interval = payload.frame_sample_interval or max(1, int(round(fps / 2.0)))
            clusters: List[Dict[str, object]] = []
            scanned_frames = 0
            detections = 0
            rejected_low_score = 0
            rejected_shape = 0
            rejected_landmarks = 0
            rejected_embedding = 0
            rejected_avatar_crop = 0

            frame_indices = range(0, len(frames), sample_interval)
            total_scan_frames = len(frame_indices)
            if payload.max_frames:
                total_scan_frames = min(total_scan_frames, payload.max_frames)
            for frame_index in _progress(
                frame_indices,
                "scan faces",
                total=total_scan_frames,
                unit="frame",
            ):
                if payload.max_frames and scanned_frames >= payload.max_frames:
                    break

                frame = frames[frame_index]
                scanned_frames += 1
                landmarks = self._pose_face_landmarks(frame) if payload.require_landmark_match else []
                for bbox, detection_score in self._detect_face_boxes(frame):
                    if detection_score < payload.min_detection_score:
                        rejected_low_score += 1
                        continue
                    if not self._passes_face_filters(
                        frame,
                        bbox,
                        detection_score,
                        payload.min_detection_score,
                        landmarks,
                        payload.require_landmark_match,
                        payload.min_landmark_points,
                        payload.min_landmark_overlap,
                        payload.min_face_area,
                    ):
                        rejected_shape += 1
                        if payload.require_landmark_match:
                            rejected_landmarks += 1
                        continue
                    descriptor = self._face_descriptor(
                        frame,
                        bbox,
                        embedding_only=payload.require_face_embedding,
                    )
                    if descriptor is None:
                        rejected_shape += 1
                        continue
                    if payload.require_face_embedding and "embedding" not in descriptor:
                        rejected_embedding += 1
                        continue
                    detections += 1
                    self._add_face_to_clusters(
                        clusters,
                        frame,
                        bbox,
                        descriptor,
                        frame_index,
                        detection_score,
                        payload.similarity_threshold,
                        payload.crop_padding,
                    )

            clusters.sort(
                key=lambda item: (int(item["count"]), int(item["max_area"])),
                reverse=True,
            )
            faces_dir = output_dir / "faces"
            faces_dir.mkdir(parents=True, exist_ok=True)

            face_paths = []
            face_items = []
            for cluster in clusters:
                avatar_crop = self._avatar_ready_face_crop(
                    frames[int(cluster["best_frame_index"])],
                    cluster["best_bbox"],
                    payload.crop_padding,
                    payload.require_face_embedding,
                )
                if avatar_crop is None:
                    rejected_avatar_crop += 1
                    continue
                index = len(face_paths)
                face_path = faces_dir / f"face_{index:03d}.jpg"
                cv2.imwrite(str(face_path), avatar_crop)
                face_paths.append(face_path)
                face_items.append(
                    {
                        "path": face_path,
                        "max_area": int(cluster["max_area"]),
                        "frame_index": int(cluster["best_frame_index"]),
                        "detection_score": float(cluster["best_detection_score"]),
                        "count": int(cluster["count"]),
                    }
                )

            return {
                "face_paths": face_paths,
                "faces": face_items,
                "source_frame_count": len(frames),
                "frame_sample_interval": sample_interval,
                "scanned_frame_count": scanned_frames,
                "detected_face_count": detections,
                "rejected_low_score_count": rejected_low_score,
                "rejected_shape_count": rejected_shape,
                "rejected_landmark_count": rejected_landmarks,
                "rejected_embedding_count": rejected_embedding,
                "rejected_avatar_crop_count": rejected_avatar_crop,
                "face_identity_backend": "embedding" if payload.require_face_embedding else "visual",
            }

    def _select_target_bbox(
        self,
        frame: np.ndarray,
        avatar_descriptor: Optional[Dict[str, np.ndarray]],
        threshold: float,
        bbox_shift: int,
        identity_margin: float,
        min_detection_score: float = 0.0,
        require_landmark_match: bool = False,
        min_landmark_points: int = 8,
        min_landmark_overlap: float = 0.08,
        expected_descriptors: Optional[List[Dict[str, np.ndarray]]] = None,
        negative_descriptors: Optional[List[Dict[str, np.ndarray]]] = None,
        require_embedding: bool = False,
        allow_crop_embedding_fallback: bool = True,
        crop_embedding_min_detection_score: float = 0.0,
        previous_bbox: Optional[Tuple[int, int, int, int]] = None,
        temporal_tracking_weight: float = 0.0,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        face_boxes = self._detect_face_boxes(frame)
        if not face_boxes:
            return None, 0.0

        best_bbox = None
        best_score = -1.0
        best_identity_score = -1.0
        best_negative_score = 0.0
        second_score = -1.0
        for bbox, detection_score in face_boxes:
            if not self._passes_face_filters(
                frame,
                bbox,
                detection_score,
                min_detection_score,
                [],
                False,
                min_landmark_points,
                min_landmark_overlap,
            ):
                continue
            descriptor = self._face_descriptor(
                frame,
                bbox,
                embedding_only=require_embedding,
                allow_crop_embedding_fallback=(
                    allow_crop_embedding_fallback
                    and (not require_embedding or detection_score >= crop_embedding_min_detection_score)
                ),
            )
            if descriptor is None:
                continue
            if require_embedding and "embedding" not in descriptor:
                continue
            identity_score = (
                max(self._descriptor_similarity(expected, descriptor) for expected in expected_descriptors)
                if expected_descriptors
                else 0.0
            )
            avatar_score = (
                self._descriptor_similarity(avatar_descriptor, descriptor)
                if avatar_descriptor is not None
                else identity_score
            )
            if require_embedding and avatar_score < threshold:
                continue
            negative_score = (
                max(self._descriptor_similarity(negative, descriptor) for negative in negative_descriptors)
                if negative_descriptors
                else 0.0
            )
            tracking_score = _box_iou(previous_bbox, bbox) if previous_bbox is not None else 0.0
            score = 0.4 * avatar_score + 0.6 * identity_score + temporal_tracking_weight * tracking_score
            if score > best_score:
                second_score = best_score
                best_score = score
                best_identity_score = identity_score
                best_negative_score = negative_score
                best_bbox = bbox
            elif score > second_score:
                second_score = score

        if best_bbox is None or best_identity_score < threshold:
            return None, max(0.0, best_score)
        if best_negative_score > 0.0 and best_identity_score < best_negative_score + identity_margin:
            return None, max(0.0, best_score)
        if second_score >= 0.0 and best_score < second_score + identity_margin:
            return None, max(0.0, best_score)

        landmarks = self._pose_face_landmarks(frame) if require_landmark_match else []
        if require_landmark_match and not self._face_box_matches_landmarks(
            landmarks,
            best_bbox,
            frame.shape,
            min_landmark_points,
            min_landmark_overlap,
        ):
            landmarks = []
        target_bbox = self._landmark_bbox_for_face(landmarks, best_bbox, bbox_shift, frame.shape)
        return target_bbox, best_score

    @staticmethod
    def _mouth_openness_score(face_crop: np.ndarray) -> float:
        """Estimate mouth openness from a face crop (BGR uint8).

        Crops the mouth region (y 55-80%, x 30-70% of the face)
        and returns the standard deviation of grayscale intensity.
        Open mouth has more variation (lips + teeth + cavity) so
        the std is higher; a closed mouth is uniform skin tone and
        the std is low. Higher score = more open.

        Returns 0.0 on empty/invalid crops. The score is a
        within-frame relative ranking, not an absolute openness
        measurement, so cross-frame lighting changes are not an
        issue.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0
        h, w = face_crop.shape[:2]
        y0, y1 = int(h * 0.55), int(h * 0.80)
        x0, x1 = int(w * 0.30), int(w * 0.70)
        if y1 <= y0 or x1 <= x0:
            return 0.0
        mouth = face_crop[y0:y1, x0:x1]
        if mouth.size == 0:
            return 0.0
        gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        return float(np.std(gray))

    def _select_most_open_mouth_bbox(
        self,
        frame: np.ndarray,
        bbox_shift: int,
        min_detection_score: float,
        require_landmark_match: bool,
        min_landmark_points: int,
        min_landmark_overlap: float,
        previous_bbox: Optional[Tuple[int, int, int, int]] = None,
        track_min_iou: float = 0.0,
        track_max_center_shift_ratio: float = 0.0,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        """Pick the face whose mouth is most open in the current
        frame, with a soft cross-frame lock to keep the same
        target across consecutive frames. Used by the no-avatar
        fast path.

        The picker does TWO passes:

        1. Among candidates that pass the size/detection filters
           AND the cross-frame lock, pick the most-open-mouth.
           This is the "preferred" face (the tracked target).
        2. Among ALL candidates that pass the size/detection
           filters (lock ignored), pick the most-open-mouth.
           This is the "fallback" if no candidate passes the
           lock -- e.g., the tracked face briefly left the
           frame or moved outside the lock window.

        The preferred face wins when it exists; the fallback is
        only used when the lock filters out every candidate. This
        way, fast motion never causes the picker to give up --
        the lipsync keeps editing, but the lock still steers it
        toward the previously-tracked face when possible.

        Lock knobs (both default 0 = disabled):
          - ``track_min_iou > 0``: bbox IoU against
            ``previous_bbox``. Strict bbox-overlap check; breaks
            under bbox shape / scale changes.
          - ``track_max_center_shift_ratio > 0``: bbox center
            distance normalized by the previous face width.
            Robust to shape / scale changes; tolerates motion
            up to ``ratio`` face widths per frame.

        Returns ``(bbox, detection_score)`` or ``(None, 0.0)`` if
        no face passes the basic size/detection filters.
        """
        face_boxes = self._detect_face_boxes(frame)
        if not face_boxes:
            return None, 0.0

        landmarks = self._pose_face_landmarks(frame) if require_landmark_match else []
        prev_face_width = 1
        if previous_bbox is not None:
            prev_face_width = max(
                1, int(previous_bbox[2]) - int(previous_bbox[0])
            )

        def _passes_lock(bbox: Tuple[int, int, int, int]) -> bool:
            if previous_bbox is None:
                return True
            if track_min_iou > 0.0:
                if _box_iou(previous_bbox, bbox) < track_min_iou:
                    return False
            if track_max_center_shift_ratio > 0.0:
                pcx, pcy = _box_center(previous_bbox)
                ccx, ccy = _box_center(bbox)
                dist = math.hypot(pcx - ccx, pcy - ccy)
                if dist > track_max_center_shift_ratio * prev_face_width:
                    return False
            return True

        def _openness(bbox: Tuple[int, int, int, int]) -> float:
            x1, y1, x2, y2 = bbox
            crop = frame[
                max(0, y1):min(frame.shape[0], y2),
                max(0, x1):min(frame.shape[1], x2),
            ]
            return self._mouth_openness_score(crop)

        best_locked_bbox = None
        best_locked_openness = -1.0
        best_locked_detection_score = 0.0
        best_unlocked_bbox = None
        best_unlocked_openness = -1.0
        best_unlocked_detection_score = 0.0
        any_passed_filters = False
        for bbox, detection_score in face_boxes:
            if not self._passes_face_filters(
                frame,
                bbox,
                detection_score,
                min_detection_score,
                landmarks,
                require_landmark_match,
                min_landmark_points,
                min_landmark_overlap,
            ):
                continue
            any_passed_filters = True
            openness = _openness(bbox)
            if openness > best_unlocked_openness:
                best_unlocked_openness = openness
                best_unlocked_bbox = bbox
                best_unlocked_detection_score = float(detection_score)
            if _passes_lock(bbox) and openness > best_locked_openness:
                best_locked_openness = openness
                best_locked_bbox = bbox
                best_locked_detection_score = float(detection_score)

        if not any_passed_filters:
            return None, 0.0
        # Lock preferred, but fall back to the most-open-mouth
        # face overall if no candidate cleared the lock. This
        # keeps lipsync running on motion-heavy frames instead of
        # giving up.
        if best_locked_bbox is not None:
            best_bbox = best_locked_bbox
            best_detection_score = best_locked_detection_score
        else:
            best_bbox = best_unlocked_bbox
            best_detection_score = best_unlocked_detection_score

        if best_bbox is None:
            return None, 0.0

        if require_landmark_match and not self._face_box_matches_landmarks(
            landmarks,
            best_bbox,
            frame.shape,
            min_landmark_points,
            min_landmark_overlap,
        ):
            landmarks = []
        target_bbox = self._landmark_bbox_for_face(landmarks, best_bbox, bbox_shift, frame.shape)
        return target_bbox, best_detection_score

    def _find_target_identity(
        self,
        frames: List[np.ndarray],
        fps: float,
        avatar_descriptor: Optional[Dict[str, np.ndarray]],
        payload: LipSyncRequest,
    ) -> Optional[Dict[str, object]]:
        if avatar_descriptor is None:
            # No avatar: skip cross-frame identity matching entirely.
            # The per-frame loop picks the largest face in each frame
            # directly. This avoids unstable cluster behavior (low
            # coverage, count=2 from a noisy detector) and is the
            # right default when the caller has not specified a
            # target identity -- the largest face in a frame is
            # almost always the main subject.
            return {
                "selection_source": "most_open_mouth_per_frame",
                "avatar_score": 0.0,
                "count": 0,
                "identity_coverage": 0.0,
                "target_descriptors": [],
                "negative_descriptors": [],
                "no_face_detected": False,
            }
        clusters: List[Dict[str, object]] = []
        sample_interval = payload.identity_scan_interval or max(1, int(round(fps / 2.0)))
        frame_indices = range(0, len(frames), sample_interval)
        total_scan_frames = len(frame_indices)
        if payload.identity_scan_max_frames:
            total_scan_frames = min(total_scan_frames, payload.identity_scan_max_frames)

        scanned_identity_frames = 0
        for frame_index in _progress(
            frame_indices,
            "scan identity",
            total=total_scan_frames,
            unit="frame",
        ):
            if payload.identity_scan_max_frames and scanned_identity_frames >= payload.identity_scan_max_frames:
                break
            frame = frames[frame_index]
            scanned_identity_frames += 1
            landmarks = self._pose_face_landmarks(frame) if payload.identity_scan_require_landmark_match else []
            for bbox, detection_score in self._detect_face_boxes(frame):
                if not self._passes_face_filters(
                    frame,
                    bbox,
                    detection_score,
                    payload.min_detection_score,
                    landmarks,
                    payload.identity_scan_require_landmark_match,
                    payload.min_landmark_points,
                    payload.min_landmark_overlap,
                ):
                    continue
                descriptor = self._face_descriptor(
                    frame,
                    bbox,
                    embedding_only=payload.require_face_embedding,
                    allow_crop_embedding_fallback=(
                        payload.allow_crop_embedding_fallback
                        and detection_score >= payload.crop_embedding_min_detection_score
                    ),
                )
                if descriptor is None:
                    continue
                if payload.require_face_embedding and "embedding" not in descriptor:
                    continue
                self._add_face_to_clusters(
                    clusters,
                    frame,
                    bbox,
                    descriptor,
                    frame_index,
                    detection_score,
                    payload.identity_cluster_threshold,
                    0.0,
                )

        if not clusters:
            return {
                "avatar_score": 0.0,
                "count": 0,
                "target_descriptors": [],
                "negative_descriptors": [],
                "no_face_detected": True,
            }

        if avatar_descriptor is None:
            return self._pick_cluster_without_avatar(clusters, scanned_identity_frames, payload)

        for cluster in clusters:
            descriptors = cluster.get("descriptors") or []
            cluster["avatar_score"] = max(
                self._descriptor_similarity(avatar_descriptor, descriptor)
                for descriptor in descriptors
            )

        clusters.sort(
            key=lambda item: (
                float(item["avatar_score"]),
                int(item["count"]),
                int(item["max_area"]),
            ),
            reverse=True,
        )
        best_cluster = clusters[0]
        if float(best_cluster["avatar_score"]) < payload.similarity_threshold:
            # Avatar was provided but no cluster in the source video
            # matched it. Fall back to the same heuristic the no-avatar
            # path uses (most-frequent / first-largest) instead of
            # hard-stopping with zero matching frames and a misleading
            # `target_identity_source = "avatar"`.
            return self._pick_cluster_without_avatar(clusters, scanned_identity_frames, payload)

        target_descriptors = [
            descriptor
            for descriptor in (best_cluster.get("descriptors") or [])
            if self._descriptor_similarity(avatar_descriptor, descriptor) >= payload.similarity_threshold
        ]
        if not target_descriptors:
            target_descriptors = list(best_cluster.get("descriptors") or [])
        negative_descriptors = []
        for cluster in clusters[1:]:
            negative_descriptors.extend(cluster.get("descriptors") or [])
        best_cluster["target_descriptors"] = target_descriptors
        best_cluster["negative_descriptors"] = negative_descriptors
        return best_cluster

    def _pick_cluster_without_avatar(
        self,
        clusters: List[Dict[str, object]],
        scanned_identity_frames: int,
        payload: LipSyncRequest,
    ) -> Dict[str, object]:
        """Pick a target face cluster using only the source video.

        Used both when no avatar is provided AND when the avatar was
        provided but no cluster in the source video cleared the
        similarity threshold (fallback path).
        """
        clusters.sort(
            key=lambda item: (int(item["count"]), int(item["max_area"])),
            reverse=True,
        )
        best_cluster = clusters[0]
        most_frequent_coverage = int(best_cluster["count"]) / max(1, scanned_identity_frames)
        if most_frequent_coverage < payload.default_identity_min_coverage:
            clusters.sort(
                key=lambda item: (
                    -int(item.get("first_frame_index", item["best_frame_index"])),
                    int(item.get("first_area", item["max_area"])),
                ),
                reverse=True,
            )
            best_cluster = clusters[0]
            selection_source = "first_largest_face"
        else:
            selection_source = "most_frequent_face"
        identity_coverage = int(best_cluster["count"]) / max(1, scanned_identity_frames)
        best_cluster["avatar_score"] = 0.0
        best_cluster["selection_source"] = selection_source
        best_cluster["identity_coverage"] = identity_coverage
        best_cluster["most_frequent_identity_coverage"] = most_frequent_coverage
        best_cluster["target_descriptors"] = list(best_cluster.get("descriptors") or [])
        negative_descriptors = []
        for cluster in clusters[1:]:
            negative_descriptors.extend(cluster.get("descriptors") or [])
        best_cluster["negative_descriptors"] = negative_descriptors
        return best_cluster

    def _bbox_with_margin(
        self,
        bbox: Tuple[int, int, int, int],
        frame: np.ndarray,
        extra_margin: int,
    ) -> Optional[Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = bbox
        if settings.version == "v15":
            y2 = min(frame.shape[0], y2 + extra_margin)
        return _clip_box((x1, y1, x2, y2), frame.shape)

    def _bbox_center_shift(
        self,
        left: Tuple[int, int, int, int],
        right: Tuple[int, int, int, int],
    ) -> float:
        left_center = _box_center(left)
        right_center = _box_center(right)
        distance = math.hypot(left_center[0] - right_center[0], left_center[1] - right_center[1])
        left_scale = math.sqrt(max(1, _box_area(left)))
        right_scale = math.sqrt(max(1, _box_area(right)))
        return distance / max(1.0, (left_scale + right_scale) / 2.0)

    def _bbox_scale_change(
        self,
        left: Tuple[int, int, int, int],
        right: Tuple[int, int, int, int],
    ) -> float:
        left_scale = math.sqrt(max(1, _box_area(left)))
        right_scale = math.sqrt(max(1, _box_area(right)))
        return abs(left_scale - right_scale) / max(1.0, (left_scale + right_scale) / 2.0)

    def _interpolate_bbox(
        self,
        left: Tuple[int, int, int, int],
        right: Tuple[int, int, int, int],
        ratio: float,
        frame_shape: Tuple[int, int, int],
    ) -> Optional[Tuple[int, int, int, int]]:
        bbox = tuple(
            int(round(left[index] + (right[index] - left[index]) * ratio))
            for index in range(4)
        )
        return _clip_box(bbox, frame_shape)

    def _fill_short_target_gaps(
        self,
        targets: List[Dict[str, object]],
        fps: float,
        frame_shape: Tuple[int, int, int],
        max_gap_seconds: float,
        window_seconds: float,
        min_match_ratio: float,
        max_center_shift: float,
    ) -> int:
        if not targets or max_gap_seconds <= 0.0:
            return 0

        max_gap_frames = max(1, int(round(max_gap_seconds * fps)))
        window_frames = max(1, int(round(window_seconds * fps)))
        original_matches = [target.get("bbox") is not None for target in targets]
        filled = 0
        index = 0
        while index < len(targets):
            if original_matches[index]:
                index += 1
                continue

            gap_start = index
            while index < len(targets) and not original_matches[index]:
                index += 1
            gap_end = index
            gap_length = gap_end - gap_start
            left_index = gap_start - 1
            right_index = gap_end
            if left_index < 0 or right_index >= len(targets):
                continue
            if gap_length > max_gap_frames:
                continue

            left_bbox = targets[left_index].get("bbox")
            right_bbox = targets[right_index].get("bbox")
            if left_bbox is None or right_bbox is None:
                continue
            if max_center_shift > 0.0 and self._bbox_center_shift(left_bbox, right_bbox) > max_center_shift:
                continue

            gap_center = (gap_start + gap_end) // 2
            half_window = max(gap_length, window_frames // 2)
            window_start = max(0, min(left_index, gap_center - half_window))
            window_end = min(len(targets), max(right_index + 1, gap_center + half_window + 1))
            match_count = sum(1 for matched in original_matches[window_start:window_end] if matched)
            match_ratio = match_count / max(1, window_end - window_start)
            if match_ratio < min_match_ratio:
                continue

            left_score = float(targets[left_index].get("score") or 0.0)
            right_score = float(targets[right_index].get("score") or 0.0)
            for fill_index in range(gap_start, gap_end):
                ratio = (fill_index - left_index) / max(1, right_index - left_index)
                bbox = self._interpolate_bbox(left_bbox, right_bbox, ratio, frame_shape)
                if bbox is None:
                    continue
                targets[fill_index] = {
                    "bbox": bbox,
                    "score": min(left_score, right_score),
                    "filled": True,
                }
                filled += 1
        return filled

    def _filter_motion_targets(
        self,
        targets: List[Dict[str, object]],
        frame_shape: Tuple[int, int, int],
        enabled: bool,
        max_center_shift: float,
        max_scale_change: float,
    ) -> int:
        if not enabled or not targets:
            return 0

        valid_indices = [index for index, target in enumerate(targets) if target.get("bbox") is not None]
        if len(valid_indices) < 3:
            return 0

        filtered = 0
        for valid_position, index in enumerate(valid_indices):
            if valid_position == 0 or valid_position == len(valid_indices) - 1:
                continue

            previous_index = valid_indices[valid_position - 1]
            next_index = valid_indices[valid_position + 1]
            previous_bbox = targets[previous_index].get("bbox")
            next_bbox = targets[next_index].get("bbox")
            current_bbox = targets[index].get("bbox")
            if previous_bbox is None or next_bbox is None or current_bbox is None:
                continue

            ratio = (index - previous_index) / max(1, next_index - previous_index)
            expected_bbox = self._interpolate_bbox(previous_bbox, next_bbox, ratio, frame_shape)
            if expected_bbox is None:
                continue

            center_shift = self._bbox_center_shift(expected_bbox, current_bbox)
            scale_change = self._bbox_scale_change(expected_bbox, current_bbox)
            if (
                (max_center_shift > 0.0 and center_shift > max_center_shift)
                or (max_scale_change > 0.0 and scale_change > max_scale_change)
            ):
                targets[index] = {
                    **targets[index],
                    "bbox": None,
                    "filtered_reason": "motion_outlier",
                    "motion_center_shift": center_shift,
                    "motion_scale_change": scale_change,
                }
                filtered += 1
        return filtered

    def _filter_fast_motion_targets(
        self,
        targets: List[Dict[str, object]],
        enabled: bool,
        max_center_shift_per_frame: float,
        max_scale_change_per_frame: float,
        min_run_frames: int,
    ) -> int:
        if not enabled or not targets:
            return 0

        valid_indices = [index for index, target in enumerate(targets) if target.get("bbox") is not None]
        if len(valid_indices) < 2:
            return 0

        candidate_indices = set()
        previous_index = valid_indices[0]
        previous_bbox = targets[previous_index].get("bbox")
        for index in valid_indices[1:]:
            current_bbox = targets[index].get("bbox")
            if previous_bbox is None or current_bbox is None:
                previous_index = index
                previous_bbox = current_bbox
                continue

            frame_gap = max(1, index - previous_index)
            # Cap the gap at 2: when valid frames are sparse (other
            # filters or detection losses in between) dividing by the
            # raw frame_gap dilutes a real single-frame jump down to
            # near-zero, so genuine detection errors that need to be
            # filtered slip through. Capping keeps the per-frame
            # semantics for actual consecutive motion (gap == 1) and
            # a "within 2 frames" view for sparse gaps, matching the
            # spirit of the threshold.
            effective_gap = min(frame_gap, 2)
            center_shift = self._bbox_center_shift(previous_bbox, current_bbox) / effective_gap
            scale_change = self._bbox_scale_change(previous_bbox, current_bbox) / effective_gap
            if (
                (max_center_shift_per_frame > 0.0 and center_shift > max_center_shift_per_frame)
                or (max_scale_change_per_frame > 0.0 and scale_change > max_scale_change_per_frame)
            ):
                candidate_indices.add(index)
            previous_index = index
            previous_bbox = current_bbox

        filtered_indices = set()
        candidate_runs = []
        for index in sorted(candidate_indices):
            if not candidate_runs or index != candidate_runs[-1][-1] + 1:
                candidate_runs.append([index])
            else:
                candidate_runs[-1].append(index)

        min_run_frames = max(1, int(min_run_frames))
        for run in candidate_runs:
            if len(run) >= min_run_frames:
                filtered_indices.update(run)

        for index in filtered_indices:
            targets[index] = {
                **targets[index],
                "bbox": None,
                "filtered_reason": "fast_motion",
            }
        return len(filtered_indices)

    @staticmethod
    def _mouth_region_diff(prev_crop: np.ndarray, curr_crop: np.ndarray) -> float:
        """Mean absolute diff in the mouth band of two face crops.

        Both inputs are BGR uint8 numpy arrays. Returns the diff
        normalized to [0, 1]. The mouth band is the region y 55-74%,
        x 30-70% of the crop (same ROI as LatentSync's check).

        Returns 0.0 on None / empty / shape mismatch (defensive).
        """
        if prev_crop is None or curr_crop is None:
            return 0.0
        if prev_crop.size == 0 or curr_crop.size == 0:
            return 0.0
        if prev_crop.shape != curr_crop.shape:
            return 0.0
        try:
            h, w = prev_crop.shape[:2]
            y0, y1 = int(h * 0.55), int(h * 0.74)
            x0, x1 = int(w * 0.30), int(w * 0.70)
            if y1 <= y0 or x1 <= x0:
                return 0.0
            prev_mouth = prev_crop[y0:y1, x0:x1].astype(np.float32)
            curr_mouth = curr_crop[y0:y1, x0:x1].astype(np.float32)
            return float(np.mean(np.abs(curr_mouth - prev_mouth))) / 255.0
        except Exception:
            return 0.0

    def _filter_mouth_diff_targets(
        self,
        targets: List[Dict[str, object]],
        frames: List[np.ndarray],
        threshold: float,
    ) -> int:
        """Filter target frames where the mouth-region pixel diff
        between consecutive face crops exceeds the threshold.

        Catches face switches the embedding-similarity check misses
        (similar-looking people, side faces). Complementary to
        identity matching: embedding asks "same person?", pixel diff
        asks "same content?".
        """
        if threshold <= 0.0 or not targets or not frames:
            return 0

        valid_indices = [i for i, t in enumerate(targets) if t.get("bbox") is not None]
        if len(valid_indices) < 2:
            return 0

        filtered = 0
        prev_crop = None
        prev_index = -1
        for index in valid_indices:
            bbox = targets[index].get("bbox")
            if bbox is None or index >= len(frames):
                continue
            x1, y1, x2, y2 = bbox
            frame = frames[index]
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if crop.size == 0:
                prev_crop = None
                prev_index = index
                continue
            if prev_crop is not None and (index - prev_index) <= 3:
                diff = self._mouth_region_diff(prev_crop, crop)
                if diff > threshold:
                    targets[index] = {
                        **targets[index],
                        "bbox": None,
                        "filtered_reason": "mouth_diff_break",
                        "mouth_region_diff": diff,
                    }
                    filtered += 1
                    # Reset to None: a single large mouth-region
                    # diff is usually a normal open/close transition
                    # during speech, not a real face switch. Keeping
                    # the break crop as the next reference would make
                    # the *next* frame (mouth returning to neutral)
                    # also compare high against the open-mouth crop
                    # and get filtered too -- cascading false
                    # positives that remove lip-sync from every other
                    # frame. The 1-3 frame blind window here is
                    # intentional; bridged segments are recovered by
                    # the continuity fill (lipsync_continuity_*).
                    prev_crop = None
                    prev_index = index
                    continue
            prev_crop = crop
            prev_index = index
        return filtered

    def _filter_lipsync_targets(
        self,
        targets: List[Dict[str, object]],
        frame_shape: Tuple[int, int, int],
        min_segment_frames: int,
        min_face_area_ratio: float,
    ) -> Tuple[int, int]:
        if not targets:
            return 0, 0

        frame_area = max(1, int(frame_shape[0]) * int(frame_shape[1]))
        valid_targets = [False] * len(targets)
        small_face_frames = 0
        for index, target in enumerate(targets):
            bbox = target.get("bbox")
            if bbox is None:
                continue
            face_area_ratio = _box_area(bbox) / frame_area
            if face_area_ratio < min_face_area_ratio:
                targets[index] = {
                    **target,
                    "bbox": None,
                    "face_area_ratio": face_area_ratio,
                    "filtered_reason": "face_too_small",
                }
                small_face_frames += 1
                continue
            targets[index]["face_area_ratio"] = face_area_ratio
            valid_targets[index] = True

        short_segment_frames = 0
        index = 0
        min_segment_frames = max(1, int(min_segment_frames))
        while index < len(targets):
            if not valid_targets[index]:
                index += 1
                continue
            segment_start = index
            while index < len(targets) and valid_targets[index]:
                index += 1
            segment_end = index
            if segment_end - segment_start >= min_segment_frames:
                continue
            for filtered_index in range(segment_start, segment_end):
                targets[filtered_index] = {
                    **targets[filtered_index],
                    "bbox": None,
                    "filtered_reason": "short_target_segment",
                }
                short_segment_frames += 1
        return small_face_frames, short_segment_frames

    def _smooth_target_bboxes(
        self,
        targets: List[Dict[str, object]],
        frame_shape: Tuple[int, int, int],
        window_size: int,
        max_center_shift: float,
    ) -> int:
        if window_size <= 1 or not targets:
            return 0

        radius = window_size // 2
        original_bboxes = [target.get("bbox") for target in targets]
        smoothed_bboxes: Dict[int, Tuple[int, int, int, int]] = {}
        for index, bbox in enumerate(original_bboxes):
            if bbox is None:
                continue

            weighted_boxes = []
            start = max(0, index - radius)
            end = min(len(original_bboxes), index + radius + 1)
            for neighbor_index in range(start, end):
                neighbor_bbox = original_bboxes[neighbor_index]
                if neighbor_bbox is None:
                    continue
                if (
                    neighbor_index != index
                    and max_center_shift > 0.0
                    and self._bbox_center_shift(bbox, neighbor_bbox) > max_center_shift
                ):
                    continue
                weight = 2.0 if neighbor_index == index else 1.0
                weighted_boxes.append((neighbor_bbox, weight))

            if len(weighted_boxes) < 2:
                continue
            total_weight = sum(weight for _, weight in weighted_boxes)
            averaged = tuple(
                int(round(sum(box[coord] * weight for box, weight in weighted_boxes) / total_weight))
                for coord in range(4)
            )
            clipped = _clip_box(averaged, frame_shape)
            if clipped is not None and clipped != bbox:
                smoothed_bboxes[index] = clipped

        for index, bbox in smoothed_bboxes.items():
            targets[index] = {
                **targets[index],
                "bbox": bbox,
                "smoothed": True,
            }
        return len(smoothed_bboxes)

    def _encode_latents(
        self,
        frames: List[np.ndarray],
        targets: List[Dict[str, object]],
        extra_margin: int,
    ) -> Dict[int, torch.Tensor]:
        latents_by_frame: Dict[int, torch.Tensor] = {}
        for index, target in _progress(
            enumerate(targets),
            "encode latents",
            total=len(targets),
            unit="frame",
        ):
            bbox = target.get("bbox")
            if bbox is None:
                continue
            frame = frames[index]
            crop_bbox = self._bbox_with_margin(bbox, frame, extra_margin)
            if crop_bbox is None:
                continue
            x1, y1, x2, y2 = crop_bbox
            crop_frame = frame[y1:y2, x1:x2]
            if crop_frame.size == 0:
                continue
            crop_frame = cv2.resize(crop_frame, (256, 256), interpolation=cv2.INTER_LANCZOS4)
            latents_by_frame[index] = self.vae.get_latents_for_unet(crop_frame)
        return latents_by_frame

    def _source_index_for_output(self, output_index: int, frame_count: int) -> int:
        # Output frame count is tied to source video length (see
        # AGENTS.md), so this is a defensive identity mapping. The
        # older bounce logic is dead code; keep the min() so a
        # caller that overshoots frame_count clamps to the last
        # source frame instead of raising.
        if frame_count <= 0:
            return 0
        return min(max(int(output_index), 0), frame_count - 1)

    @staticmethod
    def _resolve_audio_feature_fps(source_fps: float, payload: LipSyncRequest) -> float:
        source_fps = float(source_fps or 25.0)
        if source_fps <= 0.0:
            source_fps = 25.0
        requested_fps = float(payload.audio_feature_fps or 0.0)
        if requested_fps > 0.0:
            return requested_fps
        max_feature_fps = float(payload.max_audio_feature_fps or 0.0)
        if max_feature_fps > 0.0:
            return min(source_fps, max_feature_fps)
        return source_fps

    @staticmethod
    def _audio_feature_index_for_output(
        output_index: int,
        source_fps: float,
        audio_feature_fps: float,
        audio_frame_count: int,
        offset_frames: int,
    ) -> int:
        if audio_frame_count <= 1:
            return 0
        source_fps = max(1e-6, float(source_fps))
        audio_feature_fps = max(1e-6, float(audio_feature_fps))
        time_seconds = (float(output_index) + 0.5) / source_fps
        audio_index = int(round(time_seconds * audio_feature_fps - 0.5)) + int(offset_frames)
        return min(max(audio_index, 0), audio_frame_count - 1)

    @staticmethod
    def _fill_activity_gaps(mask: np.ndarray, max_gap_frames: int) -> np.ndarray:
        if max_gap_frames <= 0 or not mask.any():
            return mask

        filled = mask.copy()
        index = 0
        while index < len(filled):
            if filled[index]:
                index += 1
                continue

            gap_start = index
            while index < len(filled) and not filled[index]:
                index += 1
            gap_end = index
            if gap_start == 0 or gap_end >= len(filled):
                continue
            if gap_end - gap_start <= max_gap_frames:
                filled[gap_start:gap_end] = True
        return filled

    @staticmethod
    def _pad_activity_mask(mask: np.ndarray, pre_roll_frames: int, post_roll_frames: int) -> np.ndarray:
        if (pre_roll_frames <= 0 and post_roll_frames <= 0) or not mask.any():
            return mask

        padded = mask.copy()
        index = 0
        while index < len(mask):
            if not mask[index]:
                index += 1
                continue

            segment_start = index
            while index < len(mask) and mask[index]:
                index += 1
            segment_end = index
            padded[
                max(0, segment_start - pre_roll_frames):min(len(mask), segment_end + post_roll_frames)
            ] = True
        return padded

    def _audio_activity_mask(
        self,
        audio_path: Path,
        fps: float,
        frame_count: int,
        enabled: bool,
        relative_db: float,
        min_rms: float,
        window_seconds: float,
        pre_roll_seconds: float,
        post_roll_seconds: float,
        fill_gap_seconds: float,
    ) -> Tuple[List[bool], Dict[str, object]]:
        if frame_count <= 0:
            return [], {"enabled": enabled, "active_frames": 0, "silent_frames": 0}
        if not enabled:
            return [True] * frame_count, {
                "enabled": False,
                "active_frames": frame_count,
                "silent_frames": 0,
            }
        fps = float(fps)
        if fps <= 0:
            raise ValueError(f"fps must be positive, got {fps}")

        try:
            import librosa

            audio, sample_rate = librosa.load(str(audio_path), sr=16000, mono=True)
        except Exception as exc:
            logger.warning("Could not compute speech gate for %s: %s", audio_path, exc, exc_info=True)
            return [True] * frame_count, {
                "enabled": False,
                "error": str(exc),
                "active_frames": frame_count,
                "silent_frames": 0,
            }

        if audio.size == 0:
            return [False] * frame_count, {
                "enabled": True,
                "active_frames": 0,
                "silent_frames": frame_count,
                "threshold_rms": 0.0,
                "peak_rms": 0.0,
                "noise_floor_rms": 0.0,
            }

        audio = audio.astype(np.float32, copy=False)
        half_window = max(1, int(round(window_seconds * sample_rate / 2.0)))
        rms_values = []
        for frame_index in range(frame_count):
            center = int(round((frame_index + 0.5) * sample_rate / fps))
            start = max(0, center - half_window)
            end = min(len(audio), center + half_window)
            if end <= start:
                rms_values.append(0.0)
                continue
            samples = audio[start:end]
            rms_values.append(float(np.sqrt(np.mean(np.square(samples)))))

        rms = np.asarray(rms_values, dtype=np.float32)
        nonzero_rms = rms[rms > 1e-8]
        if nonzero_rms.size == 0:
            return [False] * frame_count, {
                "enabled": True,
                "active_frames": 0,
                "silent_frames": frame_count,
                "threshold_rms": 0.0,
                "peak_rms": 0.0,
                "noise_floor_rms": 0.0,
            }

        peak_rms = float(np.percentile(nonzero_rms, 95))
        noise_floor = float(np.percentile(nonzero_rms, 20))
        relative_threshold = peak_rms * (10.0 ** (relative_db / 20.0))
        noise_spread = peak_rms / max(noise_floor, 1e-8)
        noise_threshold = noise_floor * 2.0 if noise_spread > 5.0 else noise_floor * 0.5
        threshold = max(float(min_rms), relative_threshold, noise_threshold)

        mask = rms >= threshold
        fill_gap_frames = int(round(fill_gap_seconds * fps))
        pre_roll_frames = int(round(pre_roll_seconds * fps))
        post_roll_frames = int(round(post_roll_seconds * fps))
        mask = self._fill_activity_gaps(mask, fill_gap_frames)
        mask = self._pad_activity_mask(mask, pre_roll_frames, post_roll_frames)

        active_frames = int(mask.sum())
        return mask.tolist(), {
            "enabled": True,
            "active_frames": active_frames,
            "silent_frames": frame_count - active_frames,
            "threshold_rms": threshold,
            "peak_rms": peak_rms,
            "noise_floor_rms": noise_floor,
        }

    def _run_inference_batches(
        self,
        process_items: List[Tuple[int, torch.Tensor, torch.Tensor]],
        batch_size: int,
    ) -> Dict[int, np.ndarray]:
        generated: Dict[int, np.ndarray] = {}
        batch_starts = range(0, len(process_items), batch_size)
        total_batches = math.ceil(len(process_items) / batch_size) if process_items else 0
        for start in _progress(batch_starts, "run inference", total=total_batches, unit="batch"):
            batch = process_items[start:start + batch_size]
            output_indices = [item[0] for item in batch]
            whisper_batch = torch.stack([item[1] for item in batch]).to(self.device)
            latent_batch = torch.cat([item[2] for item in batch], dim=0)
            latent_batch = latent_batch.to(device=self.device, dtype=self.unet.model.dtype)

            audio_feature_batch = self.pe(whisper_batch)
            pred_latents = self.unet.model(
                latent_batch,
                self.timesteps,
                encoder_hidden_states=audio_feature_batch,
            ).sample
            recon = self.vae.decode_latents(pred_latents)
            for output_index, result_frame in zip(output_indices, recon):
                generated[output_index] = result_frame
        return generated

    def _match_color_stats(
        self,
        image: np.ndarray,
        reference: np.ndarray,
        strength: float,
    ) -> np.ndarray:
        if strength <= 0.0 or image.size == 0 or reference.size == 0:
            return image
        if image.shape != reference.shape:
            return image

        image_float = image.astype(np.float32)
        reference_float = reference.astype(np.float32)
        # Compute color stats on the upper face only (y < 55%) so the
        # mouth area (y 55-100%) doesn't pull the stats toward lip/
        # teeth colors. Apply the color transfer ONLY to the upper
        # face -- the mouth is left untouched and keeps its
        # generated colors. A hard cutoff at 55% avoids the visible
        # color band a smooth-mask approach can produce in the
        # transition zone.
        height = image_float.shape[0]
        cutoff = int(height * 0.55)
        if cutoff <= 1 or cutoff >= height - 1:
            return image
        image_upper = image_float[:cutoff, :, :]
        reference_upper = reference_float[:cutoff, :, :]
        image_mean, image_std = cv2.meanStdDev(image_upper)
        reference_mean, reference_std = cv2.meanStdDev(reference_upper)
        image_mean = image_mean.reshape(1, 1, 3)
        image_std = image_std.reshape(1, 1, 3)
        reference_mean = reference_mean.reshape(1, 1, 3)
        reference_std = reference_std.reshape(1, 1, 3)
        matched_upper = (
            (image_upper - image_mean)
            * (reference_std / np.maximum(image_std, 1.0))
            + reference_mean
        )
        blended_upper = image_upper * (1.0 - strength) + matched_upper * strength
        image_float[:cutoff, :, :] = blended_upper
        return np.clip(image_float, 0, 255).astype(np.uint8)

    def _sharpen_image(self, image: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0.0 or image.size == 0:
            return image
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _match_mouth_to_skin_tone(
        self,
        image: np.ndarray,
        reference: np.ndarray,
        skin_mask: Optional[np.ndarray],
        strength: float = 0.40,
    ) -> np.ndarray:
        """Color-match the generated mouth to the reference's
        skin tone.

        MuseTalk's output mouth often sits at a different
        color temperature / brightness than the source subject
        (the inpainter was trained on a different skin-tone
        distribution). On a darker-skinned source the gap is
        especially obvious: the generated mouth reads as
        noticeably lighter than the rest of the face even
        though the rest of the face is fine. The upper-face
        color match does not see the mouth (hard-cut at 55%),
        and CLAHE only equalizes local contrast, not global
        brightness / temperature.

        Pull the generated mouth's per-channel mean and std
        toward the reference's *skin* mean / std (the source
        subject's true skin tone, not the mouth itself, which
        is closed in the reference and would over-tint the
        generated open mouth). Apply only inside the mouth
        region (skin_mask < 128) so the surrounding skin is
        untouched.
        """
        if strength <= 0.0 or image.size == 0 or reference.size == 0:
            return image
        if image.shape != reference.shape:
            return image
        if skin_mask is None or skin_mask.shape != image.shape[:2]:
            return image
        ref_skin_pixels = reference[skin_mask > 128]
        if ref_skin_pixels.size == 0 or ref_skin_pixels.shape[0] < 100:
            return image
        ref_mean = ref_skin_pixels.mean(axis=0)
        ref_std = ref_skin_pixels.std(axis=0)
        mouth_mask = (skin_mask < 128)
        if int(mouth_mask.sum()) < 100:
            return image
        gen_mouth_pixels = image[mouth_mask]
        gen_mean = gen_mouth_pixels.mean(axis=0)
        gen_std = gen_mouth_pixels.std(axis=0)
        # Per-channel scale + shift that maps gen_mouth's
        # distribution to ref_skin's. Strength blends: 0 keeps
        # gen, 1 fully matches.
        scale = ref_std / np.maximum(gen_std, 1.0)
        shift = ref_mean - gen_mean * scale
        image_float = image.astype(np.float32)
        out = image_float * (1.0 - strength + strength * scale) + strength * shift
        mouth_mask_3 = mouth_mask[:, :, None].astype(np.float32)
        result = image_float * (1.0 - mouth_mask_3) + out * mouth_mask_3
        return np.clip(result, 0, 255).astype(np.uint8)

    def _fix_mouth_color_block(
        self,
        image: np.ndarray,
        skin_mask: Optional[np.ndarray],
    ) -> np.ndarray:
        """Local color-block fix on the mouth region. MuseTalk's
        per-tile color drift shows up as hard color blocks in the
        mouth that the post-process masks cannot see (color match
        is hard-cut to the upper face; reference detail restore is
        gated to skin by the BiSeNet mask). The drift is a local
        contrast / brightness imbalance within the mouth ROI, so
        the right fix is a CLAHE-style local histogram equalization
        on the luminance channel -- and it must be applied to the
        mouth only, otherwise the surrounding skin tone would
        shift.

        Falls back to the input when the skin mask is missing or
        the mouth area is too small (< 100 px) to make the
        transform safe.
        """
        if skin_mask is None or image.size == 0:
            return image
        if skin_mask.shape != image.shape[:2]:
            return image
        mouth_mask = (skin_mask < 128).astype(np.uint8)
        if int(mouth_mask.sum()) < 100:
            return image
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        ycrcb[..., 0] = clahe.apply(ycrcb[..., 0])
        equalized = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        mouth_mask_f = mouth_mask.astype(np.float32) / 255.0
        mouth_mask_3 = mouth_mask_f[:, :, None]
        blended = (
            image.astype(np.float32) * (1.0 - mouth_mask_3)
            + equalized.astype(np.float32) * mouth_mask_3
        )
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _restore_reference_detail(
        self,
        image: np.ndarray,
        reference: np.ndarray,
        strength: float,
        skin_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if strength <= 0.0 or image.size == 0 or reference.size == 0:
            return image
        if image.shape != reference.shape:
            return image

        height = image.shape[0]
        if height <= 1:
            return image
        # Apply detail restore on the face but EXCLUDE the mouth /
        # lips / teeth / nose / eyes. Layering the reference's
        # high-frequency content on the generated *open* mouth
        # introduces a visible color block -- the reference is
        # closed-mouth, so its detail (lip lines, skin texture)
        # does not match the generated open-mouth area. Use a
        # skin-only mask (class 1 from BiSeNet / CelebAMask-HQ)
        # when available so the rest of the face still picks up
        # the reference's clean detail.
        reference_float = reference.astype(np.float32)
        reference_blur = cv2.GaussianBlur(reference_float, (0, 0), 1.0)
        detail = reference_float - reference_blur
        restored = image.astype(np.float32) + detail * strength
        if skin_mask is None:
            return np.clip(restored, 0, 255).astype(np.uint8)
        if skin_mask.shape != image.shape[:2]:
            return np.clip(restored, 0, 255).astype(np.uint8)
        skin_mask_f = skin_mask.astype(np.float32) / 255.0
        skin_mask_3 = skin_mask_f[:, :, None]
        result = image.astype(np.float32) * (1.0 - skin_mask_3) + restored * skin_mask_3
        return np.clip(result, 0, 255).astype(np.uint8)

    def _laplacian_variance(self, image: np.ndarray) -> float:
        if image.size == 0:
            return 0.0
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    @staticmethod
    def _soft_upper_mask(
        height: int,
        soft_start_ratio: float = 0.40,
        soft_end_ratio: float = 0.60,
    ) -> np.ndarray:
        """Soft upper-face mask, shape ``(height, 1, 1)``.

        Returns 1.0 above ``soft_start_ratio * height``, linear
        ramp down to 0 at ``soft_end_ratio * height``, 0 below.
        Used to apply upper-face post-processing (color match /
        detail restore / sharpen) without a hard seam at the
        upper/lower boundary.
        """
        mask = np.ones((height, 1, 1), dtype=np.float32)
        soft_start = int(height * soft_start_ratio)
        soft_end = int(height * soft_end_ratio)
        if soft_start < 0:
            soft_start = 0
        if soft_end > height:
            soft_end = height
        if soft_end > soft_start:
            ramp = np.linspace(1.0, 0.0, soft_end - soft_start, dtype=np.float32)
            mask[soft_start:soft_end, 0, 0] = ramp
            mask[soft_end:, 0, 0] = 0.0
        elif soft_start >= height:
            mask[:] = 0.0
        return mask

    @staticmethod
    def _face_color_histogram_distance(
        generated: np.ndarray,
        reference: np.ndarray,
        bins: int = 32,
    ) -> float:
        """Sum of per-channel CHISQR histogram distances on the
        upper face (y < 55%, outside the mouth ROI).

        Unlike MSE, this metric compares the COLOR DISTRIBUTION
        rather than per-pixel values. Normal lip motion shifts
        the per-pixel content of the mouth but barely changes
        the upper-face color distribution; a hard color block
        in the upper face, on the other hand, does shift the
        distribution noticeably. Returns 0.0 when histograms
        are identical; higher = more different.
        """
        if (
            generated is None
            or reference is None
            or generated.size == 0
            or reference.size == 0
        ):
            return 0.0
        if generated.shape != reference.shape:
            return 0.0
        h = generated.shape[0]
        cutoff = int(h * 0.55)
        if cutoff <= 1:
            return 0.0
        gen_upper = generated[:cutoff, :, :]
        ref_upper = reference[:cutoff, :, :]
        total = 0.0
        for channel in range(3):
            gen_hist = cv2.calcHist(
                [gen_upper], [channel], None, [bins], [0, 256]
            )
            ref_hist = cv2.calcHist(
                [ref_upper], [channel], None, [bins], [0, 256]
            )
            cv2.normalize(gen_hist, gen_hist)
            cv2.normalize(ref_hist, ref_hist)
            total += float(
                cv2.compareHist(gen_hist, ref_hist, cv2.HISTCMP_CHISQR)
            )
        return total

    @staticmethod
    def _face_max_tile_mse(
        generated: np.ndarray,
        reference: np.ndarray,
        tile_size: int = 32,
    ) -> float:
        """Max per-tile MSE between the generated and reference
        crops. Tiles are square ``tile_size x tile_size`` patches
        tiled across the face. Returns the worst tile's MSE, which
        is the right metric for catching LOCAL color blocks that
        the mean-MSE drift check (which dilutes with clean
        regions) misses.
        """
        if (
            generated is None
            or reference is None
            or generated.size == 0
            or reference.size == 0
        ):
            return 0.0
        if generated.shape != reference.shape:
            return 0.0
        h, w = generated.shape[:2]
        if tile_size <= 0 or tile_size > h or tile_size > w:
            return 0.0
        gen = generated.astype(np.float32)
        ref = reference.astype(np.float32)
        max_tile_mse = 0.0
        for y in range(0, h, tile_size):
            for x in range(0, w, tile_size):
                y1 = min(y + tile_size, h)
                x1 = min(x + tile_size, w)
                tile_diff_sq = (gen[y:y1, x:x1] - ref[y:y1, x:x1]) ** 2
                tile_mse = float(tile_diff_sq.mean())
                if tile_mse > max_tile_mse:
                    max_tile_mse = tile_mse
        return max_tile_mse

    @staticmethod
    def _face_outside_mouth_mse(generated: np.ndarray, reference: np.ndarray) -> float:
        """Mean squared error on the face crop, **excluding** the
        deep mouth ROI (y 55-80%, x 30-70%). The "non-mouth"
        region covers the upper face, the lip-border band right
        around the lips, the jaw, and the chin. A large MSE here
        means post-processing or the generation has drifted from
        the source in the area the user can actually see around
        the lips -- a reliable signal of a bad frame even when
        the deep mouth area itself looks plausible.
        """
        if (
            generated is None
            or reference is None
            or generated.size == 0
            or reference.size == 0
        ):
            return 0.0
        if generated.shape != reference.shape:
            return 0.0
        h, w = generated.shape[:2]
        y0 = int(h * 0.55)
        y1 = int(h * 0.80)
        x0 = int(w * 0.30)
        x1 = int(w * 0.70)
        gen = generated.astype(np.float32)
        ref = reference.astype(np.float32)
        # 0 inside the mouth ROI, 1 outside
        mask = np.ones((h, w), dtype=np.float32)
        if 0 < y0 < h and 0 < x0 < w and y0 < y1 and x0 < x1:
            mask[y0:y1, x0:x1] = 0.0
        weight = mask.sum()
        if weight < 1.0:
            return 0.0
        diff_sq = (gen - ref) ** 2
        mse_per_pixel = diff_sq.mean(axis=2)
        return float((mse_per_pixel * mask).sum() / weight)

    @staticmethod
    def _mouth_region_laplacian(face_crop: np.ndarray) -> float:
        """Laplacian variance over the mouth ROI of a face crop.

        Used by the postfilter to detect a blurry-patch mouth even
        when the surrounding face looks fine. Mouth region follows
        the same y 55-80% / x 30-70% as
        ``_mouth_openness_score``.
        """
        if face_crop is None or face_crop.size == 0:
            return 0.0
        h, w = face_crop.shape[:2]
        y0, y1 = int(h * 0.55), int(h * 0.80)
        x0, x1 = int(w * 0.30), int(w * 0.70)
        if y1 <= y0 or x1 <= x0:
            return 0.0
        mouth = face_crop[y0:y1, x0:x1]
        if mouth.size == 0:
            return 0.0
        gray = cv2.cvtColor(mouth, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _is_low_quality_generation(
        self,
        image: np.ndarray,
        reference: np.ndarray,
        min_laplacian: float,
        min_sharpness_ratio: float,
    ) -> bool:
        image_sharpness = self._laplacian_variance(image)
        if image_sharpness >= min_laplacian:
            return False

        reference_sharpness = self._laplacian_variance(reference)
        if reference_sharpness <= 1e-6:
            return False

        return image_sharpness / reference_sharpness < min_sharpness_ratio

    def _write_result_frames(
        self,
        frames: List[np.ndarray],
        targets: List[Dict[str, object]],
        generated: Dict[int, np.ndarray],
        output_frame_count: int,
        output_dir: Path,
        extra_margin: int,
        parsing_mode: str,
        blend_upper_boundary_ratio: float,
        blend_mask_blur_ratio: float,
        color_match_strength: float,
        mouth_detail_strength: float,
        mouth_sharpen_strength: float,
        output_temporal_blend: float,
        quality_gate_enabled: bool,
        quality_min_laplacian: float,
        quality_min_sharpness_ratio: float,
        quality_mouth_min_laplacian: float,
        quality_max_face_outside_mouth_mse: float,
        quality_max_face_tile_mse: float,
        quality_max_face_color_histogram_distance: float,
        face_parser: Optional[FaceParsing],
    ) -> List[str]:
        """Write every output frame to ``output_dir`` and return a
        per-frame provenance list of length ``output_frame_count``.

        Each entry is one of:
          - ``"passthrough"``: source frame written unchanged (no
            target, no inference result, blend material missing,
            invalid crop, or speech-gate-silent).
          - ``"quality_fallback"``: inference ran and was blended,
            but the per-frame quality gate replaced it with the
            source frame.
          - ``"blend_error"``: inference ran but ``get_image_blending``
            raised; the source frame was written as a defensive
            fallback.
          - ``"generated"``: the generated face crop was blended
            into the source frame.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_count = len(frames)
        provenance: List[str] = ["passthrough"] * output_frame_count
        blend_materials: Dict[int, Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]] = {}
        skin_masks: Dict[int, Optional[np.ndarray]] = {}
        previous_resized: Optional[np.ndarray] = None
        for output_index in _progress(
            range(output_frame_count),
            "render frames",
            total=output_frame_count,
            unit="frame",
        ):
            source_index = self._source_index_for_output(output_index, frame_count)
            original_frame = frames[source_index].copy()
            target = targets[source_index]
            bbox = target.get("bbox")
            result_frame = generated.get(output_index)

            if bbox is None or result_frame is None or face_parser is None:
                cv2.imwrite(str(output_dir / f"{output_index:08d}.png"), original_frame)
                continue

            crop_bbox = self._bbox_with_margin(bbox, original_frame, extra_margin)
            if crop_bbox is None:
                cv2.imwrite(str(output_dir / f"{output_index:08d}.png"), original_frame)
                continue

            x1, y1, x2, y2 = crop_bbox
            blend_status = "generated"
            try:
                if source_index not in blend_materials:
                    try:
                        mask_array, crop_box = get_image_prepare_material(
                            original_frame,
                            [x1, y1, x2, y2],
                            upper_boundary_ratio=blend_upper_boundary_ratio,
                            fp=face_parser,
                            mode=parsing_mode,
                            mask_blur_ratio=blend_mask_blur_ratio,
                        )
                        blend_materials[source_index] = (mask_array, tuple(crop_box))
                    except Exception:
                        blend_materials[source_index] = None
                material = blend_materials[source_index]
                if material is None:
                    cv2.imwrite(str(output_dir / f"{output_index:08d}.png"), original_frame)
                    continue
                resized = cv2.resize(
                    result_frame.astype(np.uint8),
                    (x2 - x1, y2 - y1),
                    interpolation=cv2.INTER_LANCZOS4,
                )
                reference_crop = original_frame[y1:y2, x1:x2]
                # Skin-only mask for detail restore. Limits the
                # reference's high-frequency detail to the cheek /
                # forehead / jaw area so it is NOT layered on the
                # generated open-mouth area (where it produces a
                # visible color block because the reference is
                # closed-mouth).
                skin_mask: Optional[np.ndarray] = None
                if face_parser is not None and source_index not in skin_masks:
                    try:
                        mask_array_b, crop_box_b = material
                        body = Image.fromarray(original_frame[:, :, ::-1])
                        face_large = body.crop(crop_box_b)
                        skin_parsing = face_parser(face_large, mode="skin_only")
                        skin_full = np.array(skin_parsing)
                        xs = max(0, x1 - crop_box_b[0])
                        ys = max(0, y1 - crop_box_b[1])
                        xe = xs + (x2 - x1)
                        ye = ys + (y2 - y1)
                        xe = min(xe, skin_full.shape[1])
                        ye = min(ye, skin_full.shape[0])
                        cropped = skin_full[ys:ye, xs:xe]
                        if cropped.shape == (y2 - y1, x2 - x1):
                            skin_masks[source_index] = cropped
                        else:
                            skin_masks[source_index] = None
                    except Exception:
                        skin_masks[source_index] = None
                skin_mask = skin_masks.get(source_index)
                resized = self._match_color_stats(resized, reference_crop, color_match_strength)
                # Pull the generated mouth's mean / std toward
                # the reference's skin tone. After this the
                # generated mouth is roughly the same global
                # brightness / temperature as the source skin,
                # and the per-tile drift left over (local
                # contrast imbalance) is what CLAHE cleans up
                # next.
                resized = self._match_mouth_to_skin_tone(
                    resized, reference_crop, skin_mask
                )
                resized = self._restore_reference_detail(
                    resized, reference_crop, mouth_detail_strength, skin_mask=skin_mask
                )
                # Local color-block fix on the mouth. Done after
                # detail restore so the per-tile drift in the
                # generated open-mouth area is the only thing the
                # equalizer sees -- skin and other regions are
                # left alone.
                resized = self._fix_mouth_color_block(resized, skin_mask)
                resized = self._sharpen_image(resized, mouth_sharpen_strength)
                # Output-level temporal blend. After all post-
                # processing, mix the current face crop with the
                # previous frame's face crop. Cures the per-frame
                # content jitter that bbox smoothing alone cannot
                # fix (bbox position is stable, but the generated
                # mouth shape / texture still shakes between
                # frames). 0 disables. Default 0 = current per-
                # frame behavior; per-request 0.2-0.3 is a light
                # smooth, 0.4-0.5 is heavy and may ghost on fast
                # motion.
                if (
                    output_temporal_blend > 0.0
                    and previous_resized is not None
                    and previous_resized.shape == resized.shape
                ):
                    resized = cv2.addWeighted(
                        resized,
                        1.0 - output_temporal_blend,
                        previous_resized,
                        output_temporal_blend,
                        0,
                    )
                previous_resized = resized
                # Light color-block check. Compares the upper-face
                # color distribution between the post-processed
                # crop and the reference. Tolerates normal per-
                # pixel variation (lip motion) and only fires on a
                # hard color block. Runs first because it is the
                # most semantically appropriate for color blocks;
                # the heavier MSE checks below are opt-in for
                # stricter behavior.
                if (
                    quality_max_face_color_histogram_distance > 0.0
                    and self._face_color_histogram_distance(
                        resized, reference_crop
                    )
                    > quality_max_face_color_histogram_distance
                ):
                    blend_status = "quality_fallback"
                    provenance[output_index] = blend_status
                    cv2.imwrite(str(output_dir / f"{output_index:08d}.png"), original_frame)
                    continue
                # Drift fallback. Compares the post-processed face
                # crop to the reference on the area OUTSIDE the
                # deep mouth ROI (upper face + lip border + jaw).
                # When the MSE exceeds the threshold, the
                # post-processing or the generation has drifted
                # from the source in a way that would produce a
                # visible color/seam around the lips -- fall back
                # to the source frame so the user sees the
                # unmodified original instead of a half-broken
                # edit.
                if (
                    quality_max_face_outside_mouth_mse > 0.0
                    and self._face_outside_mouth_mse(resized, reference_crop)
                    > quality_max_face_outside_mouth_mse
                ):
                    blend_status = "quality_fallback"
                    provenance[output_index] = blend_status
                    cv2.imwrite(str(output_dir / f"{output_index:08d}.png"), original_frame)
                    continue
                if (
                    quality_max_face_tile_mse > 0.0
                    and self._face_max_tile_mse(resized, reference_crop)
                    > quality_max_face_tile_mse
                ):
                    blend_status = "quality_fallback"
                    provenance[output_index] = blend_status
                    cv2.imwrite(str(output_dir / f"{output_index:08d}.png"), original_frame)
                    continue
                # Mouth-region postfilter. Catches a single large
                # blurry patch in the generated mouth (CodeFormer
                # failure, VAE collapse) that the whole-image
                # quality gate misses because the surrounding face
                # still looks fine.
                if (
                    quality_mouth_min_laplacian > 0.0
                    and self._mouth_region_laplacian(resized)
                    < quality_mouth_min_laplacian
                ):
                    blend_status = "quality_fallback"
                    provenance[output_index] = blend_status
                    cv2.imwrite(str(output_dir / f"{output_index:08d}.png"), original_frame)
                    continue
                if quality_gate_enabled and self._is_low_quality_generation(
                    resized,
                    reference_crop,
                    quality_min_laplacian,
                    quality_min_sharpness_ratio,
                ):
                    blend_status = "quality_fallback"
                    provenance[output_index] = blend_status
                    cv2.imwrite(str(output_dir / f"{output_index:08d}.png"), original_frame)
                    continue
                mask_array, crop_box = material
                combined = get_image_blending(
                    original_frame,
                    resized,
                    [x1, y1, x2, y2],
                    mask_array,
                    crop_box,
                )
            except Exception as exc:
                blend_status = "blend_error"
                combined = original_frame
                logger.warning(
                    "[LipSync] blend_error output_index=%d source_index=%d "
                    "bbox=(%d,%d,%d,%d) face_crop_shape=%s: %s",
                    output_index,
                    source_index,
                    x1, y1, x2, y2,
                    tuple(resized.shape) if 'resized' in locals() else None,
                    exc,
                    exc_info=True,
                )
            provenance[output_index] = blend_status
            cv2.imwrite(str(output_dir / f"{output_index:08d}.png"), combined)
        return provenance

    def _frames_to_video(self, frames_dir: Path, fps: float, temp_video_path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "warning",
                "-r",
                str(fps),
                "-f",
                "image2",
                "-i",
                str(frames_dir / "%08d.png"),
                "-vcodec",
                "libx264",
                "-vf",
                "format=yuv420p",
                "-crf",
                "15",
                str(temp_video_path),
            ],
            check=True,
        )

    def _combine_audio(
        self,
        audio_path: Path,
        temp_video_path: Path,
        output_path: Path,
        video_duration_seconds: float,
    ) -> None:
        duration = max(0.001, float(video_duration_seconds))
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "warning",
                "-i",
                str(temp_video_path),
                "-i",
                str(audio_path),
                "-map",
                "0:v:0",
                "-map",
                "1:a:0",
                "-c:v",
                "copy",
                "-af",
                f"apad,atrim=0:{duration:.6f}",
                "-ar",
                "44100",
                "-c:a",
                "aac",
                "-t",
                f"{duration:.6f}",
                str(output_path),
            ],
            check=True,
        )

    @staticmethod
    def _log_unsupported_fields(payload: LipSyncRequest) -> None:
        """Warn about LipSyncRequest fields that are accepted for
        cross-backend API compatibility but do not affect MuseTalk's
        encoder-decoder pipeline. We log at INFO rather than WARNING
        because the field defaults are common; a real warning is
        reserved for non-default opt-ins.
        """
        # Diffusion-only inference overrides. None means "use server
        # default", which is the only mode MuseTalk knows about.
        diffusion_overrides = (
            ("guidance_scale_override", payload.guidance_scale_override),
            ("inference_steps_override", payload.inference_steps_override),
            ("enable_deepcache_override", payload.enable_deepcache_override),
        )
        for name, value in diffusion_overrides:
            if value is not None:
                logger.info(
                    "[LipSync] %s=%s is a LatentSync diffusion-only field; "
                    "MuseTalk is encoder-decoder single-step, this hint is ignored.",
                    name, value,
                )
        if payload.mask_image_path is not None:
            logger.info(
                "[LipSync] mask_image_path=%s is a LatentSync diffusion-only field; "
                "MuseTalk uses the in-house blending mask and ignores this override.",
                payload.mask_image_path,
            )

        # Pre-filter thresholds. The defaults below are the
        # "filter disabled" values (matching LatentSync's recent
        # relaxations), so we only log when the caller explicitly
        # tightened them.
        prefilter_thresholds = (
            ("mouth_occlusion_skip_threshold", payload.mouth_occlusion_skip_threshold, 1.0),
            ("motion_blur_skip_threshold", payload.motion_blur_skip_threshold, 0.08),
            ("face_jump_center_threshold", payload.face_jump_center_threshold, 0.0),
            ("face_jump_scale_threshold", payload.face_jump_scale_threshold, 0.0),
        )
        for name, value, disabled_value in prefilter_thresholds:
            if value != disabled_value:
                logger.info(
                    "[LipSync] %s=%s is a LatentSync prefilter knob; "
                    "MuseTalk does not currently implement %s-based skipping, "
                    "the value is accepted but ignored.",
                    name, value, name.split("_")[0],
                )
        if (
            payload.yaw_skip_threshold != 45.0
            or payload.yaw_rate_skip_threshold != 28.0
            or payload.side_face_episode_pre_pad != 0
            or payload.side_face_episode_post_pad != 0
            or payload.yaw_warn_threshold_ratio != 0.75
            or payload.side_face_warn_min_run_frames != 0
        ):
            logger.info(
                "[LipSync] yaw-based side-face filters (yaw_skip=%s, yaw_rate=%s, "
                "pre_pad=%s, post_pad=%s, warn_ratio=%s, warn_min_run=%s) are "
                "LatentSync diffusion-only knobs; MuseTalk accepts them but does not skip frames on yaw.",
                payload.yaw_skip_threshold,
                payload.yaw_rate_skip_threshold,
                payload.side_face_episode_pre_pad,
                payload.side_face_episode_post_pad,
                payload.yaw_warn_threshold_ratio,
                payload.side_face_warn_min_run_frames,
            )

        if payload.codeformer_enabled and settings.codeformer_required:
            restorer, load_error = self._get_codeformer_restorer()
            if restorer is None or not restorer.is_loaded:
                raise HTTPException(
                    status_code=503,
                    detail=f"CodeFormer is required but not available: {load_error or 'model not loaded'}",
                )

    @staticmethod
    def _apply_seed_override(payload: LipSyncRequest) -> None:
        """Apply ``seed_override`` to torch / numpy RNG. The MuseTalk
        inference path is mostly deterministic (UNet forward + VAE
        decode given the same input), so this is best-effort: it
        covers torch.manual_seed and numpy.random for any code that
        consults them on the request thread. ``-1`` means random.
        """
        seed = payload.seed_override
        if seed is None:
            return
        if int(seed) == -1:
            torch.seed()
            return
        try:
            torch.manual_seed(int(seed))
            np.random.seed(int(seed))
        except Exception as exc:
            logger.warning("[LipSync] seed_override=%s could not be applied: %s", seed, exc)

    @torch.no_grad()
    def synthesize(self, payload: LipSyncRequest, paths: Dict[str, Path], job_output_dir: Path) -> Dict[str, object]:
        self.load_detectors()
        with self.run_lock:
            self._log_unsupported_fields(payload)
            self._apply_seed_override(payload)
            frames, fps = _read_video_frames(paths["video"])
            avatar_descriptor = self._avatar_descriptor(paths["avatar"]) if paths.get("avatar") else None
            if (
                avatar_descriptor is not None
                and payload.require_face_embedding
                and "embedding" not in avatar_descriptor
            ):
                detail = "Face embedding is required for /api/lipsync, but the avatar image did not produce one."
                if self.face_embedding_error:
                    detail = f"{detail} InsightFace error: {self.face_embedding_error}"
                else:
                    detail = f"{detail} InsightFace loaded, but did not detect a face in the avatar image."
                raise RuntimeError(detail)
            target_identity = self._find_target_identity(frames, fps, avatar_descriptor, payload)
            if target_identity and target_identity.get("no_face_detected"):
                no_face_response = {
                    "passthrough": True,
                    "passthrough_reason": "no_face_detected",
                    "source_frame_count": len(frames),
                    "output_frame_count": len(frames),
                    "audio_frame_count": 0,
                    "source_fps": round(float(fps), 6),
                    "audio_feature_fps": 0.0,
                    "audio_sync_offset_frames": 0,
                    "audio_sync_offset_output_frames": 0,
                    "audio_sync_offset_seconds": payload.audio_sync_offset_seconds,
                    "matched_source_frames": 0,
                    "filled_source_frames": 0,
                    "filtered_motion_frames": 0,
                    "filtered_fast_motion_frames": 0,
                    "filtered_mouth_diff_frames": 0,
                    "continuity_filled_source_frames": 0,
                    "filtered_small_face_frames": 0,
                    "filtered_short_segment_frames": 0,
                    "smoothed_source_frames": 0,
                    "matched_or_filled_source_frames": 0,
                    "eligible_source_frames": 0,
                    "generated_output_frames": 0,
                    "quality_fallback_frames": 0,
                    "effective_generated_output_frames": 0,
                    "skipped_output_frames": len(frames),
                    "frame_provenance": ["passthrough"] * len(frames),
                    "best_similarity": 0.0,
                    "target_identity_similarity": 0.0,
                    "target_identity_count": 0,
                    "target_identity_coverage": 0.0,
                    "target_identity_source": "none",
                    "face_identity_backend": "embedding" if payload.require_face_embedding else "visual",
                    # LatentSync-style stats (always zero on the no-face passthrough).
                    "pre_skip_frames": 0,
                    "quality_skip_frames": 0,
                    "yaw_skip_count": 0,
                    "yaw_rate_skip_count": 0,
                    "mouth_occlusion_skip_count": 0,
                    "motion_blur_skip_count": 0,
                    "face_jump_skip_count": 0,
                    "side_face_episode_extra_skip_count": 0,
                    "side_face_warn_run_skip_count": 0,
                    "silent_skip_frames": 0,
                    "skipped_inference_batches": 0,
                    "skipped_inference_frames": 0,
                    "effective_guidance_scale": payload.guidance_scale_override,
                    "effective_inference_steps": payload.inference_steps_override,
                    "effective_seed": payload.seed_override,
                    "identity_skip_count": 0,
                    "identity_similarity_min": 0.0,
                    "identity_similarity_median": 0.0,
                    "identity_similarity_max": 0.0,
                    "identity_similarity_threshold": float(payload.similarity_threshold),
                    "speech_gate": {
                        "enabled": payload.speech_gate_enabled,
                        "active_frames": 0,
                        "silent_frames": 0,
                    },
                    "mouth_temporal": {
                        "unsupported": True,
                        "stabilization_strength": float(payload.mouth_temporal_stabilization_strength),
                        "stabilization_max_delta": float(payload.mouth_temporal_stabilization_max_delta),
                        "delta_min": 0.0,
                        "delta_median": 0.0,
                        "delta_max": 0.0,
                        "delta_skip_frames": 0,
                        "stabilized_frames": 0,
                    },
                    "codeformer": {
                        "requested": bool(payload.codeformer_enabled),
                        "fidelity_weight": float(payload.codeformer_fidelity_weight),
                        "adain": bool(payload.codeformer_adain),
                        "required": bool(payload.codeformer_required or settings.codeformer_required),
                        "runtime_available": False,
                        "runtime_load_error": "",
                        "checkpoint_path": settings.codeformer_checkpoint_path,
                        "frames_total": 0,
                        "frames_enhanced": 0,
                        "frames_fallback": 0,
                        "frames_skipped_by_pipeline": 0,
                        "elapsed_seconds": 0.0,
                        "error": "no target face detected -- CodeFormer skipped",
                    },
                    "quality_ok": True,
                }
                _log_lipsync_report(job_id, no_face_response)
                return no_face_response
            target_descriptors = target_identity.get("target_descriptors") if target_identity else []
            negative_descriptors = target_identity.get("negative_descriptors") if target_identity else []
            target_identity_score = (
                float(target_identity["avatar_score"])
                if target_identity and avatar_descriptor is not None
                else 0.0
            )
            target_identity_count = int(target_identity["count"]) if target_identity else 0
            target_identity_coverage = float(target_identity.get("identity_coverage", 0.0)) if target_identity else 0.0
            face_identity_backend = "embedding" if payload.require_face_embedding else "visual"
            target_identity_source = (
                _describe_target_identity_source(avatar_descriptor, target_identity)
            )

            targets = []
            matched_source_frames = 0
            prefiltered_blur_frames = 0
            prefiltered_side_face_frames = 0
            best_scores = []
            previous_bbox = None
            largest_face_mode = bool(
                target_identity and target_identity.get("selection_source") == "most_open_mouth_per_frame"
            )
            for frame in _progress(frames, "match target", total=len(frames), unit="frame"):
                if target_descriptors:
                    bbox, score = self._select_target_bbox(
                        frame,
                        avatar_descriptor,
                        payload.similarity_threshold,
                        payload.bbox_shift,
                        payload.identity_margin,
                        min_detection_score=payload.min_detection_score,
                        require_landmark_match=payload.require_landmark_match,
                        min_landmark_points=payload.min_landmark_points,
                        min_landmark_overlap=payload.min_landmark_overlap,
                        expected_descriptors=target_descriptors,
                        negative_descriptors=negative_descriptors,
                        require_embedding=payload.require_face_embedding,
                        allow_crop_embedding_fallback=payload.allow_crop_embedding_fallback,
                        crop_embedding_min_detection_score=payload.crop_embedding_min_detection_score,
                        previous_bbox=previous_bbox,
                        temporal_tracking_weight=payload.temporal_tracking_weight,
                    )
                elif largest_face_mode:
                    bbox, score = self._select_most_open_mouth_bbox(
                        frame,
                        payload.bbox_shift,
                        payload.min_detection_score,
                        payload.require_landmark_match,
                        payload.min_landmark_points,
                        payload.min_landmark_overlap,
                        previous_bbox=previous_bbox,
                        track_min_iou=payload.target_track_min_iou,
                        track_max_center_shift_ratio=payload.target_track_max_center_shift_ratio,
                    )
                else:
                    bbox, score = None, 0.0
                # Motion-blur prefilter on the source face. When the
                # detected face crop is too blurry the lipsync output
                # would be even worse (CodeFormer cannot recover what
                # the source never had), so we drop the target and
                # let the frame passthrough.
                if (
                    bbox is not None
                    and payload.prefilter_min_face_laplacian > 0.0
                ):
                    fx1, fy1, fx2, fy2 = bbox
                    face_crop = frame[
                        max(0, fy1):min(frame.shape[0], fy2),
                        max(0, fx1):min(frame.shape[1], fx2),
                    ]
                    if (
                        self._laplacian_variance(face_crop)
                        < payload.prefilter_min_face_laplacian
                    ):
                        bbox = None
                        score = 0.0
                        prefiltered_blur_frames += 1
                # Side-face prefilter via bbox aspect ratio. A face
                # turned to profile has a bbox significantly wider
                # than tall (w/h > 1.3). MuseTalk is trained mostly
                # on front/3-quarter views and produces poor output
                # (color blocks, wrong mouth shape) for near-profile
                # faces. We skip lipsync for these frames and let
                # the original pass through. 0 disables.
                if (
                    bbox is not None
                    and payload.prefilter_side_face_aspect_ratio > 0.0
                ):
                    fx1, fy1, fx2, fy2 = bbox
                    fw = max(1, fx2 - fx1)
                    fh = max(1, fy2 - fy1)
                    aspect = float(fw) / float(fh)
                    if aspect > payload.prefilter_side_face_aspect_ratio:
                        bbox = None
                        score = 0.0
                        prefiltered_side_face_frames += 1
                targets.append({"bbox": bbox, "score": score})
                best_scores.append(score)
                if bbox is not None:
                    matched_source_frames += 1
                    previous_bbox = bbox

            filled_source_frames = self._fill_short_target_gaps(
                targets,
                fps,
                frames[0].shape if frames else (0, 0, 3),
                payload.target_fill_max_gap_seconds,
                payload.target_fill_window_seconds,
                payload.target_fill_min_match_ratio,
                payload.target_fill_max_center_shift,
            )
            filtered_motion_frames = self._filter_motion_targets(
                targets,
                frames[0].shape if frames else (0, 0, 3),
                payload.target_motion_gate_enabled,
                payload.target_motion_max_center_shift,
                payload.target_motion_max_scale_change,
            )
            filtered_fast_motion_frames = self._filter_fast_motion_targets(
                targets,
                payload.target_fast_motion_gate_enabled,
                payload.target_fast_motion_max_center_shift_per_frame,
                payload.target_fast_motion_max_scale_change_per_frame,
                payload.target_fast_motion_min_run_frames,
            )
            filtered_mouth_diff_frames = self._filter_mouth_diff_targets(
                targets,
                frames,
                payload.lipsync_mouth_diff_break_threshold,
            )
            continuity_filled_source_frames = self._fill_short_target_gaps(
                targets,
                fps,
                frames[0].shape if frames else (0, 0, 3),
                payload.lipsync_continuity_max_gap_seconds,
                payload.lipsync_continuity_window_seconds,
                payload.lipsync_continuity_min_match_ratio,
                payload.lipsync_continuity_max_center_shift,
            )
            filtered_small_face_frames, filtered_short_segment_frames = self._filter_lipsync_targets(
                targets,
                frames[0].shape if frames else (0, 0, 3),
                payload.lipsync_min_segment_frames,
                payload.lipsync_min_face_area_ratio,
            )
            smoothed_source_frames = self._smooth_target_bboxes(
                targets,
                frames[0].shape if frames else (0, 0, 3),
                payload.target_bbox_smoothing_window,
                payload.target_bbox_smoothing_max_center_shift,
            )
            eligible_source_frames = sum(1 for target in targets if target.get("bbox") is not None)

            self.load()
            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(str(paths["audio"]))
            if whisper_input_features is None:
                raise RuntimeError(f"Could not read audio: {paths['audio']}")
            audio_feature_fps = self._resolve_audio_feature_fps(fps, payload)
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.device,
                self.weight_dtype,
                self.whisper,
                librosa_length,
                fps=audio_feature_fps,
                audio_padding_length_left=payload.audio_padding_length_left,
                audio_padding_length_right=payload.audio_padding_length_right,
            )
            audio_frame_count = len(whisper_chunks)
            if audio_frame_count == 0:
                raise RuntimeError("Audio is too short to produce video frames.")
            output_frame_count = len(frames)
            audio_sync_offset_frames = int(round(payload.audio_sync_offset_seconds * audio_feature_fps))
            speech_sync_offset_frames = int(round(payload.audio_sync_offset_seconds * fps))
            speech_activity_mask, speech_gate_stats = self._audio_activity_mask(
                paths["audio"],
                fps,
                output_frame_count,
                payload.speech_gate_enabled,
                payload.speech_gate_relative_db,
                payload.speech_gate_min_rms,
                payload.speech_gate_window_seconds,
                payload.speech_gate_pre_roll_seconds,
                payload.speech_gate_post_roll_seconds,
                payload.speech_gate_fill_gap_seconds,
            )
            # Count silent frames for the LatentSync-style
            # ``silent_skip_frames`` stat. When the speech gate is
            # disabled every frame is "active" by definition, so
            # silent_skip_frames is 0 and the count matches the
            # pipeline's observed behavior.
            if speech_activity_mask:
                silent_skip_frames = sum(1 for active in speech_activity_mask if not active)
            else:
                silent_skip_frames = 0
            if speech_gate_stats is not None and isinstance(speech_gate_stats, dict):
                speech_gate_stats.setdefault("silent_frames", silent_skip_frames)
            # Summary stats over the per-frame best similarity score.
            # LatentSync publishes min/median/max identity-similarity;
            # MuseTalk stores the per-frame "score" the same way, so
            # expose the same summary so cross-backend dashboards can
            # compare apples to apples.
            nonzero_scores = [score for score in best_scores if score]
            identity_similarity_min = float(min(nonzero_scores)) if nonzero_scores else 0.0
            identity_similarity_median = (
                float(np.median(nonzero_scores)) if nonzero_scores else 0.0
            )
            identity_similarity_max = float(max(nonzero_scores)) if nonzero_scores else 0.0

            latents_by_frame = self._encode_latents(frames, targets, payload.extra_margin)
            process_items = []
            for output_index in range(output_frame_count):
                audio_index = self._audio_feature_index_for_output(
                    output_index,
                    fps,
                    audio_feature_fps,
                    audio_frame_count,
                    audio_sync_offset_frames,
                )
                speech_index = min(max(output_index + speech_sync_offset_frames, 0), len(speech_activity_mask) - 1)
                if speech_activity_mask and not speech_activity_mask[speech_index]:
                    continue
                source_index = self._source_index_for_output(output_index, len(frames))
                latent = latents_by_frame.get(source_index)
                if latent is None:
                    continue
                process_items.append((output_index, whisper_chunks[audio_index], latent))

            generated = self._run_inference_batches(process_items, payload.batch_size) if process_items else {}

            # --- CodeFormer face restoration ---
            codeformer_stats = {
                "requested": bool(payload.codeformer_enabled),
                "fidelity_weight": float(payload.codeformer_fidelity_weight),
                "adain": bool(payload.codeformer_adain),
                "required": bool(payload.codeformer_required or settings.codeformer_required),
                "runtime_available": False,
                "runtime_load_error": "",
                "checkpoint_path": settings.codeformer_checkpoint_path,
                "frames_total": 0,
                "frames_enhanced": 0,
                "frames_fallback": 0,
                "frames_skipped_by_pipeline": 0,
                "elapsed_seconds": 0.0,
                "error": "",
            }
            if payload.codeformer_enabled and generated:
                restorer, load_error = self._get_codeformer_restorer()
                if restorer is not None and restorer.is_loaded:
                    codeformer_stats["runtime_available"] = True
                    # Build a (T, 3, H, W) tensor in [-1, 1] from generated BGR faces.
                    # CodeFormer expects 512x512 input; MuseTalk produces 256x256
                    # faces so we upsample before the model and downsample after.
                    CODEFORMER_INPUT_SIZE = 512
                    sorted_indices = sorted(generated.keys())
                    original_sizes = []
                    face_tensors = []
                    for idx in sorted_indices:
                        face_bgr = generated[idx]  # (H, W, 3) BGR uint8
                        original_sizes.append((face_bgr.shape[0], face_bgr.shape[1]))
                        face_bgr_512 = cv2.resize(face_bgr, (CODEFORMER_INPUT_SIZE, CODEFORMER_INPUT_SIZE), interpolation=cv2.INTER_LANCZOS4)
                        face_rgb = face_bgr_512[..., ::-1]  # RGB
                        face_float = face_rgb.astype(np.float32) / 255.0 * 2.0 - 1.0  # [-1, 1]
                        face_chw = np.transpose(face_float, (2, 0, 1))  # (3, H, W)
                        face_tensors.append(face_chw)
                    face_batch = torch.from_numpy(np.stack(face_tensors, axis=0)).to(self.device)
                    restored_batch, cf_stats = restorer.restore_faces(
                        face_batch,
                        fidelity_weight=payload.codeformer_fidelity_weight,
                        adain=payload.codeformer_adain,
                    )
                    # Cross-frame EMA on the CodeFormer-restored
                    # crops. CodeFormer is a per-frame model with
                    # no temporal constraint, so on a real
                    # lipsync input the restored output flickers
                    # frame-to-frame. Mixing each restored crop
                    # with the previous one in sorted-index order
                    # damps the variation without lagging the
                    # lipsync motion too much (alpha 0.7 keeps
                    # 70% of the current frame). Sorted-index
                    # order is the natural play order of the
                    # video; the EMA chain walks through it.
                    restored_np = restored_batch.cpu().numpy()
                    cf_alpha = float(payload.codeformer_temporal_alpha)
                    prev_restored: Optional[np.ndarray] = None
                    for i, idx in enumerate(sorted_indices):
                        face_out = restored_np[i]  # (3, 512, 512) in [-1, 1]
                        face_out = np.transpose(face_out, (1, 2, 0))  # (512, 512, 3) RGB
                        face_out = ((face_out + 1.0) / 2.0 * 255.0)
                        face_out = np.clip(face_out, 0, 255).astype(np.uint8)
                        if (
                            cf_alpha < 1.0
                            and prev_restored is not None
                            and prev_restored.shape == face_out.shape
                        ):
                            face_out = cv2.addWeighted(
                                face_out,
                                cf_alpha,
                                prev_restored,
                                1.0 - cf_alpha,
                                0,
                            )
                        prev_restored = face_out
                        face_out = face_out[..., ::-1]  # RGB -> BGR
                        orig_h, orig_w = original_sizes[i]
                        if face_out.shape[0] != orig_h or face_out.shape[1] != orig_w:
                            face_out = cv2.resize(face_out, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                        generated[idx] = face_out
                    codeformer_stats.update(cf_stats.as_dict())
                    codeformer_stats["runtime_available"] = True
                    codeformer_stats["runtime_load_error"] = ""
                else:
                    codeformer_stats["runtime_available"] = False
                    codeformer_stats["runtime_load_error"] = load_error or "CodeFormer model not loaded"
                    codeformer_stats["error"] = load_error or "CodeFormer model not loaded"
                    logger.warning(
                        "[LipSync] codeformer_enabled=True but CodeFormer is not available: %s",
                        load_error or "model not loaded",
                    )

            face_parser = (
                self._get_face_parser(payload.left_cheek_width, payload.right_cheek_width)
                if generated
                else None
            )
            render_dir = job_output_dir / "frames"
            frame_provenance = self._write_result_frames(
                frames,
                targets,
                generated,
                output_frame_count,
                render_dir,
                payload.extra_margin,
                payload.parsing_mode,
                payload.blend_upper_boundary_ratio,
                payload.blend_mask_blur_ratio,
                payload.color_match_strength,
                payload.mouth_detail_strength,
                payload.mouth_sharpen_strength,
                payload.output_temporal_blend,
                payload.quality_gate_enabled,
                payload.quality_min_laplacian,
                payload.quality_min_sharpness_ratio,
                payload.quality_mouth_min_laplacian,
                payload.quality_max_face_outside_mouth_mse,
                payload.quality_max_face_tile_mse,
                payload.quality_max_face_color_histogram_distance,
                face_parser,
            )
            quality_fallback_frames = sum(
                1 for status in frame_provenance if status == "quality_fallback"
            )

            temp_video_path = job_output_dir / "temp_video.mp4"
            output_path = job_output_dir / "result.mp4"
            self._frames_to_video(render_dir, fps, temp_video_path)
            self._combine_audio(paths["audio"], temp_video_path, output_path, output_frame_count / fps)

            shutil.rmtree(render_dir, ignore_errors=True)
            temp_video_path.unlink(missing_ok=True)

            skipped_output_frames = output_frame_count - len(generated)
            return {
                "output_path": output_path,
                "source_frame_count": len(frames),
                "output_frame_count": output_frame_count,
                "audio_frame_count": audio_frame_count,
                "source_fps": round(float(fps), 6),
                "audio_feature_fps": round(float(audio_feature_fps), 6),
                "audio_sync_offset_frames": audio_sync_offset_frames,
                "audio_sync_offset_output_frames": speech_sync_offset_frames,
                "audio_sync_offset_seconds": payload.audio_sync_offset_seconds,
                "matched_source_frames": matched_source_frames,
                "filled_source_frames": filled_source_frames,
                "filtered_motion_frames": filtered_motion_frames,
                "filtered_fast_motion_frames": filtered_fast_motion_frames,
                "filtered_mouth_diff_frames": filtered_mouth_diff_frames,
                "continuity_filled_source_frames": continuity_filled_source_frames,
                "filtered_small_face_frames": filtered_small_face_frames,
                "filtered_short_segment_frames": filtered_short_segment_frames,
                "prefiltered_blur_frames": prefiltered_blur_frames,
                "prefiltered_side_face_frames": prefiltered_side_face_frames,
                "smoothed_source_frames": smoothed_source_frames,
                "matched_or_filled_source_frames": (
                    matched_source_frames + filled_source_frames + continuity_filled_source_frames
                ),
                "eligible_source_frames": eligible_source_frames,
                "generated_output_frames": len(generated),
                "quality_fallback_frames": quality_fallback_frames,
                "effective_generated_output_frames": max(0, len(generated) - quality_fallback_frames),
                "skipped_output_frames": skipped_output_frames,
                # Per-frame provenance: "passthrough" / "generated" /
                # "quality_fallback" / "blend_error". Length always
                # equals output_frame_count so callers can correlate
                # it with frame indices and audio alignment.
                "frame_provenance": frame_provenance,
                "best_similarity": max(best_scores) if best_scores else 0.0,
                "target_identity_similarity": target_identity_score,
                "target_identity_count": target_identity_count,
                "target_identity_coverage": target_identity_coverage,
                "target_identity_source": target_identity_source,
                "face_identity_backend": face_identity_backend,
                "speech_gate": speech_gate_stats,
                # --- LatentSync-style per-frame filter / pre-skip stats ---
                # MuseTalk's encoder-decoder pipeline does not perform
                # yaw / occlusion / motion-blur / face-jump / side-face
                # episode filtering -- those knobs are diffusion-only and
                # are accepted on the wire but ignored at runtime (see
                # ``_log_unsupported_fields``). We report zeros so
                # cross-backend dashboards have stable columns.
                "pre_skip_frames": 0,
                "quality_skip_frames": 0,
                "yaw_skip_count": 0,
                "yaw_rate_skip_count": 0,
                "mouth_occlusion_skip_count": 0,
                "motion_blur_skip_count": 0,
                "face_jump_skip_count": 0,
                "side_face_episode_extra_skip_count": 0,
                "side_face_warn_run_skip_count": 0,
                "silent_skip_frames": silent_skip_frames,
                "skipped_inference_batches": 0,
                "skipped_inference_frames": 0,
                # --- LatentSync-style inference overrides (no-op in MuseTalk) ---
                "effective_guidance_scale": payload.guidance_scale_override,
                "effective_inference_steps": payload.inference_steps_override,
                "effective_seed": payload.seed_override,
                # --- Per-frame identity similarity summary (matches the
                #     LatentSync response schema; computed from
                #     ``best_scores`` collected during target selection). ---
                "identity_skip_count": 0,
                "identity_similarity_min": identity_similarity_min,
                "identity_similarity_median": identity_similarity_median,
                "identity_similarity_max": identity_similarity_max,
                "identity_similarity_threshold": float(payload.similarity_threshold),
                # --- Mouth temporal stabilization (diffusion-only stat) ---
                # MuseTalk is a single-step encoder-decoder; there is no
                # per-frame delta to stabilize. We echo the request
                # settings back so the dashboard's "current run config"
                # panel still shows the caller's intent, and mark the
                # block unsupported so consumers can branch on it.
                "mouth_temporal": {
                    "unsupported": True,
                    "stabilization_strength": float(payload.mouth_temporal_stabilization_strength),
                    "stabilization_max_delta": float(payload.mouth_temporal_stabilization_max_delta),
                    "delta_min": 0.0,
                    "delta_median": 0.0,
                    "delta_max": 0.0,
                    "delta_skip_frames": 0,
                    "stabilized_frames": 0,
                },
                "codeformer": codeformer_stats,
                "quality_ok": True,
            }


runtime = MuseTalkApiRuntime()


def _output_url(request: Request, output_path: Path) -> str:
    relative = output_path.relative_to(OUTPUT_ROOT).as_posix()
    # Force https and take host:port from the actual request. When the
    # service sits behind a TLS-terminating reverse proxy (e.g.
    # seetacloud), ``request.url.netloc`` reflects whatever the proxy
    # forwarded -- if the proxy is on the standard 443, netloc has no
    # port; if a non-default port is forwarded it is preserved.
    return f"https://{request.url.netloc}/outputs/{relative}"


def _log_lipsync_report(job_id: str, result: Dict[str, object]) -> None:
    """Emit a multi-line INFO summary of the ``/api/lipsync``
    response. Designed to be the canonical one-stop log line for
    diagnosing filter strictness: every per-filter count and the
    provenance breakdown are included so a single log entry is
    enough to tell which gate is over-firing.

    ``frame_provenance`` is summarized into per-status counts
    rather than printed in full (the list can be hundreds of
    entries on a long video).
    """
    provenance = result.get("frame_provenance") or []
    prov_counts = {
        "passthrough": 0,
        "generated": 0,
        "quality_fallback": 0,
        "blend_error": 0,
    }
    for status in provenance:
        prov_counts[status] = prov_counts.get(status, 0) + 1
    speech_gate = result.get("speech_gate") or {}
    codeformer = result.get("codeformer") or {}
    gen = int(result.get("generated_output_frames", 0))
    qf = int(result.get("quality_fallback_frames", 0))
    be = prov_counts["blend_error"]
    effective_blended = max(0, gen - qf - be)
    logger.info(
        "[LipSync-report] job_id=%s\n"
        "  source: src_frames=%d out_frames=%d src_fps=%.3f audio_fps=%.3f\n"
        "  identity: source=%s count=%d coverage=%.3f sim=%.3f backend=%s\n"
        "  target: matched=%d filled_init=%d continuity_filled=%d smoothed=%d eligible=%d prefiltered_blur=%d prefiltered_side=%d\n"
        "  filtered: motion=%d fast_motion=%d mouth_diff=%d small_face=%d short_segment=%d\n"
        "  generated: gen=%d quality_fallback=%d passthrough=%d blend_error=%d blended=%d\n"
        "  identity_sim: min/med/max=%.3f/%.3f/%.3f\n"
        "  speech_gate: enabled=%s active=%s\n"
        "  codeformer: enabled=%s available=%s",
        job_id,
        int(result.get("source_frame_count", 0)),
        int(result.get("output_frame_count", 0)),
        float(result.get("source_fps", 0.0)),
        float(result.get("audio_feature_fps", 0.0)),
        result.get("target_identity_source", "none"),
        int(result.get("target_identity_count", 0)),
        float(result.get("target_identity_coverage", 0.0)),
        float(result.get("target_identity_similarity", 0.0)),
        result.get("face_identity_backend", "unknown"),
        int(result.get("matched_source_frames", 0)),
        int(result.get("filled_source_frames", 0)),
        int(result.get("continuity_filled_source_frames", 0)),
        int(result.get("smoothed_source_frames", 0)),
        int(result.get("eligible_source_frames", 0)),
        int(result.get("prefiltered_blur_frames", 0)),
        int(result.get("prefiltered_side_face_frames", 0)),
        int(result.get("filtered_motion_frames", 0)),
        int(result.get("filtered_fast_motion_frames", 0)),
        int(result.get("filtered_mouth_diff_frames", 0)),
        int(result.get("filtered_small_face_frames", 0)),
        int(result.get("filtered_short_segment_frames", 0)),
        int(result.get("generated_output_frames", 0)),
        int(result.get("quality_fallback_frames", 0)),
        prov_counts["passthrough"],
        prov_counts["blend_error"],
        effective_blended,
        float(result.get("identity_similarity_min", 0.0)),
        float(result.get("identity_similarity_median", 0.0)),
        float(result.get("identity_similarity_max", 0.0)),
        bool(speech_gate.get("enabled", False)),
        int(speech_gate.get("active_frames", 0)),
        bool(codeformer.get("requested", False)),
        bool(codeformer.get("runtime_available", False)),
    )


def _local_output_from_url(url: str) -> Optional[Path]:
    parsed = urlparse(url)
    path = parsed.path
    if path == "/api/download":
        query_url = parse_qs(parsed.query).get("url", [""])[0]
        if query_url:
            return _local_output_from_url(query_url)

    if not path.startswith("/outputs/"):
        return None
    relative = unquote(path[len("/outputs/"):]).lstrip("/")
    candidate = (OUTPUT_ROOT / relative).resolve()
    try:
        candidate.relative_to(OUTPUT_ROOT.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


@app.get("/health")
def health() -> Dict[str, object]:
    return {
        "status": "ok",
        "detectors_loaded": runtime.detectors_loaded,
        "model_loaded": runtime.loaded,
        "face_embedding_loaded": runtime.face_embedder is not None,
        "crop_face_embedding_loaded": runtime.face_recognition_session is not None,
        "face_embedding_backend": settings.face_embedding_backend,
        "face_embedding_error": runtime.face_embedding_error,
        "port": settings.port,
        "codeformer": {
            "checkpoint_path": settings.codeformer_checkpoint_path,
            "loaded": runtime.codeformer_restorer is not None and runtime.codeformer_restorer.is_loaded,
            "preload_requested": settings.codeformer_preload,
            "load_error": runtime.codeformer_load_error or "",
        },
    }


@app.post("/api/faces")
def list_distinct_faces(payload: FaceListRequest, request: Request) -> Dict[str, object]:
    job_id = uuid.uuid4().hex
    job_input_dir = INPUT_ROOT / job_id
    job_output_dir = OUTPUT_ROOT / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    video_path = _download_to_file(payload.video_url, job_input_dir, "video", VIDEO_SUFFIXES, ".mp4")

    try:
        result = runtime.extract_distinct_faces(video_path, job_output_dir, payload)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    face_paths = result.pop("face_paths")
    face_urls = [_output_url(request, face_path) for face_path in face_paths]
    faces = []
    for face_url, item in zip(face_urls, result.pop("faces")):
        faces.append(
            {
                "url": face_url,
                "max_area": item["max_area"],
                "frame_index": item["frame_index"],
                "detection_score": item["detection_score"],
                "count": item["count"],
            }
        )

    return {
        "job_id": job_id,
        "face_urls": face_urls,
        "faces": faces,
        **result,
    }


@app.post("/api/lipsync")
def create_lipsync(payload: LipSyncRequest, request: Request) -> Dict[str, object]:
    if payload.parsing_mode not in {"jaw", "raw"}:
        raise HTTPException(status_code=400, detail="parsing_mode must be 'jaw' or 'raw'")

    job_id = uuid.uuid4().hex
    job_input_dir = INPUT_ROOT / job_id
    job_output_dir = OUTPUT_ROOT / job_id
    job_input_dir.mkdir(parents=True, exist_ok=True)
    job_output_dir.mkdir(parents=True, exist_ok=True)

    video_path = _download_to_file(payload.video_url, job_input_dir, "video", VIDEO_SUFFIXES, ".mp4")
    audio_path = _download_to_file(payload.audio_url, job_input_dir, "audio", AUDIO_SUFFIXES, ".wav")
    # Distinguish "client did not send the field" (None -> auto-detect
    # reference face from the source video) from "client sent an empty
    # string" (almost certainly a bug -> fail loudly with 400). The
    # old `if payload.avatar_url` truthiness check conflated the two
    # and silently fell back when a client sent avatar_url: "".
    if payload.avatar_url is not None and not payload.avatar_url.strip():
        raise HTTPException(
            status_code=400,
            detail="avatar_url must not be empty; omit the field to use auto-detected reference face.",
        )
    avatar_path = (
        _download_to_file(payload.avatar_url, job_input_dir, "avatar", IMAGE_SUFFIXES, ".jpg")
        if payload.avatar_url is not None
        else None
    )

    try:
        input_paths = {"video": video_path, "audio": audio_path}
        if avatar_path is not None:
            input_paths["avatar"] = avatar_path
        result = runtime.synthesize(
            payload,
            input_paths,
            job_output_dir,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    if result.get("passthrough"):
        video_url = payload.video_url
    else:
        output_path = result.pop("output_path")
        video_url = _output_url(request, output_path)
    download_url = f"https://{request.url.netloc}/api/download?url={quote(video_url, safe='')}"
    response_body = {
        "job_id": job_id,
        "video_url": video_url,
        "download_url": download_url,
        **result,
    }
    logger.info(
        "[LipSync] job_id=%s video_url=%s download_url=%s passthrough=%s",
        job_id,
        video_url,
        download_url,
        bool(result.get("passthrough")),
    )
    _log_lipsync_report(job_id, response_body)
    return response_body


@app.get("/api/download")
def download_by_url(url: str = Query(..., description="Generated or remote video URL")):
    local_path = _local_output_from_url(url)
    if local_path is not None:
        return FileResponse(
            str(local_path),
            filename=local_path.name,
            media_type="video/mp4",
        )

    _validate_url(url)
    try:
        response = _get_download_response(url, "URL")
    except HTTPException:
        raise

    filename = Path(unquote(urlparse(url).path)).name or "download.mp4"
    media_type = response.headers.get("content-type", "application/octet-stream")

    def iterator():
        try:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    yield chunk
        finally:
            response.close()

    return StreamingResponse(
        iterator(),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MuseTalk HTTP API")
    parser.add_argument("--host", default=settings.host)
    parser.add_argument("--port", type=int, default=settings.port)
    parser.add_argument("--ffmpeg_path", default=settings.ffmpeg_path)
    parser.add_argument("--gpu_id", type=int, default=settings.gpu_id)
    parser.add_argument("--use_float16", action="store_true", default=settings.use_float16)
    parser.add_argument("--unet_model_path", default=settings.unet_model_path)
    parser.add_argument("--unet_config", default=settings.unet_config)
    parser.add_argument("--whisper_dir", default=settings.whisper_dir)
    parser.add_argument("--vae_type", default=settings.vae_type)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    settings.host = args.host
    settings.port = args.port
    settings.ffmpeg_path = args.ffmpeg_path
    settings.gpu_id = args.gpu_id
    settings.use_float16 = args.use_float16
    settings.unet_config, settings.unet_model_path = _resolve_musetalk_model_paths(
        args.unet_config,
        args.unet_model_path,
    )
    settings.whisper_dir = _resolve_model_dir(
        args.whisper_dir,
        ["models/whisper"],
        ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
    )
    settings.vae_type = _resolve_vae_type(args.vae_type)

    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
