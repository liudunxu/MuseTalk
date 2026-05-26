import argparse
import copy
import logging
import math
import mimetypes
import os
import shutil
import subprocess
import threading
import traceback
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, unquote, urlparse

import cv2
import numpy as np
import requests
import torch
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
from musetalk.utils.blending import get_image
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
    face_confidence: float = float(os.getenv("MUSETALK_FACE_CONFIDENCE", "0.5"))
    max_download_bytes: int = int(os.getenv("API_MAX_DOWNLOAD_BYTES", str(2 * 1024 * 1024 * 1024)))


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


settings.vae_type = _resolve_vae_type(settings.vae_type)


class LipSyncRequest(BaseModel):
    video_url: str = Field(..., description="Source video URL")
    avatar_url: str = Field(..., description="Reference avatar image URL")
    audio_url: str = Field(..., description="Driving audio URL")
    similarity_threshold: float = Field(0.48, ge=0.0, le=1.0)
    bbox_shift: int = 0
    extra_margin: int = Field(10, ge=0, le=80)
    parsing_mode: str = "jaw"
    left_cheek_width: int = Field(90, ge=1, le=240)
    right_cheek_width: int = Field(90, ge=1, le=240)
    batch_size: int = Field(8, ge=1, le=64)
    audio_padding_length_left: int = Field(2, ge=0, le=10)
    audio_padding_length_right: int = Field(2, ge=0, le=10)


class FaceListRequest(BaseModel):
    video_url: str = Field(..., description="Source video URL")
    similarity_threshold: float = Field(0.62, ge=0.0, le=1.0)
    frame_sample_interval: int = Field(1, ge=1, le=300)
    max_frames: int = Field(0, ge=0, description="0 means scan all sampled frames")
    min_face_area: int = Field(400, ge=1)
    min_detection_score: float = Field(0.85, ge=0.0, le=1.0)
    require_landmark_match: bool = True
    min_landmark_points: int = Field(8, ge=1, le=68)
    min_landmark_overlap: float = Field(0.08, ge=0.0, le=1.0)
    crop_padding: float = Field(0.25, ge=0.0, le=1.0)


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
    try:
        response = requests.get(url, stream=True, timeout=(10, 120))
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download {prefix}: {exc}") from exc

    content_length = response.headers.get("content-length")
    if content_length and int(content_length) > settings.max_download_bytes:
        response.close()
        raise HTTPException(status_code=413, detail=f"{prefix} is larger than API_MAX_DOWNLOAD_BYTES")

    suffix = _guess_suffix(url, response.headers.get("content-type", ""), allowed, fallback)
    output_path = dest_dir / f"{prefix}{suffix}"
    downloaded = 0
    try:
        with output_path.open("wb") as file_obj:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                downloaded += len(chunk)
                if downloaded > settings.max_download_bytes:
                    raise HTTPException(status_code=413, detail=f"{prefix} is larger than API_MAX_DOWNLOAD_BYTES")
                file_obj.write(chunk)
    finally:
        response.close()
    return output_path


def _read_video_frames(video_path: Path) -> Tuple[List[np.ndarray], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

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
        raise RuntimeError(f"No frames were decoded from video: {video_path}")
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


class MuseTalkApiRuntime:
    def __init__(self) -> None:
        self.loaded = False
        self.detectors_loaded = False
        self.load_lock = threading.RLock()
        self.run_lock = threading.Lock()
        self.face_parser_cache: Dict[Tuple[int, int], FaceParsing] = {}

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
            self.detectors_loaded = True

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

            self.audio_processor = AudioProcessor(feature_extractor_path=settings.whisper_dir)
            self.whisper = WhisperModel.from_pretrained(settings.whisper_dir)
            self.whisper = self.whisper.to(device=self.device, dtype=self.weight_dtype).eval()
            self.whisper.requires_grad_(False)
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

    def _face_descriptor(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> Optional[Dict[str, np.ndarray]]:
        clipped = _clip_box(bbox, frame.shape)
        if clipped is None:
            return None
        x1, y1, x2, y2 = clipped
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 4], [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten().astype(np.float32)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        small = small - float(np.mean(small))
        dct = cv2.dct(small)[:8, :8].flatten()

        return {"hist": hist, "dct": _normalize_vector(dct)}

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
        hist_score = cv2.compareHist(left["hist"], right["hist"], cv2.HISTCMP_CORREL)
        hist_score = float(np.clip((hist_score + 1.0) / 2.0, 0.0, 1.0))
        dct_score = float(np.dot(left["dct"], right["dct"]))
        dct_score = float(np.clip((dct_score + 1.0) / 2.0, 0.0, 1.0))
        return 0.45 * hist_score + 0.55 * dct_score

    def _avatar_descriptor(self, avatar_path: Path) -> Dict[str, np.ndarray]:
        avatar = cv2.imread(str(avatar_path))
        if avatar is None:
            raise RuntimeError(f"Could not read avatar image: {avatar_path}")

        boxes = self._detect_face_boxes(avatar)
        if not boxes:
            raise RuntimeError("No face was detected in the avatar image.")

        descriptor = self._face_descriptor(avatar, boxes[0][0])
        if descriptor is None:
            raise RuntimeError("Could not build avatar face descriptor.")
        return descriptor

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
            frames, _ = _read_video_frames(video_path)
            clusters: List[Dict[str, object]] = []
            scanned_frames = 0
            detections = 0
            rejected_low_score = 0
            rejected_shape = 0
            rejected_landmarks = 0

            for frame_index in range(0, len(frames), payload.frame_sample_interval):
                if payload.max_frames and scanned_frames >= payload.max_frames:
                    break

                frame = frames[frame_index]
                scanned_frames += 1
                landmarks = self._pose_face_landmarks(frame) if payload.require_landmark_match else []
                for bbox, detection_score in self._detect_face_boxes(frame):
                    if detection_score < payload.min_detection_score:
                        rejected_low_score += 1
                        continue
                    if _box_area(bbox) < payload.min_face_area:
                        rejected_shape += 1
                        continue
                    if not self._is_reasonable_face_box(bbox, frame.shape):
                        rejected_shape += 1
                        continue
                    if payload.require_landmark_match and not self._face_box_matches_landmarks(
                        landmarks,
                        bbox,
                        frame.shape,
                        payload.min_landmark_points,
                        payload.min_landmark_overlap,
                    ):
                        rejected_landmarks += 1
                        continue
                    descriptor = self._face_descriptor(frame, bbox)
                    if descriptor is None:
                        rejected_shape += 1
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

            clusters.sort(key=lambda item: int(item["max_area"]), reverse=True)
            faces_dir = output_dir / "faces"
            faces_dir.mkdir(parents=True, exist_ok=True)

            face_paths = []
            face_items = []
            for index, cluster in enumerate(clusters):
                face_path = faces_dir / f"face_{index:03d}.jpg"
                cv2.imwrite(str(face_path), cluster["best_crop"])
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
                "scanned_frame_count": scanned_frames,
                "detected_face_count": detections,
                "rejected_low_score_count": rejected_low_score,
                "rejected_shape_count": rejected_shape,
                "rejected_landmark_count": rejected_landmarks,
            }

    def _select_target_bbox(
        self,
        frame: np.ndarray,
        avatar_descriptor: Dict[str, np.ndarray],
        threshold: float,
        bbox_shift: int,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float]:
        face_boxes = self._detect_face_boxes(frame)
        if not face_boxes:
            return None, 0.0

        best_bbox = None
        best_score = -1.0
        for bbox, _ in face_boxes:
            descriptor = self._face_descriptor(frame, bbox)
            if descriptor is None:
                continue
            score = self._descriptor_similarity(avatar_descriptor, descriptor)
            if score > best_score:
                best_score = score
                best_bbox = bbox

        if best_bbox is None or best_score < threshold:
            return None, max(0.0, best_score)

        landmarks = self._pose_face_landmarks(frame)
        target_bbox = self._landmark_bbox_for_face(landmarks, best_bbox, bbox_shift, frame.shape)
        return target_bbox, best_score

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

    def _encode_latents(
        self,
        frames: List[np.ndarray],
        targets: List[Dict[str, object]],
        extra_margin: int,
    ) -> Dict[int, torch.Tensor]:
        latents_by_frame: Dict[int, torch.Tensor] = {}
        for index, target in enumerate(targets):
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
        cycle = list(range(frame_count)) + list(range(frame_count - 1, -1, -1))
        return cycle[output_index % len(cycle)]

    def _run_inference_batches(
        self,
        process_items: List[Tuple[int, torch.Tensor, torch.Tensor]],
        batch_size: int,
    ) -> Dict[int, np.ndarray]:
        generated: Dict[int, np.ndarray] = {}
        for start in range(0, len(process_items), batch_size):
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

    def _write_result_frames(
        self,
        frames: List[np.ndarray],
        targets: List[Dict[str, object]],
        generated: Dict[int, np.ndarray],
        output_frame_count: int,
        output_dir: Path,
        extra_margin: int,
        parsing_mode: str,
        face_parser: Optional[FaceParsing],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_count = len(frames)
        for output_index in range(output_frame_count):
            source_index = self._source_index_for_output(output_index, frame_count)
            original_frame = copy.deepcopy(frames[source_index])
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
            try:
                resized = cv2.resize(result_frame.astype(np.uint8), (x2 - x1, y2 - y1))
                if settings.version == "v15":
                    combined = get_image(
                        original_frame,
                        resized,
                        [x1, y1, x2, y2],
                        mode=parsing_mode,
                        fp=face_parser,
                    )
                else:
                    combined = get_image(original_frame, resized, [x1, y1, x2, y2], fp=face_parser)
            except Exception:
                combined = original_frame
            cv2.imwrite(str(output_dir / f"{output_index:08d}.png"), combined)

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
                "18",
                str(temp_video_path),
            ],
            check=True,
        )

    def _combine_audio(self, audio_path: Path, temp_video_path: Path, output_path: Path) -> None:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "warning",
                "-i",
                str(audio_path),
                "-i",
                str(temp_video_path),
                "-map",
                "1:v:0",
                "-map",
                "0:a:0",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-shortest",
                str(output_path),
            ],
            check=True,
        )

    @torch.no_grad()
    def synthesize(self, payload: LipSyncRequest, paths: Dict[str, Path], job_output_dir: Path) -> Dict[str, object]:
        self.load()
        with self.run_lock:
            frames, fps = _read_video_frames(paths["video"])
            avatar_descriptor = self._avatar_descriptor(paths["avatar"])

            targets = []
            matched_source_frames = 0
            best_scores = []
            for frame in frames:
                bbox, score = self._select_target_bbox(
                    frame,
                    avatar_descriptor,
                    payload.similarity_threshold,
                    payload.bbox_shift,
                )
                targets.append({"bbox": bbox, "score": score})
                best_scores.append(score)
                if bbox is not None:
                    matched_source_frames += 1

            whisper_input_features, librosa_length = self.audio_processor.get_audio_feature(str(paths["audio"]))
            if whisper_input_features is None:
                raise RuntimeError(f"Could not read audio: {paths['audio']}")
            whisper_chunks = self.audio_processor.get_whisper_chunk(
                whisper_input_features,
                self.device,
                self.weight_dtype,
                self.whisper,
                librosa_length,
                fps=fps,
                audio_padding_length_left=payload.audio_padding_length_left,
                audio_padding_length_right=payload.audio_padding_length_right,
            )
            output_frame_count = len(whisper_chunks)
            if output_frame_count == 0:
                raise RuntimeError("Audio is too short to produce video frames.")

            latents_by_frame = self._encode_latents(frames, targets, payload.extra_margin)
            process_items = []
            for output_index in range(output_frame_count):
                source_index = self._source_index_for_output(output_index, len(frames))
                latent = latents_by_frame.get(source_index)
                if latent is None:
                    continue
                process_items.append((output_index, whisper_chunks[output_index], latent))

            generated = self._run_inference_batches(process_items, payload.batch_size) if process_items else {}

            face_parser = (
                self._get_face_parser(payload.left_cheek_width, payload.right_cheek_width)
                if generated
                else None
            )
            render_dir = job_output_dir / "frames"
            self._write_result_frames(
                frames,
                targets,
                generated,
                output_frame_count,
                render_dir,
                payload.extra_margin,
                payload.parsing_mode,
                face_parser,
            )

            temp_video_path = job_output_dir / "temp_video.mp4"
            output_path = job_output_dir / "result.mp4"
            self._frames_to_video(render_dir, fps, temp_video_path)
            self._combine_audio(paths["audio"], temp_video_path, output_path)

            shutil.rmtree(render_dir, ignore_errors=True)
            temp_video_path.unlink(missing_ok=True)

            skipped_output_frames = output_frame_count - len(generated)
            return {
                "output_path": output_path,
                "source_frame_count": len(frames),
                "output_frame_count": output_frame_count,
                "matched_source_frames": matched_source_frames,
                "generated_output_frames": len(generated),
                "skipped_output_frames": skipped_output_frames,
                "best_similarity": max(best_scores) if best_scores else 0.0,
            }


runtime = MuseTalkApiRuntime()


def _output_url(request: Request, output_path: Path) -> str:
    relative = output_path.relative_to(OUTPUT_ROOT).as_posix()
    return f"{str(request.base_url).rstrip('/')}/outputs/{relative}"


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
        "port": settings.port,
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
    avatar_path = _download_to_file(payload.avatar_url, job_input_dir, "avatar", IMAGE_SUFFIXES, ".jpg")
    audio_path = _download_to_file(payload.audio_url, job_input_dir, "audio", AUDIO_SUFFIXES, ".wav")

    try:
        result = runtime.synthesize(
            payload,
            {"video": video_path, "avatar": avatar_path, "audio": audio_path},
            job_output_dir,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    output_path = result.pop("output_path")
    video_url = _output_url(request, output_path)
    return {
        "job_id": job_id,
        "video_url": video_url,
        "download_url": f"{str(request.base_url).rstrip('/')}/api/download?url={quote(video_url, safe='')}",
        **result,
    }


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
        response = requests.get(url, stream=True, timeout=(10, 120))
        response.raise_for_status()
    except requests.RequestException as exc:
        raise HTTPException(status_code=400, detail=f"Failed to download URL: {exc}") from exc

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
    settings.unet_model_path = args.unet_model_path
    settings.unet_config = args.unet_config
    settings.whisper_dir = args.whisper_dir
    settings.vae_type = _resolve_vae_type(args.vae_type)

    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
