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
    face_confidence: float = float(os.getenv("MUSETALK_FACE_CONFIDENCE", "0.5"))
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
settings.unet_config = _resolve_model_file(
    settings.unet_config,
    ["models/musetalkV15/musetalk.json", "models/musetalk/musetalk.json"],
)
settings.unet_model_path = _resolve_model_file(
    settings.unet_model_path,
    ["models/musetalkV15/unet.pth", "models/musetalk/pytorch_model.bin"],
)
settings.whisper_dir = _resolve_model_dir(
    settings.whisper_dir,
    ["models/whisper"],
    ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
)


class LipSyncRequest(BaseModel):
    video_url: str = Field(..., description="Source video URL")
    avatar_url: Optional[str] = Field(None, description="Reference avatar image URL")
    audio_url: str = Field(..., description="Driving audio URL")
    similarity_threshold: float = Field(0.52, ge=0.0, le=1.0)
    identity_margin: float = Field(0.0, ge=0.0, le=1.0)
    require_face_embedding: bool = True
    allow_crop_embedding_fallback: bool = True
    crop_embedding_min_detection_score: float = Field(0.0, ge=0.0, le=1.0)
    temporal_tracking_weight: float = Field(0.08, ge=0.0, le=0.5)
    target_fill_max_gap_seconds: float = Field(0.6, ge=0.0, le=3.0)
    target_fill_window_seconds: float = Field(2.0, ge=0.1, le=10.0)
    target_fill_min_match_ratio: float = Field(0.40, ge=0.0, le=1.0)
    target_fill_max_center_shift: float = Field(1.5, ge=0.0, le=5.0)
    target_bbox_smoothing_window: int = Field(5, ge=1, le=15)
    target_bbox_smoothing_max_center_shift: float = Field(0.75, ge=0.0, le=5.0)
    identity_scan_interval: int = Field(0, ge=0, le=300, description="0 means scan about 2 frames per second")
    identity_scan_max_frames: int = Field(0, ge=0, description="0 means scan all sampled identity frames")
    identity_scan_require_landmark_match: bool = False
    min_detection_score: float = Field(0.5, ge=0.0, le=1.0)
    require_landmark_match: bool = True
    min_landmark_points: int = Field(8, ge=1, le=68)
    min_landmark_overlap: float = Field(0.08, ge=0.0, le=1.0)
    bbox_shift: int = 0
    extra_margin: int = Field(18, ge=0, le=100)
    parsing_mode: str = "jaw"
    blend_upper_boundary_ratio: float = Field(0.58, ge=0.0, le=1.0)
    blend_mask_blur_ratio: float = Field(0.06, ge=0.0, le=0.2)
    color_match_strength: float = Field(0.35, ge=0.0, le=1.0)
    mouth_sharpen_strength: float = Field(0.25, ge=0.0, le=1.0)
    left_cheek_width: int = Field(75, ge=1, le=240)
    right_cheek_width: int = Field(75, ge=1, le=240)
    batch_size: int = Field(8, ge=1, le=64)
    audio_padding_length_left: int = Field(2, ge=0, le=10)
    audio_padding_length_right: int = Field(2, ge=0, le=10)


class FaceListRequest(BaseModel):
    video_url: str = Field(..., description="Source video URL")
    similarity_threshold: float = Field(0.78, ge=0.0, le=1.0)
    frame_sample_interval: int = Field(0, ge=0, le=300, description="0 means sample about 2 frames per second")
    max_frames: int = Field(0, ge=0, description="0 means scan all sampled frames")
    min_face_area: int = Field(400, ge=1)
    min_detection_score: float = Field(0.8, ge=0.0, le=1.0)
    require_face_embedding: bool = True
    require_landmark_match: bool = True
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

    def _find_target_identity(
        self,
        frames: List[np.ndarray],
        fps: float,
        avatar_descriptor: Optional[Dict[str, np.ndarray]],
        payload: LipSyncRequest,
    ) -> Optional[Dict[str, object]]:
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
                    payload.similarity_threshold,
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
            clusters.sort(
                key=lambda item: (int(item["count"]), int(item["max_area"])),
                reverse=True,
            )
            best_cluster = clusters[0]
            best_cluster["avatar_score"] = 0.0
            best_cluster["target_descriptors"] = list(best_cluster.get("descriptors") or [])
            negative_descriptors = []
            for cluster in clusters[1:]:
                negative_descriptors.extend(cluster.get("descriptors") or [])
            best_cluster["negative_descriptors"] = negative_descriptors
            return best_cluster

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
            return None

        target_descriptors = [
            descriptor
            for descriptor in (best_cluster.get("descriptors") or [])
            if self._descriptor_similarity(avatar_descriptor, descriptor) >= payload.similarity_threshold
        ]
        if not target_descriptors:
            target_descriptors = list(best_cluster.get("descriptors") or [])
        negative_descriptors = []
        for cluster in clusters[1:]:
            descriptors = cluster.get("descriptors") or []
            target_score = max(
                self._descriptor_similarity(target_descriptor, descriptor)
                for target_descriptor in target_descriptors
                for descriptor in descriptors
            )
            avatar_score = float(cluster["avatar_score"])
            if avatar_score >= payload.similarity_threshold and target_score >= payload.similarity_threshold:
                target_descriptors.extend(descriptors)
            else:
                negative_descriptors.extend(descriptors)
        best_cluster["target_descriptors"] = target_descriptors
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
        if frame_count <= 1:
            return 0
        cycle_length = frame_count * 2
        cycle_index = output_index % cycle_length
        if cycle_index < frame_count:
            return cycle_index
        return cycle_length - cycle_index - 1

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
        image_mean, image_std = cv2.meanStdDev(image_float)
        reference_mean, reference_std = cv2.meanStdDev(reference_float)
        image_mean = image_mean.reshape(1, 1, 3)
        image_std = image_std.reshape(1, 1, 3)
        reference_mean = reference_mean.reshape(1, 1, 3)
        reference_std = reference_std.reshape(1, 1, 3)
        matched = (image_float - image_mean) * (reference_std / np.maximum(image_std, 1.0)) + reference_mean
        blended = image_float * (1.0 - strength) + matched * strength
        return np.clip(blended, 0, 255).astype(np.uint8)

    def _sharpen_image(self, image: np.ndarray, strength: float) -> np.ndarray:
        if strength <= 0.0 or image.size == 0:
            return image
        blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
        sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

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
        mouth_sharpen_strength: float,
        face_parser: Optional[FaceParsing],
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        frame_count = len(frames)
        blend_materials: Dict[int, Optional[Tuple[np.ndarray, Tuple[int, int, int, int]]]] = {}
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
                resized = self._match_color_stats(resized, reference_crop, color_match_strength)
                resized = self._sharpen_image(resized, mouth_sharpen_strength)
                mask_array, crop_box = material
                combined = get_image_blending(
                    original_frame,
                    resized,
                    [x1, y1, x2, y2],
                    mask_array,
                    crop_box,
                )
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
                "15",
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
        self.load_detectors()
        with self.run_lock:
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
                return {
                    "passthrough": True,
                    "passthrough_reason": "no_face_detected",
                    "source_frame_count": len(frames),
                    "output_frame_count": len(frames),
                    "matched_source_frames": 0,
                    "filled_source_frames": 0,
                    "smoothed_source_frames": 0,
                    "matched_or_filled_source_frames": 0,
                    "generated_output_frames": 0,
                    "skipped_output_frames": len(frames),
                    "best_similarity": 0.0,
                    "target_identity_similarity": 0.0,
                    "target_identity_count": 0,
                    "target_identity_source": "none",
                    "face_identity_backend": "embedding" if payload.require_face_embedding else "visual",
                }
            target_descriptors = target_identity.get("target_descriptors") if target_identity else []
            negative_descriptors = target_identity.get("negative_descriptors") if target_identity else []
            target_identity_score = (
                float(target_identity["avatar_score"])
                if target_identity and avatar_descriptor is not None
                else 0.0
            )
            target_identity_count = int(target_identity["count"]) if target_identity else 0
            face_identity_backend = "embedding" if payload.require_face_embedding else "visual"
            target_identity_source = "avatar" if avatar_descriptor is not None else "most_frequent_face"

            targets = []
            matched_source_frames = 0
            best_scores = []
            previous_bbox = None
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
                else:
                    bbox, score = None, 0.0
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
            smoothed_source_frames = self._smooth_target_bboxes(
                targets,
                frames[0].shape if frames else (0, 0, 3),
                payload.target_bbox_smoothing_window,
                payload.target_bbox_smoothing_max_center_shift,
            )

            self.load()
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
                payload.blend_upper_boundary_ratio,
                payload.blend_mask_blur_ratio,
                payload.color_match_strength,
                payload.mouth_sharpen_strength,
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
                "filled_source_frames": filled_source_frames,
                "smoothed_source_frames": smoothed_source_frames,
                "matched_or_filled_source_frames": matched_source_frames + filled_source_frames,
                "generated_output_frames": len(generated),
                "skipped_output_frames": skipped_output_frames,
                "best_similarity": max(best_scores) if best_scores else 0.0,
                "target_identity_similarity": target_identity_score,
                "target_identity_count": target_identity_count,
                "target_identity_source": target_identity_source,
                "face_identity_backend": face_identity_backend,
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
        "face_embedding_loaded": runtime.face_embedder is not None,
        "crop_face_embedding_loaded": runtime.face_recognition_session is not None,
        "face_embedding_backend": settings.face_embedding_backend,
        "face_embedding_error": runtime.face_embedding_error,
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
    audio_path = _download_to_file(payload.audio_url, job_input_dir, "audio", AUDIO_SUFFIXES, ".wav")
    avatar_path = (
        _download_to_file(payload.avatar_url, job_input_dir, "avatar", IMAGE_SUFFIXES, ".jpg")
        if payload.avatar_url
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
    settings.unet_model_path = _resolve_model_file(
        args.unet_model_path,
        ["models/musetalkV15/unet.pth", "models/musetalk/pytorch_model.bin"],
    )
    settings.unet_config = _resolve_model_file(
        args.unet_config,
        ["models/musetalkV15/musetalk.json", "models/musetalk/musetalk.json"],
    )
    settings.whisper_dir = _resolve_model_dir(
        args.whisper_dir,
        ["models/whisper"],
        ["config.json", "pytorch_model.bin", "preprocessor_config.json"],
    )
    settings.vae_type = _resolve_vae_type(args.vae_type)

    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port)
