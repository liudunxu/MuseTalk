# AGENTS.md

Behavioral guidelines for coding agents working in this repository.

Source: https://github.com/multica-ai/andrej-karpathy-skills/blob/main/CLAUDE.md
License: MIT, per the upstream repository README.

Tradeoff: These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

Do not assume, do not hide confusion, and surface tradeoffs.

Before implementing:

- State assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them instead of choosing silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop, name what is confusing, and ask.

## 2. Simplicity First

Write the minimum code that solves the problem. Avoid speculative work.

- No features beyond what was asked.
- No abstractions for single-use code.
- No flexibility or configurability that was not requested.
- No error handling for impossible scenarios.
- If 200 lines could be 50, rewrite it.

Ask: would a senior engineer say this is overcomplicated? If yes, simplify.

## 3. Surgical Changes

Touch only what is necessary. Clean up only your own mess.

When editing existing code:

- Do not improve adjacent code, comments, or formatting unless required.
- Do not refactor things that are not broken.
- Match existing style, even if you would choose a different style elsewhere.
- If you notice unrelated dead code, mention it instead of deleting it.

When your changes create orphans:

- Remove imports, variables, and functions made unused by your changes.
- Do not remove pre-existing dead code unless asked.

Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

Define success criteria and loop until verified.

Transform tasks into verifiable goals:

- "Add validation" means write tests for invalid inputs, then make them pass.
- "Fix the bug" means write or identify a repro, then make it pass.
- "Refactor X" means ensure tests pass before and after.

For multi-step tasks, state a brief plan:

```text
1. [Step] -> verify: [check]
2. [Step] -> verify: [check]
3. [Step] -> verify: [check]
```

Strong success criteria let agents loop independently. Weak criteria like "make it work" require clarification.

These guidelines are working if diffs contain fewer unnecessary changes, implementations avoid needless rewrites, and clarifying questions happen before mistakes rather than after them.

## Repository Runtime Notes

The API may be deployed from a server where the VAE weights directory is named `models/sd-vae-ft-mse` instead of `models/sd-vae`, and where MuseTalk weights are under `models/musetalk` without `models/musetalkV15`. Do not assume local model directory names; resolve or validate actual paths before loading models. Whisper must include `config.json`, `pytorch_model.bin`, and `preprocessor_config.json` in `models/whisper`.

The production API is run on the server, not only on the local workstation. Local syntax checks are useful, but full lipsync validation usually requires the server GPU/model environment.

When working on `/api/lipsync`:

- Prefer InsightFace embeddings for identity decisions. The avatar image may be omitted; in that case select the most frequently observed face identity from the input video.
- If a frame has no matching target face, keep the original frame. Do not raise `No Face Detected` for intermittent misses.
- Avoid fixing identity misses by only lowering thresholds. Use short-gap filling and temporal continuity so brief embedding failures do not create audio-mouth discontinuities.
- Keep target matching conservative around other people: fill only short gaps bounded by confirmed target frames, and use position continuity limits to reduce wrong-person edits.
- For lip/audio alignment, preserve fractional video fps. Common rates such as `29.97` and `23.976` must not be truncated to integers when slicing Whisper audio features.
- For visual quality, small bbox smoothing and light color matching before blending are safer first-line improvements than large crop/mask changes.

When working on `/api/faces`, sort distinct identities by observed count first, then by face area as a tie-breaker. The face crops produced by this API should remain compatible with `/api/lipsync` avatar detection.

InsightFace/ONNXRuntime should use GPU providers on the server when available. If logs show `CPUExecutionProvider`, check the installed `onnxruntime-gpu` build against the server CUDA stack before changing matching code for performance.

## Code Structure

Primary entry points:

- `api.py`: FastAPI service used by the deployed server. It owns request models, download handling, output URLs, model/runtime loading, face identity selection, lipsync inference orchestration, frame rendering, and `/api/download`.
- `app.py`: Original Gradio/demo-style entry point. Do not assume API behavior is shared with `api.py`; check both before moving logic.
- `scripts/inference.py`, `scripts/realtime_inference.py`, `scripts/preprocess.py`: CLI/offline workflows from the upstream project.

Core MuseTalk modules:

- `musetalk/utils/audio_processor.py`: Whisper feature extraction for the API path. Keep fps handling precise; do not truncate fractional fps.
- `musetalk/utils/utils.py`: Model loading helpers and upstream inference utilities.
- `musetalk/utils/preprocessing.py`: Face detection, landmark/DWPose preprocessing, and shared detector globals used by `api.py`.
- `musetalk/utils/blending.py`: Face mask preparation and generated-mouth blending back into original frames.
- `musetalk/utils/face_parsing/`: Face parsing model used to build blend masks.
- `musetalk/utils/face_detection/`: SFD face detector implementation.
- `musetalk/data/`, `musetalk/loss/`, `train.py`: Training/data code; avoid touching these for API-only fixes unless the request is explicitly about training.

Runtime directories:

- `models/`: Server-side model weights. Names vary between deployments, so resolve and validate actual files rather than hard-coding one upstream layout.
- `results/api/inputs/`: Downloaded request inputs for API jobs.
- `results/api/outputs/`: Generated API outputs served under `/outputs/...`.

## API Surface

`api.py` exposes:

- `GET /health`: Reports detector/model/load state and embedding backend status.
- `POST /api/faces`: Takes `video_url`, extracts distinct identities, and returns face URLs sorted by observed identity count first.
- `POST /api/lipsync`: Takes `video_url`, `audio_url`, and optional `avatar_url`; outputs a lipsync video URL and download URL.
- `GET /api/download`: Downloads generated local outputs or validated remote URLs.

Important `/api/lipsync` behavior:

- With `avatar_url`, only the matching identity should be modified.
- Without `avatar_url`, the default target is the identity with the highest observed count in the video.
- Frames without a matching target must pass through unchanged.
- Short target gaps may be filled when bounded by confirmed target frames; long gaps or large position jumps should remain unchanged.

## Maintenance Checklist

When changing API behavior:

- Keep `/api/faces` crops compatible with `/api/lipsync` avatar detection.
- Preserve structured error responses with both `detail` and traceback logging for server debugging.
- Run at least `python -m py_compile api.py` after API edits; include changed utility files in the same command.
- Run `git diff --check` before committing.
- Do not commit generated media, downloaded inputs, model weights, or cache files.
- If a change affects lipsync quality, test on the server when possible because local environments often lack the right GPU/model stack.
