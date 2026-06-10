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
- Keep identity matching and speech activity as separate gates: target face matching answers "which person may be edited"; audio activity answers "whether this frame should be driven by lipsync." A frame should be generated only when both gates pass.
- For "speaking moves, silence stays still" behavior, use the audio speech gate before building inference batches. Silent or weak-audio frames should pass through unchanged even if the target face is visible.
- Avoid fixing identity misses by only lowering thresholds. Use short-gap filling and temporal continuity so brief embedding failures do not create audio-mouth discontinuities.
- Keep target matching conservative around other people: fill only short gaps bounded by confirmed target frames, and use position continuity limits to reduce wrong-person edits.
- For lip/audio alignment, preserve fractional video fps. Common rates such as `29.97` and `23.976` must not be truncated to integers when slicing Whisper audio features.
- For visual quality, small bbox smoothing and light color matching before blending are safer first-line improvements than large crop/mask changes.
- Keep output frame count and duration tied to the source video. If audio is shorter, late frames should remain original; if audio is longer, extra audio chunks should not extend the video.
- The current image-sequence renderer must write every output frame for ffmpeg, but MuseTalk inference and heavy blending should only run for frames that actually need modification.

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
- Frames where the target face is visible but the driving audio is silent or below the speech gate should pass through unchanged.
- Short target gaps may be filled when bounded by confirmed target frames; long gaps or large position jumps should remain unchanged.
- API responses should expose enough counters to debug quality, including matched frames, filled frames, eligible frames, generated frames, skipped frames, target identity source, and speech gate statistics.

## Maintenance Checklist

When changing API behavior:

- Keep `/api/faces` crops compatible with `/api/lipsync` avatar detection.
- Preserve structured error responses with both `detail` and traceback logging for server debugging.
- Preserve "pass through unchanged" semantics for no-face, no-target, too-small-face, short-segment, and silent-audio cases.
- Run at least `python -m py_compile api.py` after API edits; include changed utility files in the same command.
- Run `git diff --check` before committing.
- Do not commit generated media, downloaded inputs, model weights, or cache files.
- If a change affects lipsync quality, test on the server when possible because local environments often lack the right GPU/model stack.

## Tuning Methodology & Lessons Learned

Knowledge accumulated from iterative lipsync quality work. Follow these to avoid re-discovering the same pitfalls.

### Filter tuning: change the method before changing the threshold

When a quality gate rejects too many valid frames or lets bad frames through, do not just adjust the threshold:

1. **Identify which gate is firing.** Log per-gate rejection counters (`prefiltered_blur`, `prefiltered_side`, `small_face`, `motion`, `fast_motion`, `mouth_diff`, `lipsync`). The jumping counter tells you which gate is wrong, not the cutoff.
2. **Check that the gate answers the right question.** Identity matching answers *which person may be edited*; speech activity answers *whether this frame should be driven by lipsync*. If a single gate conflates them, split it before tuning.
3. **Reconsider the metric, not just the cutoff.** Whole-face MSE is too coarse to catch local color blocks. CHISQR on the upper-face color histogram plus a Laplacian variance on the mouth region catches what MSE misses. The right metric beats the right number.
4. **Use conservative short-gap filling, not loose thresholds.** Brief identity-miss gaps are normal. Fill bounded gaps with the most-recent confirmed target rather than weakening the identity gate.

### Avoid color-block / boundary artifacts

- Sharp horizontal / vertical cuts in the mask produce visible color seams. Use a linear ramp (e.g. `_soft_upper_mask`) for transitions, not a hard threshold.
- Color match and reference detail restore should be blended, not switched. A soft mask (e.g. 0.40→0.60 transition) avoids the badcase where one half of the lips went solid-color.
- Bbox smoothing only locks position, not content. Add an output-level temporal blend (per-frame weighted average) for frame-to-frame stability.
- Teeth halos and over-sharpening are common when the generated mouth is upscaled and pasted back without matching the reference's color and detail statistics. Lighten the strength, do not disable the step.

### Face lock design

- A "soft" lock must actually implement the fallback. Track the best *unlocked* candidate as well as the best *locked* candidate; prefer locked, fall back to unlocked. Otherwise the docstring lies and output silently dies (e.g. `matched=2, eligible=0, gen=0, passthrough=702`).
- IoU is brittle for face tracking; a small rotation or scale change kills it. Center-distance (relative to face width) is more robust to in-plane motion.
- Lock knobs should default to 0 (opt-in). The default behavior is the safe / permissive path; callers opt into stricter locking per request.
- "Speaking moves, silence stays still" still applies inside a lock: silent frames should pass through unchanged even when the locked target is visible.

### Frame-stability checks

- Frame jitter and mouth-region blur are common when identity, lock, or upscaling fails. Expose a `mouth_sharpen_strength` knob (unsharp mask) and a per-frame `output_temporal_blend` knob. Both are opt-in / tunable, not forced.

### CodeFormer is optional

`CodeFormer fidelity restoration` is a quality add-on, not a required dependency. The loader returns `(None, "<error>")` when the checkpoint is missing and reports `available=False` in the response. If users see `available=False`:

- Verify `models/codeformer/codeformer.pth` exists on the server.
- If missing, either download from the upstream CodeFormer release or set `codeformer_enabled=false` per request. The lipsync pipeline continues to work without it.
- Do not let a missing checkpoint break the request — degrade gracefully.

### MuseTalk is encoder-decoder, not diffusion

MuseTalk generates the mouth region with a single UNet pass conditioned on Whisper audio features via cross-attention. It is **not** a diffusion model. There is no scheduler, timestep loop, or `num_inference_steps`. Do not add diffusion-only fields (cfg scale, denoise strength, eta, etc.) to the API — they are ignored and confuse callers.
