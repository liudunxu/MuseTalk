# Lips-Only Blend + Mouth Color Knob — Design

## Problem

Two distinct quality issues with the synthesized mouth:

1. **Color mismatch on the skin surrounding the lips.** The current
   `parsing_mode="jaw"` blend mask covers class 1 (skin) + classes
   11/12/13 (mouth + upper + lower lip) below the upper boundary.
   MuseTalk's UNet regenerates this whole area, so the model's
   "imagined" cheek / chin skin is pasted back over the source's
   real skin in the same region. On darker-skinned subjects this
   reads as a visible color block.

2. **No knob for the mouth color match.** Other post-process
   steps (`color_match_strength`, `mouth_detail_strength`,
   `mouth_sharpen_strength`) are all Pydantic fields, but
   `_match_mouth_to_skin_tone` is hard-coded at 0.45. There is a
   stale docstring claim of 0.25 that no longer matches the code.

## Goals

- A. Add a `mouth_color_match_strength` Pydantic field (default 0.45)
   so callers can tune the strength per request. Fix the docstring
   inconsistency.
- B. Add a new `parsing_mode="lips_only"` that limits the blend mask
   to just the lips (classes 11/12/13), with mild dilation to catch
   generated lip texture near the edges. Default stays `"jaw"`.

## Non-Goals

- Not changing the model's input mask. The UNet still receives the
  full "jaw" mask and generates a full face crop; we only narrow
  what we paste back. This keeps the model's behavior unchanged and
  makes the change opt-in / low-risk.
- Not retraining anything.
- Not touching upstream `blending.py` semantics for the existing
  "jaw" / "raw" modes.

## Architecture

### A. Mouth color match knob (3 small edits in `api.py`)

1. Add Pydantic field next to existing color knobs:
   ```python
   mouth_color_match_strength: float = Field(0.45, ge=0.0, le=1.0)
   ```

2. Thread it through `_write_result_frames` → `_match_mouth_to_skin_tone`:
   - `__call__` signature gains `mouth_color_match_strength: float = 0.45`
   - The static method `_match_mouth_to_skin_tone` keeps its default
     `strength: float = 0.45` (still no-op for back-compat)
   - The call site passes the param

3. Drop the "lighter 0.25 strength" claim from the docstring; the
   code's default is now the spec.

### B. `lips_only` blend mode (3 edits in `api.py` + `musetalk/utils/blending.py`)

1. **`blending.py`** — extend `get_image_prepare_material` with a
   `lips_only` branch. The function still uses the parser with
   `mode="lips_only"` (already defined in
   `musetalk/utils/face_parsing/__init__.py`). Skip the
   `upper_boundary_ratio` crop (lips can be at any vertical
   position) and add a small dilation (kernel 5x5 ellipse, 1 iter)
   so the blend edge sits a few pixels outside the actual lip
   boundary. Then run the same `mask_blur_ratio` Gaussian pass.

2. **`api.py`** — register `"lips_only"` as a valid `parsing_mode`
   value at the validator (`{"jaw", "raw"}` → `{"jaw", "raw", "lips_only"}`).

3. **`api.py`** — no other changes. The blend material is computed
   by the same `get_image_prepare_material` call; the parser mode
   flows through automatically.

## Why this is low-risk

- **Default behavior unchanged.** `parsing_mode` still defaults to
  `"jaw"`. Existing clients see no change.
- **No model changes.** MuseTalk's UNet still receives the same
  input mask. Only the post-processing blend mask changes.
- **A is purely additive.** New Pydantic field with a sensible
  default; the post-process call site gets the new arg.

## Verification

- `python -m py_compile api.py musetalk/utils/blending.py`
- `vulture --min-confidence 60 api.py` (only FastAPI-handler false
  positives remain)
- `git diff --check` (no whitespace errors)
- Server-side manual test: send a request with
  `parsing_mode="lips_only"` and a small `mouth_color_match_strength`
  adjustment, eyeball the output. The cheeks / chin should now be
  the source's actual skin; the lips should be the model's output
  blended in.
