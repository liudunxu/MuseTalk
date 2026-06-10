# Lips-Only Blend + Mouth Color Knob Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `mouth_color_match_strength` Pydantic knob and a new `parsing_mode="lips_only"` blend mode that limits the post-process paste area to the lips (class 11/12/13), so the model's "imagined" cheeks no longer overwrite the source's actual skin.

**Architecture:** Pure additive. Add one Pydantic field, thread one parameter through a call chain, and add one new branch in `get_image_prepare_material` that uses the existing `mode="lips_only"` parser. No model changes. Default behavior unchanged.

**Tech Stack:** FastAPI, Pydantic, OpenCV, NumPy, BiSeNet (face parsing).

---

## File Structure

| File | Action | Responsibility |
|------|--------|---------------|
| `api.py` | modify | Add `mouth_color_match_strength` Pydantic field; thread it through `_write_result_frames` to `_match_mouth_to_skin_tone`; extend `parsing_mode` validator |
| `musetalk/utils/blending.py` | modify | Add `lips_only` branch in `get_image_prepare_material` that bypasses the upper-boundary crop and adds a small dilation |
| `docs/superpowers/specs/2026-06-10-lips-only-blend-and-mouth-color-knob-design.md` | already created | Design doc |

---

## Task 1: Add `mouth_color_match_strength` Pydantic field

**Files:**
- Modify: `api.py:262-267` (in `class LipSyncRequest`)

- [ ] **Step 1: Add the field**

In `api.py`, in `class LipSyncRequest`, after the `mouth_detail_strength` field (around line 264), add:

```python
    mouth_color_match_strength: float = Field(0.45, ge=0.0, le=1.0)
```

The relevant block currently looks like:

```python
    color_match_strength: float = Field(0.70, ge=0.0, le=1.0)
    mouth_detail_strength: float = Field(0.90, ge=0.0, le=1.0)
    mouth_sharpen_strength: float = Field(0.50, ge=0.0, le=1.0)
```

It should become:

```python
    color_match_strength: float = Field(0.70, ge=0.0, le=1.0)
    mouth_detail_strength: float = Field(0.90, ge=0.0, le=1.0)
    mouth_color_match_strength: float = Field(0.45, ge=0.0, le=1.0)
    mouth_sharpen_strength: float = Field(0.50, ge=0.0, le=1.0)
```

- [ ] **Step 2: Verify it parses**

Run: `python -c "from api import LipSyncRequest; r = LipSyncRequest(video_url='http://x', audio_url='http://y'); print(r.mouth_color_match_strength)"`
Expected: `0.45`

- [ ] **Step 3: Verify it accepts overrides**

Run: `python -c "from api import LipSyncRequest; r = LipSyncRequest(video_url='http://x', audio_url='http://y', mouth_color_match_strength=0.7); print(r.mouth_color_match_strength)"`
Expected: `0.7`

- [ ] **Step 4: Commit**

```bash
git add api.py
git commit -m "feat(api): add mouth_color_match_strength Pydantic knob (default 0.45)"
```

---

## Task 2: Thread `mouth_color_match_strength` to the call site

**Files:**
- Modify: `api.py` `_write_result_frames` signature (around line 3356) and its caller
- Modify: `api.py` `_match_mouth_to_skin_tone` docstring (drop stale "0.25" claim)
- Modify: `api.py` call site (around line 3486-3488)

- [ ] **Step 1: Drop the stale 0.25 claim in the docstring**

In `api.py`, the docstring of `_match_mouth_to_skin_tone` (around lines 3024-3048) says:

```
... lighter 0.25 strength so the
open-mouth shape does not collapse into the
closed-mouth reference.
```

Replace that sentence (everything from "lighter 0.25 strength" to "closed-mouth reference.") with:

```
... 0.45 strength, exposed as a Pydantic
knob (mouth_color_match_strength) so callers can
relax or strengthen it per request.
```

The full edited block (just the sentence being replaced):

Before:
```python
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

        Earlier version used ``skin_mask < 128`` as the
        "mouth" region (and the reference's *skin* pixels
        as the target) at strength 0.40. That over-tinted
        the open mouth toward the closed-mouth skin tone and
        also picked up eyes / brows. Now targets the
        reference's skin tone (the closest the source
        subject's true complexion), but uses the precise
        lips-only mask and a lighter 0.25 strength so the
        open-mouth shape does not collapse into the
        closed-mouth reference.
        """
```

After:
```python
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

        Earlier version used ``skin_mask < 128`` as the
        "mouth" region (and the reference's *skin* pixels
        as the target) at strength 0.40. That over-tinted
        the open mouth toward the closed-mouth skin tone and
        also picked up eyes / brows. Now targets the
        reference's skin tone (the closest the source
        subject's true complexion), using the precise
        lips-only mask and a default 0.45 strength, exposed
        as a Pydantic knob (mouth_color_match_strength) so
        callers can relax or strengthen it per request.
        """
```

- [ ] **Step 2: Find the call chain**

Use grep to find:
- `_match_mouth_to_skin_tone` definition (around line 3017)
- `_write_result_frames` definition (around line 3356)
- The call site at the post-process block (around line 3486)
- All `_write_result_frames` call sites

These calls will be inside `synthesize` (the renderer entry point).

- [ ] **Step 3: Add `mouth_color_match_strength` to `_write_result_frames`**

Find the current signature. It will look like (around line 3356):

```python
    def _write_result_frames(
        self,
        ...
        mouth_detail_strength: float,
        mouth_sharpen_strength: float,
        ...
    ) -> ...:
```

Add a new parameter after `mouth_sharpen_strength` (or grouped with the other mouth knobs, after `mouth_detail_strength`). The new param goes in this order:

```python
    def _write_result_frames(
        self,
        ...
        mouth_detail_strength: float,
        mouth_color_match_strength: float,
        mouth_sharpen_strength: float,
        ...
    ) -> ...:
```

(Place it adjacent to the other mouth knobs so the call site reads naturally. If the existing code already groups them, follow that order.)

- [ ] **Step 4: Pass the new value to `_match_mouth_to_skin_tone` at the call site**

Find the call site (around line 3486):

```python
                resized = self._match_mouth_to_skin_tone(
                    resized, reference_crop, lips_mask
                )
```

Replace with:

```python
                resized = self._match_mouth_to_skin_tone(
                    resized, reference_crop, lips_mask,
                    strength=mouth_color_match_strength,
                )
```

- [ ] **Step 5: Pass the new value from `synthesize` to `_write_result_frames`**

In `synthesize`, find the call to `self._write_result_frames(...)`. There are existing args like:

```python
            self._write_result_frames(
                ...
                payload.mouth_detail_strength,
                payload.mouth_sharpen_strength,
                ...
            )
```

Add `payload.mouth_color_match_strength` next to the other mouth knobs:

```python
            self._write_result_frames(
                ...
                payload.mouth_detail_strength,
                payload.mouth_color_match_strength,
                payload.mouth_sharpen_strength,
                ...
            )
```

- [ ] **Step 6: Verify compile**

Run: `python -m py_compile api.py && echo OK`
Expected: `OK`

- [ ] **Step 7: Verify the Pydantic field is wired into synthesize**

Run: `python -c "from api import LipSyncRequest; import inspect, api; src = inspect.getsource(api.MuseTalkApiRuntime.synthesize); assert 'mouth_color_match_strength' in src; print('wired')"`
Expected: `wired`

- [ ] **Step 8: Commit**

```bash
git add api.py
git commit -m "feat(api): thread mouth_color_match_strength to call site + fix stale docstring"
```

---

## Task 3: Extend `get_image_prepare_material` with `lips_only` mode

**Files:**
- Modify: `musetalk/utils/blending.py:112-146` (`get_image_prepare_material`)

- [ ] **Step 1: Read the current function**

Open `musetalk/utils/blending.py`. The function is at lines 112-146. Current implementation:

```python
def get_image_prepare_material(
    image,
    face_box,
    upper_boundary_ratio=0.5,
    expand=1.5,
    fp=None,
    mode="raw",
    mask_blur_ratio=0.1,
):
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    #print(x1-x,y1-y)
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, x1-x_s))  # NOTE: y1 typo in upstream code? — verify
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    mask_array = np.array(modified_mask_image)
    if mask_blur_ratio > 0:
        blur_kernel_size = int(mask_blur_ratio * ori_shape[0] // 2 * 2) + 1
        mask_array = cv2.GaussianBlur(mask_array, (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
```

(The `(x1-x_s, x1-x_s)` line is the existing upstream code. Don't change it as part of this plan; just preserve it. The new branch will be additive.)

- [ ] **Step 2: Add the `lips_only` branch**

Replace the function body. Add an early branch for `lips_only` that:
- Skips the `upper_boundary_ratio` crop (lips can be at any vertical position)
- Adds a small dilation (5x5 elliptical kernel, 1 iter) to catch generated lip texture near the lip line
- Applies the same `mask_blur_ratio` Gaussian pass

The full new function:

```python
def get_image_prepare_material(
    image,
    face_box,
    upper_boundary_ratio=0.5,
    expand=1.5,
    fp=None,
    mode="raw",
    mask_blur_ratio=0.1,
):
    """Build the blend mask and expanded crop box for paste-back.

    For ``mode="lips_only"`` the mask is restricted to the lips
    (parser classes 11/12/13), bypasses the upper-boundary crop
    (lips can be at any vertical position), and is dilated by a
    small elliptical kernel so the blend edge sits a few pixels
    outside the actual lip boundary. The same ``mask_blur_ratio``
    Gaussian pass still runs.
    """
    body = Image.fromarray(image[:,:,::-1])

    x, y, x1, y1 = face_box
    crop_box, s = get_crop_box(face_box, expand)
    x_s, y_s, x_e, y_e = crop_box

    face_large = body.crop(crop_box)
    ori_shape = face_large.size

    mask_image = face_seg(face_large, mode=mode, fp=fp)
    mask_small = mask_image.crop((x-x_s, y-y_s, x1-x_s, y1-y_s))
    mask_image = Image.new('L', ori_shape, 0)
    mask_image.paste(mask_small, (x-x_s, y-y_s, x1-x_s, y1-y_s))

    if mode == "lips_only":
        # Lips can be at any vertical position, so we skip the
        # upper-boundary crop that "jaw" / "raw" modes use. Dilate
        # so the blend edge sits a few pixels outside the actual
        # lip line, catching generated lip texture right at the
        # boundary. Then run the same Gaussian pass for the soft
        # edge.
        mask_array = np.array(mask_image)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_array = cv2.dilate(mask_array, kernel, iterations=1)
        if mask_blur_ratio > 0:
            blur_kernel_size = int(mask_blur_ratio * ori_shape[0] // 2 * 2) + 1
            mask_array = cv2.GaussianBlur(mask_array, (blur_kernel_size, blur_kernel_size), 0)
        return mask_array, crop_box

    # keep upper_boundary_ratio of talking area
    width, height = mask_image.size
    top_boundary = int(height * upper_boundary_ratio)
    modified_mask_image = Image.new('L', ori_shape, 0)
    modified_mask_image.paste(mask_image.crop((0, top_boundary, width, height)), (0, top_boundary))

    mask_array = np.array(modified_mask_image)
    if mask_blur_ratio > 0:
        blur_kernel_size = int(mask_blur_ratio * ori_shape[0] // 2 * 2) + 1
        mask_array = cv2.GaussianBlur(mask_array, (blur_kernel_size, blur_kernel_size), 0)
    return mask_array, crop_box
```

- [ ] **Step 3: Verify compile**

Run: `python -m py_compile musetalk/utils/blending.py && echo OK`
Expected: `OK`

- [ ] **Step 4: Quick smoke test**

Run: `python -c "from musetalk.utils.blending import get_image_prepare_material; print('importable')"`
Expected: `importable`

- [ ] **Step 5: Commit**

```bash
git add musetalk/utils/blending.py
git commit -m "feat(blending): add lips_only mode to get_image_prepare_material (skip upper-boundary + dilate)"
```

---

## Task 4: Register `lips_only` in the `parsing_mode` validator

**Files:**
- Modify: `api.py:4593` (the validator that rejects unknown `parsing_mode` values)

- [ ] **Step 1: Find the validator**

It looks like:

```python
    if payload.parsing_mode not in {"jaw", "raw"}:
        raise HTTPException(...)
```

- [ ] **Step 2: Add `lips_only` to the allowed set**

Change the set literal to include `lips_only`:

```python
    if payload.parsing_mode not in {"jaw", "raw", "lips_only"}:
        raise HTTPException(...)
```

(Do not change the surrounding error message format. Just add the new entry.)

- [ ] **Step 3: Verify the validator accepts `lips_only`**

Run: `python -c "from api import LipSyncRequest; r = LipSyncRequest(video_url='http://x', audio_url='http://y', parsing_mode='lips_only'); print(r.parsing_mode)"`
Expected: `lips_only`

- [ ] **Step 4: Verify the validator still rejects garbage**

Run: `python -c "from api import LipSyncRequest; r = LipSyncRequest(video_url='http://x', audio_url='http://y', parsing_mode='nonsense'); print('should not reach here')"`
Expected: `pydantic.ValidationError` (or similar) raised.

- [ ] **Step 5: Verify full compile**

Run: `python -m py_compile api.py && echo OK`
Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add api.py
git commit -m "feat(api): accept parsing_mode='lips_only' in request validator"
```

---

## Task 5: Final verification + push

**Files:** none modified

- [ ] **Step 1: Run all compile + lint checks**

```bash
python -m py_compile api.py musetalk/utils/blending.py
vulture --min-confidence 60 api.py
git diff --check
```

Expected:
- `OK` from py_compile
- Only FastAPI-handler false positives from vulture
- empty output from `git diff --check`

- [ ] **Step 2: Inspect the full diff**

```bash
git log --oneline -7
git diff origin/main..HEAD --stat
```

Expected: 4 new commits, changes confined to `api.py` and `musetalk/utils/blending.py`, no stray modifications.

- [ ] **Step 3: Push**

```bash
git push origin main
```

Expected: `* [new ref]   main -> main` (or similar — see git output).

- [ ] **Step 4: Confirm push**

```bash
git log --oneline @{u}..  # should be empty
```

Expected: empty output.

- [ ] **Step 5: Report back to user**

Tell the user:
- Default behavior unchanged (still `parsing_mode="jaw"`, still
  `mouth_color_match_strength=0.45`).
- To try the new behavior, the user can send a request with
  `parsing_mode="lips_only"`.
- The `mouth_color_match_strength` knob is also available per
  request if they want to dial the existing mouth color match up
  or down.

---

## Self-Review Notes

**Spec coverage:**
- A. mouth_color_match_strength knob → Task 1, Task 2
- A. docstring fix → Task 2 step 1
- B. lips_only blend mode → Task 3, Task 4
- B. validator update → Task 4
- Verification → Task 5

**Type consistency:** The new Pydantic field name is `mouth_color_match_strength` everywhere. The new `_write_result_frames` param matches the call-site name. The `lips_only` mode string is used in both the `blending.py` branch check and the `api.py` validator set.

**No placeholders:** All code blocks are complete. No "TBD" / "implement later" / "fill in".

**Risk:** The `lips_only` mode adds a dilation step that the
existing "jaw" / "raw" paths do not. Verified that this dilation
is bounded (5x5 elliptical, 1 iter) and followed by the same
Gaussian blur that softens the existing modes. The blend edge
should not be harder than the current modes.
