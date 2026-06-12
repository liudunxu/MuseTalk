"""Segment-consistency primitives used by ``/api/lipsync``.

Extracted from ``api.py`` so the most-tweaked decision in the
pipeline -- "send this frame to MuseTalk, or pass it through" --
can be unit-tested in isolation without importing the heavy ML
deps (``transformers``, ``diffusers``, ``mmcv``...) that
``api.py`` requires at module load.

Public API:
  * :func:`enforce_segment_consistency` -- the post-render
    merge / majority-vote / min-duration downgrade decision.
  * :func:`face_crop_histogram_distance` -- the cheap signal
    behind hard-cut / track-switch detection.
  * :func:`last_track_id` and :func:`first_track_id` -- the
    track_id lookup helpers used by the track-aware merge gate.

These were originally static methods on
:class:`MuseTalkApiRuntime` and module-level helpers in
``api.py``.  Behaviour is unchanged; the move is purely a
testability refactor (see ``tests/test_segment_consistency.py``).

Behaviour history (oldest -> newest):
  * strict all-or-nothing (commit 63012ec)
  * majority vote (commit e396ebf)
  * time-window merge (commit aa8a653)
  * hard cut + track-aware + min-merged (commit 4b4987a)
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Track-id lookup helpers (formerly _last_track_id / _first_track_id in api.py)
# ---------------------------------------------------------------------------


def last_track_id(
    targets: List[Dict[str, object]], end_exclusive: int, fallback_index: int
) -> Optional[int]:
    """Return the track_id of the last valid (bbox!=None) frame in
    ``targets[max(0, fallback_index):end_exclusive]`` scanning
    backwards. Returns None when no valid frame in the window has
    a track_id -- callers should treat that as "no opinion" and
    fall back to the time-window merge.
    """
    upper = min(end_exclusive, len(targets))
    lower = max(0, fallback_index)
    for ti in range(upper - 1, lower - 1, -1):
        if targets[ti].get("bbox") is None:
            continue
        track_id = targets[ti].get("track_id")
        if track_id is not None:
            return int(track_id)
    return None


def first_track_id(
    targets: List[Dict[str, object]], start: int, end_exclusive: int
) -> Optional[int]:
    """Return the track_id of the first valid (bbox!=None) frame
    in ``targets[start:end_exclusive]``. Returns None when no
    valid frame in the window has a track_id.
    """
    upper = min(end_exclusive, len(targets))
    for ti in range(start, upper):
        if targets[ti].get("bbox") is None:
            continue
        track_id = targets[ti].get("track_id")
        if track_id is not None:
            return int(track_id)
    return None


# ---------------------------------------------------------------------------
# Face-crop histogram distance (formerly MuseTalkApiRuntime._face_crop_histogram_distance)
# ---------------------------------------------------------------------------


def face_crop_histogram_distance(
    crop_a: np.ndarray,
    crop_b: np.ndarray,
    bins: int = 16,
    resize_to: int = 32,
) -> float:
    """Return 1 - histogram_intersection over downsized BGR crops.

    Used for hard-cut / track-switch detection. Cheaper than the
    upper-face color histogram used for quality gating (no
    per-channel split, smaller crops, fewer bins) so it can be
    called O(N) over a video's target sequence. Returns 0.0 on
    shape mismatch / empty input so callers can safely skip a
    pair.

    Direction matches the upper-face ``_face_color_histogram_distance``
    in ``api.py`` -- 0.0 = identical content, ~1.0 = unrelated
    content. The default ``hard_cut_distance_threshold`` (0.65)
    is tuned to fire on a real cross-character / cross-shot jump
    while not triggering on detector jitter or mild head turns.
    """
    if (
        crop_a is None
        or crop_b is None
        or crop_a.size == 0
        or crop_b.size == 0
    ):
        return 0.0
    if resize_to <= 0:
        return 0.0
    # Normalize the two crops to a fixed small size so the
    # histogram is comparable across frames with slightly
    # different bbox sizes / aspect ratios.
    a = cv2.resize(crop_a, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    b = cv2.resize(crop_b, (resize_to, resize_to), interpolation=cv2.INTER_AREA)
    a_hist = cv2.calcHist([a], [0, 1, 2], None, [bins] * 3, [0, 256] * 3)
    b_hist = cv2.calcHist([b], [0, 1, 2], None, [bins] * 3, [0, 256] * 3)
    cv2.normalize(a_hist, a_hist)
    cv2.normalize(b_hist, b_hist)
    intersection = float(
        cv2.compareHist(a_hist, b_hist, cv2.HISTCMP_INTERSECT)
    )
    # intersection is in [0, 1] for normalized histograms; the
    # distance is 1 - intersection so larger = more different.
    return max(0.0, 1.0 - intersection)


# ---------------------------------------------------------------------------
# Main entry point (formerly MuseTalkApiRuntime._enforce_segment_consistency)
# ---------------------------------------------------------------------------


def enforce_segment_consistency(
    provenance: List[str],
    targets: List[Dict[str, object]],
    frames: List[np.ndarray],
    write_frame: Callable[[int, np.ndarray], None],
    passthrough_ratio: float = 0.5,
    merge_window_seconds: float = 0.0,
    source_fps: float = 25.0,
    hard_cut_enabled: bool = True,
    hard_cut_threshold: float = 0.65,
    track_aware: bool = True,
    min_merged_seconds: float = 1.5,
) -> Tuple[int, Dict[str, int]]:
    """For each contiguous "valid" run (every frame in the run
    had a detectable target face), check the run's
    passthrough ratio.  When the ratio exceeds
    ``passthrough_ratio`` (default 0.5 -- "passthrough is
    the majority"), the whole run is forced to passthrough
    by rewriting the generated frames with the source
    frame.  Segments where generated is the majority are
    left alone so as many frames as possible get
    synthesized.

    When ``merge_window_seconds`` > 0, contiguous valid
    runs separated by a gap of fewer than that many
    seconds are merged into a single run before the
    majority vote.  This is the "speaker smoothing"
    knob: a brief passthrough gap (detector jitter,
    occluded frame, short head turn) inside an otherwise
    generated speaker segment is absorbed into the
    surrounding generated content instead of starting a
    new segment that would be rated as "mostly
    passthrough" and dropped.

    Two new gates can refuse a merge inside the window
    (see ``docs/.../heygen_like_lipsync_segmentation_td.md``
    §5.1 and §5.5):

    - ``hard_cut_enabled`` + ``hard_cut_threshold``: refuse
      the merge if any adjacent-frame pair inside the
      passthrough gap has a face-crop histogram distance
      above the threshold (i.e. the camera/character
      jumped).  Catches cross-shot / cross-character cuts
      that a short passthrough gap would otherwise bridge.
    - ``track_aware``: a pre-pass walks the target sequence
      once, assigns a per-frame ``track_id`` based on
      face-crop continuity, and Step 2 refuses any merge
      where the boundary track_ids differ.  Falls back to
      the time-window-only rule when track_id is missing.

    Both are opt-in knobs.  When both are off the function
    is exactly the pre-hardcut behavior.  All counters
    fired (hard_cut, speaker_switch, too_short) are
    reported in the returned ``passthrough_reasons`` dict
    so the [LipSync-report] log line shows why a segment
    was downgraded.

    After the majority vote, segments whose total
    duration is below ``min_merged_seconds`` are forced
    entirely to passthrough (see §5.7) to avoid splice
    artifacts on short isolated runs.

    Returns ``(rewritten_count, passthrough_reasons)``.
    ``rewritten_count`` is the count of frames rewritten
    this way; ``passthrough_reasons`` records how many
    times each of the new gates fired.
    """
    empty_reasons: Dict[str, int] = {
        "hard_cut": 0,
        "speaker_switch": 0,
        "too_short": 0,
    }
    if not provenance or not targets:
        return 0, empty_reasons

    # Step 1: collect raw contiguous valid segments.
    n = len(provenance)
    raw_segments: List[Tuple[int, int]] = []  # (start, end_exclusive)
    index = 0
    while index < n:
        source_index = min(index, len(targets) - 1)
        if targets[source_index].get("bbox") is None:
            index += 1
            continue
        seg_start = index
        while index < n:
            source_index = min(index, len(targets) - 1)
            if targets[source_index].get("bbox") is None:
                break
            index += 1
        raw_segments.append((seg_start, index))

    if not raw_segments:
        return 0, empty_reasons

    # Step 1.5: pre-pass to assign a per-frame track_id and
    # detect hard cuts in the passthrough gaps.  Done before
    # Step 2 so the merge logic can consult both signals.
    # Skipped entirely when both new gates are off --
    # preserves the pre-hardcut time-window behavior.
    #
    # Compare each new valid frame against the LAST valid
    # frame regardless of how long the passthrough gap was
    # (the 3-frame constraint from the mouth-diff filter
    # would silently miss cuts separated by > 3 frames of
    # passthrough -- e.g. a 5-frame gap between two
    # speakers).  The conservative 0.65 threshold prevents
    # the natural slow drift of a single speaker's lighting
    # / pose from firing; the reset-after-fire logic below
    # stops the new track's first frame from being compared
    # against the old track's last frame and re-firing.
    if track_aware or hard_cut_enabled:
        current_track = 0
        last_valid_index: Optional[int] = None
        last_valid_crop: Optional[np.ndarray] = None
        for fi in range(n):
            source_index = min(fi, len(targets) - 1)
            bbox = targets[source_index].get("bbox")
            if bbox is None or source_index >= len(frames):
                continue
            x1, y1, x2, y2 = bbox
            frame = frames[source_index]
            crop = frame[
                max(0, y1):min(frame.shape[0], y2),
                max(0, x1):min(frame.shape[1], x2),
            ]
            if crop.size == 0:
                last_valid_crop = None
                last_valid_index = None
                continue
            if (
                last_valid_crop is not None
                and last_valid_index is not None
                and hard_cut_threshold > 0.0
            ):
                dist = face_crop_histogram_distance(crop, last_valid_crop)
                if dist > hard_cut_threshold:
                    current_track += 1
                    # Reset the reference crop so the new
                    # track starts cleanly; otherwise the
                    # first frame of the new track would
                    # be compared against the last frame
                    # of the old track and falsely fire
                    # again on the *next* frame.
                    last_valid_crop = None
                    last_valid_index = None
                    targets[source_index] = {
                        **targets[source_index],
                        "track_id": current_track,
                    }
                    continue
            targets[source_index] = {
                **targets[source_index],
                "track_id": current_track,
            }
            last_valid_crop = crop
            last_valid_index = fi

    # Step 2: merge segments that are within merge_window_seconds.
    # When hard_cut_enabled or track_aware fire, the merge is
    # refused and the segment is left as a new run.
    reasons = dict(empty_reasons)
    if merge_window_seconds > 0.0 and source_fps > 0.0:
        merge_window_frames = max(1, int(round(merge_window_seconds * source_fps)))
        merged: List[Tuple[int, int]] = [raw_segments[0]]
        for seg_start, seg_end in raw_segments[1:]:
            prev_start, prev_end = merged[-1]
            gap = seg_start - prev_end
            if gap > merge_window_frames:
                merged.append((seg_start, seg_end))
                continue
            # Candidate merge inside the time window --
            # now check the new gates.
            refused = False
            if hard_cut_enabled and hard_cut_threshold > 0.0 and frames is not None:
                # Adjacent-pair check across the gap.  Includes
                # the boundary pair (seg_start-1, seg_start) so
                # the cut into the next segment's first frame
                # is tested too.
                for gi in range(prev_end, seg_start):
                    next_i = gi + 1
                    if next_i >= len(frames):
                        break
                    crop_a = frames[gi]
                    crop_b = frames[next_i]
                    if crop_a is None or crop_b is None:
                        continue
                    dist = face_crop_histogram_distance(crop_a, crop_b)
                    if dist > hard_cut_threshold:
                        reasons["hard_cut"] += 1
                        refused = True
                        break
            if not refused and track_aware:
                prev_track = last_track_id(targets, prev_end - 1, prev_start - 1)
                next_track = first_track_id(targets, seg_start, seg_end)
                if (
                    prev_track is not None
                    and next_track is not None
                    and prev_track != next_track
                ):
                    reasons["speaker_switch"] += 1
                    refused = True
            if refused:
                merged.append((seg_start, seg_end))
            else:
                # Bridge the gap: extend the previous segment
                # to cover the gap and the new segment.  The
                # frames in the gap had bbox=None so they
                # were passthrough -- they'll be counted in
                # the merged-segment stats below.
                merged[-1] = (prev_start, seg_end)
    else:
        merged = raw_segments

    # Step 3: majority vote + min-duration downgrade per
    # merged segment.  Both rules are independent and
    # either can force the whole segment to passthrough.
    rewritten = 0
    for seg_start, seg_end in merged:
        segment_provenance = provenance[seg_start:seg_end]
        segment_length = len(segment_provenance)
        if segment_length == 0:
            continue
        generated_count = sum(
            1 for status in segment_provenance if status == "generated"
        )
        passthrough_count = segment_length - generated_count
        # Force all to passthrough only when passthrough is
        # strictly the majority.  Ties go to "leave alone"
        # so the user keeps whatever synthesized frames
        # they already have.
        majority_downgrade = (
            passthrough_count > segment_length / 2
            and passthrough_count / segment_length > passthrough_ratio
        )
        # Independent post-merge duration gate (see §5.7).
        # Short isolated segments are downgraded to avoid
        # splice artifacts at the model/output boundary.
        short_downgrade = False
        if min_merged_seconds > 0.0 and source_fps > 0.0:
            merged_duration = (seg_end - seg_start) / source_fps
            if merged_duration < min_merged_seconds:
                short_downgrade = True
        if majority_downgrade or short_downgrade:
            if short_downgrade and not majority_downgrade:
                reasons["too_short"] += segment_length
            for frame_index in range(seg_start, seg_end):
                if provenance[frame_index] != "generated":
                    continue
                source_index = min(frame_index, len(targets) - 1)
                source_index = min(source_index, len(frames) - 1)
                write_frame(frame_index, frames[source_index])
                provenance[frame_index] = "passthrough"
                rewritten += 1
    return rewritten, reasons
