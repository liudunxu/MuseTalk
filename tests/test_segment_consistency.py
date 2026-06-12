"""Unit tests for ``enforce_segment_consistency`` and the
``face_crop_histogram_distance`` helper in
``musetalk.utils.segment_consistency``.

These are the two most-tweaked primitives behind the
"send a frame to MuseTalk or passthrough it" decision (see
``api.py`` defaults history -- strict all-or-nothing, then
majority vote, then time-window merge, then hard cut +
track-aware + min-merged).  Extracting them as unit-testable
pure functions lets the defaults be tuned without breaking
the visible behaviour.

Scope:
  * All tests run on CPU with synthetic numpy frames -- no
    model load, no GPU, no face detector, no audio.  The
    segment-consistency function only looks at per-frame
    bbox presence, the pre-pass writes ``track_id`` into
    the targets dict in place, and the hard-cut detector
    compares whole-frame (or face-crop) histograms.
  * Frame fixtures use deterministic colour-block faces
    (skin + hair rectangles) so the histogram distance is
    reproducible across runs without any random seed
    dependency.

Run with::

    python -m unittest discover tests -v
"""

from __future__ import annotations

import os
import sys
import unittest
from typing import List, Optional, Tuple

import numpy as np

# Make the project root importable so the ``musetalk.*`` package
# resolves.  Matches the pattern used by LatentSync's tests/.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from musetalk.utils.segment_consistency import (  # noqa: E402
    enforce_segment_consistency,
    face_crop_histogram_distance,
)


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------


def make_face_frame(
    seed: int,
    skin_bgr: Tuple[int, int, int] = (220, 180, 150),
    hair_bgr: Tuple[int, int, int] = (30, 20, 20),
    height: int = 200,
    width: int = 200,
) -> np.ndarray:
    """Build a deterministic BGR face-crop-style frame.

    Top 30% is hair colour, the rest is skin.  Add a tiny
    seed-driven perturbation so two frames of the "same
    person" are not bit-identical (which would make
    histogram distance identically zero and defeat the
    test of a non-trivial threshold).
    """
    rng = np.random.default_rng(seed)
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[0 : int(height * 0.3), :] = hair_bgr
    frame[int(height * 0.3) :, :] = skin_bgr
    frame = frame.astype(np.int16)
    frame += rng.integers(-8, 8, frame.shape, dtype=np.int16)
    return np.clip(frame, 0, 255).astype(np.uint8)


def make_target(
    bbox: Optional[Tuple[int, int, int, int]] = (50, 50, 150, 150),
    track_id: Optional[int] = None,
) -> dict:
    """Build a single ``targets[i]`` dict with the fields
    that ``_enforce_segment_consistency`` actually reads."""
    target: dict = {}
    if bbox is not None:
        target["bbox"] = bbox
    if track_id is not None:
        target["track_id"] = track_id
    return target


def build_scene(
    n_frames: int,
    person: str = "A",
    passthrough_gaps: Optional[List[Tuple[int, int]]] = None,
    hard_cuts_at: Optional[List[int]] = None,
    track_ids: Optional[List[Optional[int]]] = None,
) -> Tuple[List[dict], List[np.ndarray]]:
    """Build a synthetic (targets, frames) sequence.

    Args:
        n_frames: total frames.
        person: ``"A"`` (light skin / dark hair) or
            ``"B"`` (dark skin / light hair).
        passthrough_gaps: list of ``(start_inclusive,
            end_exclusive)`` frame indices where the target
            has ``bbox=None`` (simulating a detection miss).
        hard_cuts_at: frame indices where the underlying
            "person" switches from A to B (or vice versa);
            used by the hard-cut detector.  Each entry
            creates a discontinuity in the generated frame
            stream starting at that index.
        track_ids: explicit per-frame ``track_id`` override;
            when provided, used to short-circuit the
            pre-pass.  When ``None``, lets the pre-pass
            assign track_ids based on hard-cut detection.
    """
    passthrough_set = set()
    if passthrough_gaps:
        for s, e in passthrough_gaps:
            passthrough_set.update(range(s, e))

    person_a = ((220, 180, 150), (30, 20, 20))
    person_b = ((110, 80, 60), (200, 180, 120))

    hard_cuts_at = hard_cuts_at or []
    current_person = "A" if "A" in person else "B"

    targets: List[dict] = []
    frames: List[np.ndarray] = []
    for i in range(n_frames):
        # Apply any hard cut that fires at this index
        if i in hard_cuts_at:
            current_person = "B" if current_person == "A" else "A"

        skin, hair = person_a if current_person == "A" else person_b
        frames.append(make_face_frame(seed=i, skin_bgr=skin, hair_bgr=hair))

        if i in passthrough_set:
            targets.append(make_target(bbox=None))
        else:
            tid = track_ids[i] if track_ids and i < len(track_ids) else None
            targets.append(make_target(bbox=(50, 50, 150, 150), track_id=tid))

    return targets, frames


def run_consistency(
    provenance: List[str],
    targets: List[dict],
    frames: List[np.ndarray],
    *,
    passthrough_ratio: float = 0.5,
    merge_window_seconds: float = 1.0,
    source_fps: float = 25.0,
    hard_cut_enabled: bool = True,
    hard_cut_threshold: float = 0.65,
    track_aware: bool = True,
    min_merged_seconds: float = 1.5,
) -> Tuple[int, dict, List[str], List[dict]]:
    """Run ``_enforce_segment_consistency`` and capture
    both the return tuple and the post-call side effects
    (provenance rewrite, targets track_id assignment)."""
    write_calls: List[Tuple[int, np.ndarray]] = []

    def write_frame(idx: int, frame: np.ndarray) -> None:
        write_calls.append((idx, frame))

    rewritten, reasons = enforce_segment_consistency(
        provenance,
        targets,
        frames,
        write_frame,
        passthrough_ratio=passthrough_ratio,
        merge_window_seconds=merge_window_seconds,
        source_fps=source_fps,
        hard_cut_enabled=hard_cut_enabled,
        hard_cut_threshold=hard_cut_threshold,
        track_aware=track_aware,
        min_merged_seconds=min_merged_seconds,
    )
    return rewritten, reasons, list(provenance), list(targets), write_calls


# ---------------------------------------------------------------------------
# Sanity / empty-inputs
# ---------------------------------------------------------------------------


class TestEmptyInputs(unittest.TestCase):
    def test_empty_provenance_returns_zero_and_empty_reasons(self) -> None:
        rewritten, reasons, *_ = run_consistency(
            provenance=[],
            targets=[],
            frames=[],
        )
        self.assertEqual(rewritten, 0)
        self.assertEqual(reasons, {"hard_cut": 0, "speaker_switch": 0, "too_short": 0})

    def test_no_valid_frames_returns_unchanged(self) -> None:
        # All targets have bbox=None, all provenance is passthrough
        targets = [make_target(bbox=None) for _ in range(10)]
        provenance = ["passthrough"] * 10
        frames = [make_face_frame(seed=i) for i in range(10)]
        rewritten, reasons, prov_after, _, _ = run_consistency(provenance, targets, frames)
        self.assertEqual(rewritten, 0)
        self.assertEqual(reasons["hard_cut"], 0)
        self.assertEqual(reasons["speaker_switch"], 0)
        self.assertEqual(reasons["too_short"], 0)
        # Provenance unchanged because there's nothing to merge
        self.assertEqual(prov_after, ["passthrough"] * 10)


# ---------------------------------------------------------------------------
# Fixture §8.4: same-person merge / hard cut / speaker switch / short / opt-out
# ---------------------------------------------------------------------------


class TestSamePersonMergesAcrossGap(unittest.TestCase):
    """Doc 8.4 fixture 1: same person 5s clip, 2s silence in the
    middle, all bbox stable -> should merge into one segment,
    no downgrade if merged length >= min_merged_seconds."""

    def test_long_merge_no_downgrade(self) -> None:
        # 25fps x 5s = 125 frames total.  Two visible runs
        # separated by a 20-frame passthrough gap (within the
        # default 1.0s = 25-frame merge window).  Each run is
        # 52-53 frames = 2.1s, well above min_merged_seconds=1.5s.
        # Merged: 105 frames = 4.2s -> no downgrade.
        n = 125
        targets, frames = build_scene(
            n_frames=n,
            person="A",
            passthrough_gaps=[(50, 70)],
        )
        provenance = ["passthrough"] * n
        for i in list(range(0, 50)) + list(range(70, 125)):
            provenance[i] = "generated"

        rewritten, reasons, prov_after, _, _ = run_consistency(provenance, targets, frames)
        self.assertEqual(rewritten, 0, "no downgrade on a 4.2s merged segment")
        self.assertEqual(reasons["hard_cut"], 0)
        self.assertEqual(reasons["speaker_switch"], 0)
        self.assertEqual(reasons["too_short"], 0)
        # All 50 + 55 = 105 generated frames preserved
        self.assertEqual(prov_after.count("generated"), 105)


class TestHardCutRefusesMerge(unittest.TestCase):
    """Doc 8.4 fixture 3: hard cut inside the merge window."""

    def test_hard_cut_blocks_merge_and_count_is_one(self) -> None:
        # 25fps x 4.4s = 110 frames.  Visible 0-58 (59 frames = 2.36s,
        # person A), gap 59-69 (11 frames = 0.44s, within the
        # 1.0s merge window), visible 70-109 (40 frames = 1.6s,
        # person B).  Each segment is >= 1.5s so too_short does
        # NOT fire (the test asserts hard_cut in isolation, not
        # too_short).
        n = 110
        targets, frames = build_scene(
            n_frames=n,
            person="A",
            passthrough_gaps=[(59, 69)],
            hard_cuts_at=[60],  # A->B switch
        )
        provenance = ["passthrough"] * n
        for i in list(range(0, 59)) + list(range(70, 110)):
            provenance[i] = "generated"

        rewritten, reasons, _, _, _ = run_consistency(provenance, targets, frames)
        # First-fail-wins: hard cut fires before track_id check
        self.assertGreaterEqual(reasons["hard_cut"], 1)
        # Each segment is >= 1.5s individually -> too_short stays at 0
        self.assertEqual(reasons["too_short"], 0)


class TestSpeakerSwitchRefusesMerge(unittest.TestCase):
    """Doc 8.4 fixture 2: cross-character switch with
    hard-cut detector disabled to isolate the track_aware path."""

    def test_track_switch_blocks_merge(self) -> None:
        # The hard-cut detector and the track-id pre-pass share
        # the same ``hard_cut_threshold`` knob -- disabling the
        # former also disables the latter.  So the only way to
        # exercise the track_aware gate in isolation is to set
        # up a real visual cut that the pre-pass WILL detect.
        # The pre-pass then assigns different track_ids; the
        # Step 2 track_aware check sees them and refuses the
        # merge.  ``hard_cut_enabled=False`` in Step 2 then
        # makes hard_cut stay 0 (the gap is short enough that
        # Step 2's hard-cut check doesn't fire either), so
        # only ``speaker_switch`` counts.
        n = 100
        targets, frames = build_scene(
            n_frames=n,
            person="A",
            passthrough_gaps=[(59, 65)],
            hard_cuts_at=[60],  # visual A->B switch (pre-pass will see this)
        )
        provenance = ["passthrough"] * n
        for i in list(range(0, 59)) + list(range(65, 100)):
            provenance[i] = "generated"

        _, reasons, _, _, _ = run_consistency(
            provenance, targets, frames,
            hard_cut_enabled=False,  # disable Step 2 hard-cut check
        )
        # Step 2 didn't run the hard-cut check, so it stays 0
        self.assertEqual(reasons["hard_cut"], 0)
        # But the pre-pass still ran (track_aware=True) and
        # assigned different track_ids across the cut, so
        # Step 2's track_aware check fired at least once.
        self.assertGreaterEqual(reasons["speaker_switch"], 1)


class TestShortSegmentDowngrade(unittest.TestCase):
    """Doc 8.4 fixture 6: 0.6s isolated segment -> downgraded."""

    def test_under_min_merged_seconds_gets_fully_passthrough(self) -> None:
        # 25fps x 0.6s = 15 frames, all valid, all generated
        n = 15
        targets, frames = build_scene(n_frames=n, person="A")
        provenance = ["generated"] * n

        rewritten, reasons, prov_after, _, _ = run_consistency(provenance, targets, frames)
        self.assertEqual(rewritten, n, "all generated frames downgraded")
        self.assertEqual(prov_after, ["passthrough"] * n)
        self.assertEqual(reasons["too_short"], n)

    def test_just_over_min_merged_seconds_preserved(self) -> None:
        # 25fps x 1.6s = 40 frames, all valid, all generated
        n = 40
        targets, frames = build_scene(n_frames=n, person="A")
        provenance = ["generated"] * n

        rewritten, reasons, prov_after, _, _ = run_consistency(provenance, targets, frames)
        self.assertEqual(rewritten, 0)
        self.assertEqual(reasons["too_short"], 0)
        self.assertEqual(prov_after, ["generated"] * n)


class TestAllOptOutMatchesLegacyBehavior(unittest.TestCase):
    """Doc 8.4 fixture 10: all three new gates off -> legacy
    (pre-phase-1) behavior.  With passthrough < 50% the
    majority vote must NOT force a downgrade."""

    def test_mixed_segment_not_downgraded_when_minority(self) -> None:
        # 4 generated + 1 passthrough interleaved.  passthrough
        # count / segment length = 1/5 = 0.2, well under 0.5.
        n = 5
        targets, frames = build_scene(
            n_frames=n,
            person="A",
            passthrough_gaps=[(2, 3)],  # one frame passthrough
        )
        provenance = ["passthrough"] * n
        for i in [0, 1, 3, 4]:
            provenance[i] = "generated"

        rewritten, reasons, _, _, _ = run_consistency(
            provenance, targets, frames,
            hard_cut_enabled=False,
            track_aware=False,
            min_merged_seconds=0.0,  # opt out of too_short
        )
        # 3 new keys all 0 -> legacy behavior
        self.assertEqual(reasons["hard_cut"], 0)
        self.assertEqual(reasons["speaker_switch"], 0)
        self.assertEqual(reasons["too_short"], 0)
        # The 1-frame passthrough minority does not trigger majority vote
        self.assertEqual(rewritten, 0)


# ---------------------------------------------------------------------------
# Counter correctness (the new diagnostic counters from ce7b684)
# ---------------------------------------------------------------------------


class TestDiagnosticCounters(unittest.TestCase):
    """``ema_chain_breaks`` and ``ema_resets_on_track_switch``
    live in the CodeFormer block, not in segment consistency;
    this class only verifies the segment-consistency side."""

    def test_passthrough_reasons_keys_present_in_return(self) -> None:
        n = 10
        targets, frames = build_scene(n_frames=n, person="A")
        provenance = ["generated"] * n
        _, reasons, _, _, _ = run_consistency(provenance, targets, frames)
        self.assertIn("hard_cut", reasons)
        self.assertIn("speaker_switch", reasons)
        self.assertIn("too_short", reasons)


# ---------------------------------------------------------------------------
# Face-crop histogram distance (the actual hard-cut signal)
# ---------------------------------------------------------------------------


class TestFaceCropHistogramDistance(unittest.TestCase):
    """``_face_crop_histogram_distance`` is the cheap signal
    behind hard-cut detection.  Its direction must match the
    threshold expectation (0 = identical, 1 = unrelated)."""

    def test_identical_returns_zero(self) -> None:
        a = make_face_frame(seed=0)
        d = face_crop_histogram_distance(a, a)
        self.assertLess(d, 0.05)

    def test_unrelated_returns_near_one(self) -> None:
        # Person A vs Person B (very different BGR distributions)
        person_a = make_face_frame(seed=0, skin_bgr=(220, 180, 150), hair_bgr=(30, 20, 20))
        person_b = make_face_frame(seed=0, skin_bgr=(110, 80, 60), hair_bgr=(200, 180, 120))
        d = face_crop_histogram_distance(person_a, person_b)
        self.assertGreater(d, 0.9)

    def test_slight_perturbation_stays_low(self) -> None:
        a = make_face_frame(seed=0)
        b = make_face_frame(seed=1)  # same person, different seed
        d = face_crop_histogram_distance(a, b)
        # Same colour distribution + tiny pixel jitter should
        # remain well below the 0.65 hard-cut threshold.
        self.assertLess(d, 0.3)

    def test_empty_input_returns_zero(self) -> None:
        a = make_face_frame(seed=0)
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        self.assertEqual(face_crop_histogram_distance(a, empty), 0.0)
        self.assertEqual(face_crop_histogram_distance(empty, a), 0.0)


# ---------------------------------------------------------------------------
# Provenance-mutation semantics
# ---------------------------------------------------------------------------


class TestProvenanceMutation(unittest.TestCase):
    def test_passthrough_frames_are_not_rewritten(self) -> None:
        # 5 generated + 5 passthrough, all in one segment.
        # The 5 passthrough frames are already passthrough
        # so write_frame should not be called for them.
        n = 10
        targets, frames = build_scene(
            n_frames=n,
            person="A",
            passthrough_gaps=[(5, 10)],
        )
        provenance: List[str] = []
        for i in range(n):
            provenance.append("passthrough" if i >= 5 else "generated")

        _, _, _, _, write_calls = run_consistency(provenance, targets, frames)
        rewritten_indices = [idx for idx, _ in write_calls]
        # None of the passthrough frames should have been re-written
        for idx in rewritten_indices:
            self.assertGreaterEqual(idx, 0)
            self.assertLess(idx, 5, f"unexpected rewrite of passthrough frame {idx}")

    def test_short_segment_rewrites_all_generated(self) -> None:
        n = 8
        targets, frames = build_scene(n_frames=n, person="A")
        provenance = ["generated"] * n
        _, reasons, _, _, write_calls = run_consistency(provenance, targets, frames)
        # All 8 generated frames should be rewritten to passthrough
        self.assertEqual(len(write_calls), n)
        self.assertEqual(reasons["too_short"], n)


if __name__ == "__main__":
    unittest.main()
