# Evaluation

All numbers come from `python -m sightline.cli evaluate` on sets that were generated after the
thresholds were frozen. Development used other seeds (synthetic 101-124, real 201-203 and 301-303);
the test seeds below were not looked at before this run.

A labelled defect counts as found when a `defect` or `needs_review` finding of the same kind overlaps
it (one frame of slack). Every other flagged finding is a false positive, including reviews.

## Synthetic test set (seeds 1001-1024)

24 clips, 5349 frames at 640x360. Each clip mixes material that must not be flagged
(intended cuts, a held title card, slow motion, a fade to black) with 3-5 injected defects.
False positives: 0.0 per 1,000 frames. First pass 9.23 ms/frame;
8.9 follow-up checks per clip on average.

| Defect | Labels | Found (defect or review) | Confirmed as defect | False positives | Recall | Precision |
|---|---|---|---|---|---|---|
| black_frame | 20 | 20 | 20 | 0 | 1.0 | 1.0 |
| frozen_run | 19 | 19 | 19 | 0 | 1.0 | 1.0 |
| flash_frame | 15 | 15 | 15 | 0 | 1.0 | 1.0 |
| damaged_frame | 20 | 19 | 19 | 0 | 0.95 | 1.0 |
| frame_skip | 19 | 15 | 12 | 0 | 0.789 | 1.0 |

## Real footage test set (seeds 2001-2003, 3001-3003)

Defects seeded into two openly licensed trailers (Sintel, Big Buck Bunny; Blender Foundation, CC BY),
re-encoded as 640-pixel MP4 like a delivery file. Positions avoid the trailers' own cuts and fades.
6 clips, 6184 frames. False positives: 0.97 per 1,000 frames.

| Defect | Labels | Found (defect or review) | Confirmed as defect | False positives | Recall | Precision |
|---|---|---|---|---|---|---|
| black_frame | 5 | 5 | 5 | 4 | 1.0 | 0.556 |
| frozen_run | 3 | 2 | 2 | 0 | 0.667 | 1.0 |
| flash_frame | 4 | 3 | 3 | 0 | 0.75 | 1.0 |
| damaged_frame | 3 | 3 | 3 | 0 | 1.0 | 1.0 |
| frame_skip | 3 | 1 | 0 | 2 | 0.333 | 0.333 |

## Clean real footage

The unmodified 1080p trailers (1,253 and 812 frames) produce no `defect` findings. Big Buck Bunny has one
`needs_review` finding: two black frames between two different shots at frames 472-473, which the agent
cannot tell apart from a timeline gap, so it asks a person.

Scored with the same evaluator and empty label files (every flagged finding counts as a false
positive): `results/eval_clean_trailers.json`, 0.48 flagged items per 1,000 frames.

```sh
.venv/bin/python -m sightline.cli evaluate data/real/clean --out results/eval_clean_trailers.json
```

`data/real/clean/` links the two downloaded trailers as `clip_*.mp4|mov` next to
`clip_*.labels.json` files of the form `{"frames": <count>, "labels": []}`.

## Known weaknesses

* Frame skips are the weakest class on real footage (1 of 3 found). Skips are judged on whole-frame
  motion so that a fast local subject (a dragon's wings) does not count, which also misses skips in
  fast camera moves. Two false skip alarms remain in the dragon-flight shot of the Sintel trailer.
* A flash frame whose inserted picture is dark is reported as a black frame (the right range, the
  wrong kind).
* Dense optical flow (Farneback) is unstable on flat or saturated regions; statistics drop the largest
  2% of vectors, and stalls are decided from pixel changes rather than flow.
