# FrameGuard Sightline

Sightline checks rendered or exported video for the defects that slip through review: black
frames, frozen runs, one- or two-frame flashes of the wrong shot, block-damaged frames and
dropped frames. OpenCV 5 measures every frame, an agent re-checks each suspicion with the tool
that fits it, and a person approves what happens to the footage. Approved fixes are applied to a
preview copy and checked again.

It is built for small studios, animation teams, independent producers and education or
nonprofit media labs that need repeatable quality control without sending footage to a person
for every pass.

**Demo video (4:51):** https://youtu.be/UEEKB2skVho

**Live demo:** https://99m6iyqdka.execute-api.ap-southeast-2.amazonaws.com (AWS, ap-southeast-2).
Upload a short MP4 or MOV clip (up to 80 MB and 3,000 frames). The page accepts at most 10 jobs per
UTC day; footage and reports are deleted after 7 days.

## How a job runs

1. **Scan.** Every frame is measured once at 320 px width: luma mean and spread, a 64-bit DCT
   perceptual hash, an HSV histogram, Canny edge density, 8x8 block-grid step and ratio, dense
   Farneback optical flow and the fraction of changed pixels against the previous frame.
2. **Suspect.** Detectors turn those series into candidates (dark runs, still runs, histogram
   jumps, block-grid spikes, motion jumps). A candidate is a suspicion with a margin, not a verdict.
3. **Check.** The agent picks a follow-up check per candidate and re-decodes only the frames it
   needs. The measurement decides the next step:
   * a dark run: compare the frames on both sides; same shot means a defect, a gradual decline is a
     fade, a long black between different shots is an editorial cut to black, a short one goes to
     a person;
   * a shot change: does the picture return to the previous shot within two frames (colour *and*
     layout)? Then it was a flash, and the matching cut back is folded into the same finding;
   * a still run: are the pixels really identical at full resolution, does it start at an edit,
     does the same shot resume and move afterwards? Only a stall inside a moving shot is a defect;
   * a block spike: do 16x16 blocks change in these frames only, while clean neighbours agree?
     Neighbours that are themselves suspect are skipped;
   * a motion jump: is it explained by an adjacent confirmed defect, and does the whole frame move
     the same way the shot was moving?
4. **Decide.** Each finding ends as `defect`, `dismissed` (intended material, with the reason) or
   `needs_review`. Ambiguous results and anything left when the follow-up budget runs out go to a
   person. Every check is in the trace with the measurement that triggered it and its result.
5. **Act.** Nothing changes until a person approves. Black, flash and damaged frames are patched
   in a preview by holding the nearest good frame of the same shot. Frozen runs and skips go to a
   re-render list, because an export cannot recover lost frames.
6. **Re-check.** The agent analyses the repair preview again and reports whether every repaired
   range is now clean.

## Results

Full tables and protocol: [docs/EVALUATION.md](docs/EVALUATION.md). Thresholds were frozen on
development seeds before these test sets were generated.

| Test set | Frames | Defects found | False positives per 1,000 frames |
|---|---|---|---|
| Synthetic, 24 clips (seeds 1001-1024) | 5,349 | 88 of 93 | 0.0 |
| Real trailers with seeded defects, 6 clips | 6,184 | 14 of 18 | 0.97 |
| Real trailers, unmodified (2 clips) | 2,065 | no `defect` findings; 1 item for review | - |

Frame skips are the weakest class on real footage (1 of 3). First-pass cost is about 6-9 ms per
frame on an Apple M4 (single process).

## Run it locally

```sh
python3.13 -m venv .venv
.venv/bin/pip install -e ".[dev]"
.venv/bin/python -m pytest
.venv/bin/python -m sightline.cli analyze path/to/clip.mp4 --out jobs/demo
.venv/bin/python -m sightline.cli approve jobs/demo flash_frame:39-40 --by editor
.venv/bin/python -m sightline.cli act jobs/demo path/to/clip.mp4
.venv/bin/python -m sightline.web --port 8765   # the review page with a local backend
```

Rebuild the evaluation sets and scores:

```sh
.venv/bin/python -m sightline.cli synth data/synthetic-test --seeds 1001-1024
.venv/bin/python -m sightline.cli evaluate data/synthetic-test --out results/eval_synthetic_test.json
```

The real-footage set is built by `sightline/realseed.py` from the official Blender Foundation
trailers (`sintel_trailer-1080p.mp4`, `trailer_1080p.mov` from download.blender.org); the
trailers themselves are not redistributed here.

## AWS deployment

`aws/template.yaml` (CloudFormation) creates:

* an S3 bucket for uploads and reports (private, encrypted, everything expires after 7 days);
* a DynamoDB table for job state and decisions;
* an arm64 (Graviton) Lambda with OpenCV 5 that runs the scan, the agent loop and the actions;
* a Step Functions workflow: analyze, then wait for a person with a task token, then act and re-check;
* an HTTP API and a small Lambda that serves the review page, issues presigned upload forms and
  resumes the workflow when the person sends decisions;
* CloudWatch logs and Embedded Metric Format metrics (frames, ms/frame, defects, reviews, checks),
  without image content.

Every role in the stack carries the `sightline-runtime-boundary` permissions boundary, so a role can
never do more than the job needs: read and write its own bucket and table, write its own logs,
start and resume its own workflow, invoke its own two functions. `aws/iam/` holds that boundary and
the two policies for the deploying user, who can create roles only with the boundary attached and
cannot remove it, edit it or change its own permissions:

```sh
python aws/iam/render.py --account <12-digit id> --region <region> --out aws/build/iam
```

An administrator creates the boundary policy once under that exact name and attaches the two
deployer policies to the user that runs `aws/deploy.sh`.

```sh
aws/deploy.sh     # credentials from an env file or the AWS CLI; builds the arm64 packages, deploys
aws/teardown.sh   # deletes job data, the stack and the artifact bucket
```

The stack is sized for a free-plan account: 80 MB uploads, 3,000 frames per job, API throttling at
5 requests per second, 7-day retention, and at most 10 jobs per UTC day (`DailyJobLimit`). The daily
limit bounds how many analyses a public page can start; it is not a cap on total spend.

## Limits and responsible use

* Thresholds are tuned for delivery files and renders, not for surveillance or people analysis.
  The tool inspects picture integrity only; it does not identify anyone.
* A `dismissed` finding is an explanation, not a guarantee. The review page lists every dismissed
  item with its reason so a person can overrule it.
* Repairs are previews built from neighbouring frames. They are meant to confirm the diagnosis and
  speed up a re-export, not to replace a re-render of the source project.
* Uploaded footage is deleted after 7 days; logs hold numbers only.

## Credits

Sintel and Big Buck Bunny trailers: (c) Blender Foundation, CC BY 3.0, used for evaluation only.
