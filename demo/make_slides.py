"""Render the demo video's slides (1920x960 PNG) from HTML with Playwright.

    python demo/make_slides.py --clip data/demo/sintel_test_2003.mp4 --job demo/build/app/job.json

Frames on the slides are read from the demo clip with OpenCV; the numbers come from
results/*.json and from the recorded job, so every figure on screen has a source in the repo.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import cv2
from playwright.async_api import async_playwright

ROOT = Path(__file__).resolve().parent.parent
STRIP = range(833, 840)  # around the seeded flash frame of the demo clip
WORDS = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
]
SHORT = {
    "black_frame": "black",
    "frozen_run": "held",
    "flash_frame": "other shot",
    "damaged_frame": "damaged",
    "frame_skip": "jump",
}
KINDS = ["black_frame", "frozen_run", "flash_frame", "damaged_frame", "frame_skip"]
NAMES = {
    "black_frame": "Black frame",
    "frozen_run": "Frozen run",
    "flash_frame": "Flash frame",
    "damaged_frame": "Damaged frame",
    "frame_skip": "Frame skip",
}

CSS = """
:root { --ink:#1d2126; --muted:#5f6873; --line:#d9dde2; --page:#f3f4f6; --accent:#1f4e8c;
  --defect:#b42318; --defect-bg:#fdecea; --review:#9a5b00; --review-bg:#fff4e0; --ok:#1f7a3e; --ok-bg:#e7f5ec;
  --mono: "SF Mono", ui-monospace, Menlo, monospace; }
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; color: var(--ink);
  background: #888; -webkit-font-smoothing: antialiased; }
.slide { width: 1920px; height: 960px; background: #fff; padding: 84px 108px; position: relative; overflow: hidden;
  display: flex; flex-direction: column; margin-bottom: 20px; }
.kicker { font-size: 24px; color: var(--muted); letter-spacing: .02em; }
h1 { font-size: 58px; line-height: 1.1; margin: 10px 0 0; font-weight: 700; letter-spacing: -.01em; }
.sub { font-size: 30px; color: var(--muted); margin-top: 14px; }
.mono { font-family: var(--mono); }
figure { display: flex; flex-direction: column; }
figure img { display: block; width: 100%; border-radius: 6px; }
figcaption { font: 21px/1.4 var(--mono); color: var(--muted); margin-top: 6px; }
/* title */
#title h1 { font-size: 104px; margin-top: 26px; }
#title .sub { font-size: 40px; margin-top: 22px; }
#title .team { font-size: 30px; margin-top: 30px; }
.strip { display: flex; gap: 14px; margin-top: auto; }
.strip figure { width: 228px; }
.strip figure.hit img { outline: 6px solid var(--defect); outline-offset: -1px; }
.strip figure.hit figcaption { color: var(--defect); font-weight: 600; }
/* problem */
.cards { display: flex; gap: 34px; margin-top: 54px; }
.card { width: 314px; display: flex; flex-direction: column; gap: 12px; }
.card h2 { font-size: 30px; }
.card p { font-size: 25px; line-height: 1.35; color: var(--muted); margin-top: 4px; }
.card figure.bad img { outline: 5px solid var(--defect); outline-offset: -1px; }
.card figure.bad figcaption { color: var(--defect); }
/* approach */
.flow { display: flex; align-items: stretch; gap: 0; margin-top: 70px; }
.step { width: 240px; border: 2px solid var(--line); border-radius: 12px; padding: 22px 20px; }
.step b { display: block; font: 600 22px var(--mono); color: var(--accent); }
.step h3 { font-size: 32px; margin: 6px 0 10px; }
.step p { font-size: 23px; line-height: 1.35; color: var(--muted); }
.arrow { width: 44px; display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: 34px; }
.gate { width: 112px; display: flex; flex-direction: column; align-items: center; justify-content: center; gap: 8px;
  color: var(--review); font-size: 21px; text-align: center; font-weight: 600; }
.gate span { width: 54px; height: 54px; border-radius: 50%; background: var(--review-bg); border: 2px solid var(--review);
  display: flex; align-items: center; justify-content: center; font-size: 30px; }
.note { font-size: 28px; color: var(--ink); margin-top: auto; }
.trace { margin-top: 56px; border: 2px solid var(--line); border-radius: 12px; padding: 22px 26px; width: 1210px; }
.trace .t-head { font-size: 21px; color: var(--muted); margin-bottom: 8px; }
.trace .t-line { font-size: 26px; line-height: 1.4; }
.trace code { font: 600 24px var(--mono); color: var(--accent); }
.trace ul { margin: 8px 0 0 26px; font-size: 23px; color: var(--muted); line-height: 1.45; }
.trace ul b { color: var(--ink); font-weight: 600; }
.strip-note { font-size: 24px; color: var(--muted); margin: auto 0 14px; }
.strip-note + .strip { margin-top: 0; }
/* architecture */
#architecture { padding: 30px 60px; align-items: center; justify-content: center; }
#architecture img { height: 900px; }
/* agent */
table { border-collapse: collapse; width: 100%; margin-top: 40px; }
th { text-align: left; font-size: 22px; color: var(--muted); font-weight: 600; padding: 0 18px 12px 0; border-bottom: 2px solid var(--line); }
td { font-size: 26px; line-height: 1.35; padding: 16px 18px 16px 0; border-bottom: 1px solid var(--line); vertical-align: top; }
td.k { font-weight: 600; width: 230px; }
.tag { display: inline-block; font-size: 20px; font-weight: 600; padding: 2px 10px; border-radius: 6px; margin: 0 6px 6px 0; }
.t-defect { color: var(--defect); background: var(--defect-bg); }
.t-review { color: var(--review); background: var(--review-bg); }
.t-dismissed { color: var(--muted); background: #eef0f2; }
/* results */
.two { display: flex; gap: 72px; margin-top: 44px; }
.two > div { flex: 1; }
.two h2 { font-size: 32px; }
.two .meta { font-size: 23px; color: var(--muted); margin-top: 4px; }
.two table { margin-top: 18px; }
.two td { font-size: 27px; padding: 11px 18px 11px 0; }
.two td.n { font-family: var(--mono); text-align: right; width: 170px; }
.bar { height: 14px; background: #eef0f2; border-radius: 7px; width: 220px; display: inline-block; vertical-align: middle; }
.bar i { display: block; height: 14px; border-radius: 7px; background: var(--accent); }
.fp td { font-weight: 600; border-bottom: 0; }
.foot { font-size: 25px; color: var(--muted); margin-top: auto; line-height: 1.45; }
/* limits */
.cols { display: flex; gap: 80px; margin-top: 48px; }
.cols > div { flex: 1; }
.cols h2 { font-size: 34px; margin-bottom: 18px; }
.cols li { font-size: 28px; line-height: 1.4; margin: 0 0 18px 30px; }
/* end */
#end { align-items: flex-start; justify-content: center; }
#end h1 { font-size: 92px; }
#end .sub { font-size: 36px; margin-top: 26px; color: var(--ink); }
#end .credits { position: absolute; left: 108px; bottom: 70px; font-size: 22px; color: var(--muted); line-height: 1.6; }
"""


def extract_frames(clip: Path, out: Path, needed: set[int]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(clip))
    index = 0
    while index <= max(needed):
        ok, frame = capture.read()
        if not ok:
            break
        if index in needed:
            cv2.imwrite(str(out / f"{index:05d}.png"), frame)
        index += 1
    capture.release()


def fig(frames: Path, i: int, caption: str, cls: str = "") -> str:
    return f'<figure class="{cls}"><img src="{(frames / f"{i:05d}.png").as_uri()}"><figcaption>{caption}</figcaption></figure>'


def count(n: int, noun: str) -> str:
    word = WORDS[n] if n < len(WORDS) else str(n)
    return f"{word} {noun}{'' if n == 1 else 's'}"


def describe(kind: str, n: int) -> str:
    return {
        "black_frame": f"The picture goes black for {count(n, 'frame')} inside a running shot.",
        "frozen_run": f"The same picture is held for {count(n, 'frame')} where the shot should move.",
        "flash_frame": f"{count(n, 'frame').capitalize()} from another shot cut into this one.",
        "damaged_frame": f"Broken blocks from a bad encode in {count(n, 'frame')}.",
        "frame_skip": "Frames are dropped, so the motion jumps forward.",
    }[kind]


def value(v) -> str:
    """The review page's measurement format: three decimals at most."""
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return str(round(v, 3))
    if isinstance(v, list):
        return ("; " if v and isinstance(v[0], dict) else ", ").join(value(x) for x in v)
    if isinstance(v, dict):
        return ", ".join(f"{k.replace('_', ' ')} {value(x)}" for k, x in v.items())
    return str(v)


def trace_card(job: dict) -> str:
    step = next((t for t in job.get("trace", []) if t["decision"].startswith("defect: flash")), None)
    if step is None:
        return ""
    rows = "".join(f"<li><b>{k.replace('_', ' ')}</b> {value(v)}</li>" for k, v in step["result"].items())
    return (
        f'<div class="trace"><div class="t-head">One check from the demo job, as the review page shows it</div>'
        f'<div class="t-line"><code>{step["tool"]}</code> because {step["trigger"]} &rarr; <i>{step["decision"]}</i></div>'
        f"<ul>{rows}</ul></div>"
    )


def build_html(
    frames: Path, job: dict, labels: list[dict], synth: dict, real: dict, clean: dict, where: str
) -> str:
    flash = next(lab["start"] for lab in labels if lab["kind"] == "flash_frame")
    strip = "".join(
        fig(frames, i, f"{i} flash" if i == flash else str(i), "hit" if i == flash else "") for i in STRIP
    )
    strip_note = f"Frames {STRIP[0]}&ndash;{STRIP[-1]} of a test clip. Frame {flash} comes from another shot."
    by_kind = {lab["kind"]: lab for lab in labels}
    card_html = ""
    for kind in KINDS:
        lab = by_kind[kind]
        a, b = lab["start"], lab["end"]
        span = str(a) if a == b else f"{a}&ndash;{b}"
        card_html += (
            f'<div class="card"><h2>{NAMES[kind]}</h2>{fig(frames, a - 1, f"{a - 1} before")}'
            f"{fig(frames, a, f'{span} {SHORT[kind]}', 'bad')}<p>{describe(kind, b - a + 1)}</p></div>"
        )
    steps = [
        ("1", "Scan", "Every frame is measured once at 320&nbsp;px."),
        ("2", "Suspect", "Detectors flag candidates with a margin."),
        ("3", "Check", "The agent re-decodes a window and measures again."),
        ("4", "Decide", "Defect, dismissed with a reason, or for review."),
        ("5", "Act", "Repair preview or re-render list."),
        ("6", "Re-check", "The agent analyses its own repair."),
    ]
    flow = []
    for n, name, text in steps:
        flow.append(f'<div class="step"><b>{n}</b><h3>{name}</h3><p>{text}</p></div>')
        if n == "4":
            flow.append('<div class="gate"><span>&#10003;</span>a person approves</div>')
        elif n != "6":
            flow.append('<div class="arrow">&rarr;</div>')
    agent_rows = [
        (
            "Dark run",
            "Compare the frames on both sides and the luma trend into the dark run.",
            [
                ("defect", "defect"),
                ("dismissed", "fade"),
                ("dismissed", "cut to black"),
                ("review", "short gap: review"),
            ],
        ),
        (
            "Shot change",
            "Does the picture return to the previous shot within two frames, in colour and layout?",
            [("defect", "flash (one finding)"), ("dismissed", "edit")],
        ),
        (
            "Still run",
            "Identical at full resolution? Starts at an edit? Does the shot move around it?",
            [("defect", "stall"), ("dismissed", "held shot"), ("dismissed", "slow motion")],
        ),
        (
            "Block spike",
            "Do 16&times;16 blocks change in this frame only, while clean neighbours agree?",
            [("defect", "damage"), ("review", "review")],
        ),
        (
            "Motion jump",
            "Explained by a neighbouring defect? Does whole-frame motion match the shot?",
            [("defect", "skip"), ("review", "review"), ("dismissed", "explained")],
        ),
    ]
    agent_html = "".join(
        f'<tr><td class="k">{k}</td><td>{q}</td><td>{"".join(f"<span class='tag t-{c}'>{t}</span>" for c, t in tags)}</td></tr>'
        for k, q, tags in agent_rows
    )

    def result_table(r: dict) -> str:
        rows = []
        for kind in KINDS:
            b = r["by_kind"][kind]
            share = b["found"] / b["labels"] if b["labels"] else 0
            rows.append(
                f'<tr><td>{NAMES[kind]}</td><td><span class="bar"><i style="width:{share * 100:.0f}%"></i></span></td>'
                f'<td class="n">{b["found"]} of {b["labels"]}</td></tr>'
            )
        rows.append(
            f'<tr class="fp"><td>False positives</td><td></td>'
            f'<td class="n">{r["false_positives_per_1000_frames"]:.2f}/1k</td></tr>'
        )
        return "<table>" + "".join(rows) + "</table>"

    s = job["summary"]
    run_line = (
        f"This demo job {where}: {s['frames_scanned']:,} frames, "
        f"{s['ms_per_frame_first_pass']} ms per frame in the first pass, {s['tool_calls']} follow-up checks."
    )
    return f"""<!doctype html><html><head><meta charset="utf-8"><style>{CSS}</style></head><body>
<section class="slide" id="title">
  <div class="kicker">OpenCV AI Competition 2026</div>
  <h1>FrameGuard Sightline</h1>
  <div class="sub">Agentic frame quality control with OpenCV&nbsp;5 on AWS</div>
  <div class="team">Hyunsik Park &middot; solo entry</div>
  <div class="strip-note">{strip_note}</div>
  <div class="strip">{strip}</div>
</section>
<section class="slide" id="problem">
  <div class="kicker">The problem</div>
  <h1>Faults that last a fraction of a second</h1>
  <div class="cards">{card_html}</div>
</section>
<section class="slide" id="approach">
  <div class="kicker">How a job runs</div>
  <h1>OpenCV 5 measures, the agent checks, a person decides</h1>
  <div class="flow">{"".join(flow)}</div>
  {trace_card(job)}
  <div class="note">Nothing changes before a person approves. Every check is written to a trace.</div>
</section>
<section class="slide" id="architecture"><img src="{(ROOT / "docs" / "architecture.svg").as_uri()}"></section>
<section class="slide" id="agent">
  <div class="kicker">The agent loop</div>
  <h1>Each suspicion gets the check that fits it</h1>
  <table><thead><tr><th>Candidate</th><th>Follow-up check</th><th>Possible verdicts</th></tr></thead><tbody>{agent_html}</tbody></table>
  <div class="foot">Budget per job: 60 checks and 1,200 re-decoded frames; anything left goes to a person.
  Each check is logged with its trigger, its measurements and its decision.</div>
</section>
<section class="slide" id="results">
  <div class="kicker">Evaluation on held-out test sets</div>
  <h1>Found defects by class</h1>
  <div class="two">
    <div><h2>Synthetic</h2><div class="meta">{synth["clips"]} clips, {synth["frames"]:,} frames</div>{result_table(synth)}</div>
    <div><h2>Real trailers, seeded defects</h2><div class="meta">{real["clips"]} clips, {real["frames"]:,} frames (Sintel, Big Buck Bunny)</div>{result_table(real)}</div>
  </div>
  <div class="foot">Thresholds were frozen on development seeds before the test sets were generated.
  Unmodified trailers ({clean["frames"]:,} frames): {clean["defects"] or "no"} defect findings, {clean["needs_review"]} item for review.<br>{run_line}</div>
</section>
<section class="slide" id="limits">
  <div class="kicker">Limits and responsible use</div>
  <h1>What Sightline does not do</h1>
  <div class="cols">
    <div><h2>Limits</h2><ul>
      <li>Frame skips are the weakest class on real footage: 1 of 3 found, 2 false alarms in one fast shot.</li>
      <li>A dark flash is reported as a black frame: the right frames, the wrong name.</li>
      <li>Repairs hold a neighbouring frame. They confirm the diagnosis; a re-render is the real fix.</li>
      <li>Grain, interlacing, HDR and very long takes are not yet measured.</li></ul></div>
    <div><h2>Responsible use</h2><ul>
      <li>Inspects picture integrity only; it does not detect or identify people.</li>
      <li>Dismissed items stay visible with their reasons, so a person can overrule them.</li>
      <li>No footage changes without an approval recorded with its author and time.</li>
      <li>Uploads and reports expire after 7 days; logs hold numbers only.</li></ul></div>
  </div>
</section>
<section class="slide" id="end">
  <h1>FrameGuard Sightline</h1>
  <div class="sub">Code, technical report and evaluation are linked from the Devpost submission.</div>
  <div class="credits">Sintel and Big Buck Bunny trailers &copy; Blender Foundation, CC BY 3.0 (durian.blender.org, peach.blender.org).<br>
  Narration: synthetic voice generated locally with Qwen3-TTS 1.7B (Apache-2.0).<br>
  Built with OpenCV 5.0 on AWS Lambda (Graviton), Step Functions, S3, DynamoDB, API Gateway and CloudWatch.</div>
</section>
</body></html>"""


async def render(html: Path, out: Path) -> None:
    async with async_playwright() as p:
        try:  # the installed Google Chrome; Playwright's own build is the fallback
            browser = await p.chromium.launch(channel="chrome")
        except Exception:
            browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        await page.goto(html.as_uri())
        await page.wait_for_load_state("networkidle")
        for section in await page.query_selector_all("section.slide"):
            name = await section.get_attribute("id")
            await section.screenshot(path=str(out / f"{name}.png"))
        await browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--out", default=Path("demo/build/slides"), type=Path)
    parser.add_argument(
        "--where",
        default="on Lambda (arm64, 2&nbsp;GB)",
        help="where the recorded job ran, for the results slide",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    frames = args.out / "frames"
    labels = json.loads(args.clip.with_suffix(".labels.json").read_text())["labels"]
    extract_frames(
        args.clip,
        frames,
        set(STRIP) | {lab["start"] - 1 for lab in labels} | {lab["start"] for lab in labels},
    )
    synth = json.loads((ROOT / "results" / "eval_synthetic_test.json").read_text())
    real = json.loads((ROOT / "results" / "eval_real_seeded_test.json").read_text())
    trailers = json.loads((ROOT / "results" / "eval_clean_trailers.json").read_text())
    flagged = [f for clip in trailers["per_clip"] for f in clip["false_positives"]]
    clean = {
        "frames": trailers["frames"],
        "needs_review": sum(f["verdict"] == "needs_review" for f in flagged),
        "defects": sum(f["verdict"] == "defect" for f in flagged),
    }
    html = args.out / "slides.html"
    job = json.loads(args.job.read_text())
    html.write_text(build_html(frames.resolve(), job, labels, synth, real, clean, args.where))
    asyncio.run(render(html.resolve(), args.out))
    print("slides ->", args.out)


if __name__ == "__main__":
    main()
