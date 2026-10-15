"""Synthesize the demo narration one sentence at a time with the local Qwen3-TTS runtime.

    python demo/build_narration.py --job demo/build/app/job.json

Numbers in the script come from the recorded job, so the narration matches the run shown on
screen. Each sentence is cached by its spoken text, so a changed number re-synthesizes one line.
Writes demo/build/narration_manifest.json with the caption text, the WAV path and its length.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

TTS = Path("/Volumes/Crucial X9/workstation/Bounty/tts-narration/bin/bounty-tts")
SEED = 7
ONES = [
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven",
    "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen",
]  # fmt: skip
TENS = ["_", "_", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def words(n: int) -> str:
    if n < 20:
        return ONES[n]
    if n < 100:
        return TENS[n // 10] + ("" if n % 10 == 0 else "-" + ONES[n % 10])
    rest = n % 100
    return ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + words(rest))


def placeholders(job: dict) -> dict:
    s = job["summary"]
    review = s["needs_review"]
    phrase = {0: "no items", 1: "one item"}.get(review, f"{review} items")
    phrase_words = {0: "no items", 1: "one item"}.get(review, f"{words(review)} items")
    values = {
        "tool_calls": s["tool_calls"],
        "defects": s["defects"],
        "dismissed": s["dismissed"],
        "needs_review_phrase": phrase,
        "needs_review_phrase_words": phrase_words,
    }
    for key in ("tool_calls", "defects", "dismissed"):
        values[f"{key}_words"] = words(int(values[key]))
    return values


def synthesize(say: str, out_dir: Path) -> dict:
    digest = hashlib.sha256(f"{SEED}|Aiden|{say}".encode()).hexdigest()[:16]
    wav = out_dir / f"{digest}.wav"
    receipt = out_dir / f"{digest}.json"
    if not (wav.exists() and receipt.exists()):
        run = subprocess.run(
            [str(TTS), "--text", say, "--output", str(wav), "--seed", str(SEED)],
            capture_output=True,
            text=True,
            check=True,
        )
        start = run.stdout.index("{")
        receipt.write_text(run.stdout[start:])
    info = json.loads(receipt.read_text())
    return {"wav": str(wav), "duration": info["duration_seconds"]}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--script", default=Path("demo/narration.json"), type=Path)
    parser.add_argument("--job", required=True, type=Path)
    parser.add_argument("--out", default=Path("demo/build"), type=Path)
    args = parser.parse_args()
    values = placeholders(json.loads(args.job.read_text()))
    tts_dir = args.out / "tts"
    tts_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for scene in json.loads(args.script.read_text())["scenes"]:
        lines = []
        for line in scene["lines"]:
            text = line["text"].format(**values)
            say = line.get("say", line["text"]).format(**values)
            lines.append({"text": text, "say": say, **synthesize(say, tts_dir)})
            print(f"{scene['id']:13s} {lines[-1]['duration']:5.2f}s  {text[:70]}", flush=True)
        manifest.append({**{k: v for k, v in scene.items() if k != "lines"}, "lines": lines})
    (args.out / "narration_manifest.json").write_text(json.dumps(manifest, indent=1))
    total = sum(line["duration"] for scene in manifest for line in scene["lines"])
    print(f"{sum(len(s['lines']) for s in manifest)} lines, {total:.1f}s of speech")


if __name__ == "__main__":
    main()
