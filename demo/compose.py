"""Assemble the demo video: slides and the recorded review flow, narration and captions.

    python demo/compose.py --app demo/build/app --slides demo/build/slides --out demo/build/video

Each scene lasts as long as its narration. Recorded steps that only wait for the cloud
(scanning, acting) are sped up to fit and labelled with the factor; other steps are held on
their last frame when the narration runs longer. Captions show the narration text in a band
under the picture. Output: sightline-demo.mp4 (1920x1080, 30 fps, H.264/AAC) and a matching .srt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import wave
from pathlib import Path

import numpy as np
from playwright.async_api import async_playwright

FPS = 30
LEAD, GAP, TAIL = 0.45, 0.35, 0.7  # seconds of silence before, between and after lines
FADE = 0.3
BAND_CSS = """
* { margin: 0; box-sizing: border-box; }
body { background: #15181c; }
#band { width: 1920px; height: 120px; background: #15181c; color: #f3f4f6; display: flex; align-items: center;
  justify-content: center; padding: 0 140px; text-align: center;
  font: 500 33px/1.32 -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; }
#band.small { font-size: 29px; }
#badge { display: inline-block; padding: 8px 18px; border-radius: 8px;
  color: #fff; background: rgba(21,24,28,.82);
  font: 600 26px -apple-system, BlinkMacSystemFont, "Helvetica Neue", Arial, sans-serif; }
"""


def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True, capture_output=True)


async def render_texts(captions: list[str], badges: list[str], out: Path) -> None:
    """Caption bands and speed badges as PNG images, so ffmpeg needs no text renderer."""
    async with async_playwright() as p:
        try:  # the installed Google Chrome; Playwright's own build is the fallback
            browser = await p.chromium.launch(channel="chrome")
        except Exception:
            browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 200})
        await page.set_content(f"<style>{BAND_CSS}</style><div id='band'></div><span id='badge'></span>")
        for i, text in enumerate(["", *captions]):
            name = "caption_blank.png" if i == 0 else f"caption_{i - 1:03d}.png"
            await page.evaluate(
                "([t, small]) => { const b = document.getElementById('band'); b.textContent = t;"
                " b.className = small ? 'small' : ''; }",
                [text, len(text) > 150],
            )
            await page.locator("#band").screenshot(path=str(out / name))
        for i, text in enumerate(badges):
            await page.evaluate("(t) => { document.getElementById('badge').textContent = t; }", text)
            await page.locator("#badge").screenshot(
                path=str(out / f"badge_{i:03d}.png"), omit_background=True
            )
        await browser.close()


def read_wav(path: str) -> np.ndarray:
    with wave.open(path) as w:
        assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(round(seconds * 24000)), dtype=np.int16)


def concat_list(entries: list[tuple[Path, float]], path: Path) -> None:
    lines = []
    for file, duration in entries:
        lines += [f"file '{file.resolve().as_posix()}'", f"duration {max(duration, 0.001):.4f}"]
    lines.append(f"file '{entries[-1][0].resolve().as_posix()}'")  # the demuxer needs the last file twice
    path.write_text("\n".join(lines) + "\n")


def app_entries(app: Path, beats: dict, start: str, end: str) -> tuple[list[tuple[Path, float]], float]:
    times = {b["name"]: b["t"] for b in beats["beats"]}
    t0, t1 = times[start], times[end]
    frames = beats["frames"]
    stamps = [f["t"] for f in frames]
    first = max(0, int(np.searchsorted(stamps, t0, side="right")) - 1)  # the picture on screen at t0
    entries = []
    k = first
    while k < len(frames) and (k == first or stamps[k] < t1):
        a = max(stamps[k], t0)
        b = min(stamps[k + 1], t1) if k + 1 < len(frames) else t1
        entries.append((app / "frames" / frames[k]["file"], b - a))
        k += 1
    return entries, t1 - t0


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--manifest", default=Path("demo/build/narration_manifest.json"), type=Path)
    parser.add_argument("--app", default=Path("demo/build/app"), type=Path)
    parser.add_argument("--slides", default=Path("demo/build/slides"), type=Path)
    parser.add_argument("--out", default=Path("demo/build/video"), type=Path)
    parser.add_argument("--draft-label", default=None, help="burn this label into every scene (drafts only)")
    args = parser.parse_args()
    out = args.out
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    scenes = json.loads(args.manifest.read_text())
    beats = json.loads((args.app / "beats.json").read_text())

    # Plan every scene: narration timing, visual length, speed factor.
    plans, captions = [], []
    badges = [args.draft_label] if args.draft_label else []  # badge 0 is the draft label when set
    for scene in scenes:
        starts, t = [], scene.get("lead", LEAD)
        for line in scene["lines"]:
            starts.append(t)
            t += line["duration"] + GAP
        narration = t - GAP + scene.get("tail", TAIL)
        plan = {"scene": scene, "starts": starts, "narration": narration, "speed": 1.0, "badge": None}
        kind, _, spec = scene["visual"].partition(":")
        if kind == "app":
            a, b = spec.split(">")
            entries, length = app_entries(args.app, beats, a, b)
            plan.update(entries=entries, length=length)
            if scene.get("timelapse") and length > narration * 1.2:
                plan["speed"] = length / narration
                plan["badge"] = len(badges)
                badges.append(f"processing, shown {plan['speed']:.0f}× faster")
            plan["duration"] = max(narration, length / plan["speed"])
        else:
            plan["duration"] = narration
        plan["caption_ids"] = []
        for line in scene["lines"]:
            plan["caption_ids"].append(len(captions))
            captions.append(line["text"])
        plans.append(plan)
    asyncio.run(render_texts(captions, badges, work))

    # One video file per scene, then a single concat with the narration track.
    parts, audio, srt, clock = [], [], [], 0.0
    for n, plan in enumerate(plans):
        scene, duration = plan["scene"], plan["duration"]
        content = work / f"content_{n:02d}.txt"
        kind, _, spec = scene["visual"].partition(":")
        if kind == "app":
            concat_list(plan["entries"], content)
        else:
            concat_list([(args.slides / f"{spec}.png", duration)], content)
        band_entries = []
        cursor = 0.0
        for i, cid in enumerate(plan["caption_ids"]):
            begin = plan["starts"][i]
            if begin > cursor:
                band_entries.append((work / "caption_blank.png", begin - cursor))
            stop = plan["starts"][i + 1] if i + 1 < len(plan["starts"]) else duration
            band_entries.append((work / f"caption_{cid:03d}.png", stop - begin))
            cursor = stop
            line = scene["lines"][i]
            srt.append((clock + begin, clock + begin + line["duration"] + 0.2, line["text"]))
        band = work / f"band_{n:02d}.txt"
        concat_list(band_entries, band)
        speed = plan["speed"]
        video = (
            f"[0:v]setpts=(PTS-STARTPTS)/{speed:.4f},fps={FPS},scale=1920:960:flags=lanczos,"
            f"tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f},setpts=PTS-STARTPTS[c0]"
        )
        inputs = [
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(content),
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(band),
        ]
        extra = 2
        if plan["badge"] is not None:
            inputs += ["-loop", "1", "-i", str(work / f"badge_{plan['badge']:03d}.png")]
            video += f";[c0][{extra}:v]overlay=W-w-36:28:shortest=1[c1]"
            extra += 1
        else:
            video += ";[c0]null[c1]"
        if args.draft_label:
            inputs += ["-loop", "1", "-i", str(work / "badge_000.png")]
            video += f";[c1][{extra}:v]overlay=W-w-36:H-h-28:shortest=1[c]"  # bottom right, clear of headings
        else:
            video += ";[c1]null[c]"
        video += (
            f";[1:v]fps={FPS},tpad=stop_mode=clone:stop_duration={duration:.3f},trim=duration={duration:.3f},"
            f"setpts=PTS-STARTPTS[b];[c][b]vstack,format=yuv420p"
        )
        fades = kind == "slide" or n == 0 or plans[n - 1]["scene"]["visual"].startswith("slide")
        if kind == "slide" or (n + 1 < len(plans) and plans[n + 1]["scene"]["visual"].startswith("slide")):
            video += f",fade=t=out:st={duration - FADE:.3f}:d={FADE}"
        if fades:
            video += f",fade=t=in:st=0:d={FADE}"
        part = work / f"scene_{n:02d}.mp4"
        run(
            [
                "ffmpeg",
                "-y",
                *inputs,
                "-filter_complex",
                video + "[v]",
                "-map",
                "[v]",
                "-t",
                f"{duration:.3f}",
                "-r",
                str(FPS),
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "17",
                "-pix_fmt",
                "yuv420p",
                str(part),
            ]
        )
        parts.append(part)
        track = [silence(plan["starts"][0])]
        for i, line in enumerate(scene["lines"]):
            track.append(read_wav(line["wav"]))
            nxt = plan["starts"][i + 1] if i + 1 < len(plan["starts"]) else duration
            track.append(silence(nxt - plan["starts"][i] - line["duration"]))
        audio.append(np.concatenate(track)[: int(round(duration * 24000))])
        audio[-1] = np.pad(audio[-1], (0, max(0, int(round(duration * 24000)) - len(audio[-1]))))
        clock += duration
        print(f"{scene['id']:13s} {duration:6.2f}s  speed {speed:4.1f}", flush=True)

    with wave.open(str(work / "narration.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(np.concatenate(audio).tobytes())
    (work / "parts.txt").write_text("".join(f"file '{p.resolve().as_posix()}'\n" for p in parts))
    final = out / "sightline-demo.mp4"
    run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(work / "parts.txt"),
            "-i",
            str(work / "narration.wav"),
            "-map",
            "0:v",
            "-map",
            "1:a",
            "-c:v",
            "copy",
            "-af",
            "loudnorm=I=-16:TP=-1.5:LRA=7,aresample=48000",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-movflags",
            "+faststart",
            "-shortest",
            str(final),
        ]
    )

    def stamp(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"

    (out / "sightline-demo.srt").write_text(
        "\n".join(f"{i}\n{stamp(a)} --> {stamp(b)}\n{text}\n" for i, (a, b, text) in enumerate(srt, 1))
    )
    print(f"{final}  {clock:.1f}s")


if __name__ == "__main__":
    main()
