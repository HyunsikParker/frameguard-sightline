"""Record the review-page flow as a sharp screencast with beat markers for the demo video.

    python demo/record_app.py --url https://<api-id>.execute-api.<region>.amazonaws.com \
        --clip data/demo/sintel_test_2003.mp4 --out demo/build/app

Frames come from the Chrome DevTools screencast. The page is laid out at 1440x720 CSS pixels
and zoomed by 4/3 into a 1920x960 viewport, so text stays sharp after encoding. ``beats.json``
records when each step of the flow started; the composer cuts the narration to those beats.
A small dot shows where the pointer is.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
import json
import shutil
import time
from pathlib import Path

from playwright.async_api import Page, async_playwright

VIEWPORT = {"width": 1920, "height": 960}
POINTER_JS = """
(() => {
  const install = () => {
    if (document.getElementById('demo-pointer')) return;
    document.body.style.zoom = '1.3333';  // a 1440x720 layout rendered at 1920x960
    const dot = document.createElement('div');
    dot.id = 'demo-pointer';
    Object.assign(dot.style, {
      position: 'fixed', left: '-40px', top: '-40px', width: '20px', height: '20px',
      margin: '-10px 0 0 -10px', borderRadius: '50%', background: 'rgba(29,33,38,.55)',
      border: '2px solid #fff', boxShadow: '0 1px 4px rgba(0,0,0,.35)', zIndex: 2147483647,
      pointerEvents: 'none', transition: 'transform .12s ease-out',
    });
    document.documentElement.appendChild(dot);  // outside the zoomed body, in viewport pixels
    const follow = (e) => { dot.style.left = e.clientX + 'px'; dot.style.top = e.clientY + 'px'; };
    addEventListener('mousemove', follow, true);
    addEventListener('mousedown', () => { dot.style.transform = 'scale(.7)'; }, true);
    addEventListener('mouseup', () => { dot.style.transform = 'scale(1)'; }, true);
  };
  if (document.readyState === 'loading') addEventListener('DOMContentLoaded', install); else install();
})();
"""


SEEK_AND_PLAY = """
async (v, at) => {
  v.muted = true;
  if (v.readyState < 1) await new Promise((r) => v.addEventListener('loadedmetadata', r, { once: true }));
  const seeked = new Promise((r) => v.addEventListener('seeked', r, { once: true }));
  v.currentTime = at;
  await seeked;
  await v.play();
}
"""


class Recorder:
    def __init__(self, out: Path):
        self.out = out
        self.frames: list[dict] = []
        self.beats: list[dict] = []
        self.pointer = (960.0, 480.0)

    def beat(self, name: str) -> None:
        self.beats.append({"name": name, "t": time.time()})
        print(f"beat {name}", flush=True)

    async def glide(self, page: Page, target, duration: float = 0.7) -> None:
        """Move the pointer smoothly to the centre of an element."""
        locator = page.locator(target) if isinstance(target, str) else target
        await locator.scroll_into_view_if_needed()
        box = await locator.bounding_box()
        await self.glide_to(page, box["x"] + box["width"] / 2, box["y"] + box["height"] / 2, duration)

    async def click(self, page: Page, target) -> None:
        await self.glide(page, target)
        await page.mouse.down()
        await asyncio.sleep(0.08)
        await page.mouse.up()

    async def rest(self, page: Page) -> None:
        """Park the pointer in the right margin so it does not cover what just opened."""
        await self.glide_to(page, 1800.0, self.pointer[1])

    async def glide_to(self, page: Page, x: float, y: float, duration: float = 0.5) -> None:
        steps = max(8, int(duration * 40))
        x0, y0 = self.pointer
        for i in range(1, steps + 1):
            k = i / steps
            k = k * k * (3 - 2 * k)
            await page.mouse.move(x0 + (x - x0) * k, y0 + (y - y0) * k)
            await asyncio.sleep(duration / steps)
        self.pointer = (x, y)


async def scroll_into_centre(page: Page, locator) -> None:
    await locator.evaluate("(el) => el.scrollIntoView({behavior: 'smooth', block: 'center'})")
    await asyncio.sleep(1.8)  # measure the element only after the smooth scroll has settled


async def scroll_to(page: Page, selector: str, block: str = "start") -> None:
    await page.evaluate(
        "([s, b]) => document.querySelector(s).scrollIntoView({behavior: 'smooth', block: b})",
        [selector, block],
    )
    await asyncio.sleep(1.4)


def narration_lines(manifest: Path | None) -> dict[str, list[float]]:
    """Seconds of each spoken line per scene."""
    if manifest is None or not manifest.exists():
        return {}
    return {s["id"]: [line["duration"] for line in s["lines"]] for s in json.loads(manifest.read_text())}


def narration_lengths(manifest: Path | None) -> dict[str, float]:
    """Seconds of narration per scene, laid out exactly as demo/compose.py lays them out."""
    if manifest is None or not manifest.exists():
        return {}
    from compose import GAP, LEAD, TAIL  # same directory

    lengths = {}
    for scene in json.loads(manifest.read_text()):
        lines = scene["lines"]
        speech = sum(line["duration"] for line in lines) + GAP * (len(lines) - 1)
        lengths[scene["id"]] = scene.get("lead", LEAD) + speech + scene.get("tail", TAIL)
    return lengths


async def record(url: str, clip: Path, out: Path, preview_at: float, manifest: Path | None) -> None:
    say = narration_lengths(manifest)
    spoken = narration_lines(manifest)

    def hold(scene: str, used: float, default: float, least: float = 1.5) -> float:
        """Wait long enough that the step lasts as long as its narration."""
        return max(least, say[scene] - used) if scene in say else default

    if out.exists():
        shutil.rmtree(out)
    (out / "frames").mkdir(parents=True)
    rec = Recorder(out)
    async with async_playwright() as p:
        try:  # Google Chrome plays H.264 previews; Playwright's Chromium only plays WebM
            browser = await p.chromium.launch(channel="chrome", headless=True)
        except Exception:
            browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(viewport=VIEWPORT)
        await context.add_init_script(POINTER_JS)
        page = await context.new_page()
        cdp = await context.new_cdp_session(page)

        async def on_frame(params: dict) -> None:
            name = f"{len(rec.frames):06d}.jpg"
            (out / "frames" / name).write_bytes(base64.b64decode(params["data"]))
            rec.frames.append({"t": params["metadata"]["timestamp"], "file": name})
            with contextlib.suppress(Exception):  # the session may already be closing
                await cdp.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})

        cdp.on("Page.screencastFrame", lambda params: asyncio.ensure_future(on_frame(params)))
        await page.goto(url)
        await page.wait_for_selector("#run")
        await cdp.send(
            "Page.startScreencast",
            {"format": "jpeg", "quality": 92, "maxWidth": 1920, "maxHeight": 960, "everyNthFrame": 1},
        )
        await page.mouse.move(*rec.pointer)
        rec.beat("page")
        await asyncio.sleep(hold("app_upload", 3.0, 3.0))

        await rec.glide(page, "#file")
        await page.set_input_files("#file", str(clip))
        rec.beat("file_chosen")
        await asyncio.sleep(1.5)
        await rec.click(page, "#run")
        rec.beat("upload")
        state = "document.getElementById('state').textContent"
        await page.wait_for_function(f"/scan|queued|analy/.test({state})", timeout=180_000)
        rec.beat("scanning")
        await page.wait_for_function("!document.getElementById('summary').hidden", timeout=900_000)
        rec.beat("summary")
        await page.wait_for_function(f"/awaiting|done/.test({state})", timeout=900_000)
        rec.beat("awaiting")
        await asyncio.sleep(2)

        await scroll_to(page, "#findings")
        rec.beat("findings")
        await asyncio.sleep(hold("app_findings", 2 + 1.4 + 1.5, 5.0))

        flash = page.locator("#rows tr", has_text="flash frame").locator("summary")
        if await flash.count():
            await flash.first.scroll_into_view_if_needed()
            await rec.click(page, flash.first)
            rec.beat("trace")
            await rec.rest(page)
            await asyncio.sleep(hold("app_trace", 2.8, 7.0, least=3.0))
            await rec.click(page, flash.first)  # fold it again
            await asyncio.sleep(0.8)

        approve = page.locator("#rows input[type=radio][value=approve]")
        approvals = await approve.count()
        await rec.click(page, "#dismissed-summary")
        rec.beat("dismissed")
        await scroll_into_centre(page, page.locator("#dismissed-box"))
        await rec.rest(page)
        lines = spoken.get("app_review", [])
        # Line 1 talks about the dismissed list, line 2 about the item for review, line 3 about approval.
        from compose import GAP, LEAD

        await asyncio.sleep(max(2.0, LEAD + lines[0] + GAP - 2.3) if lines else 5.0)
        await rec.click(page, "#dismissed-summary")
        await asyncio.sleep(0.6)
        review = page.locator("#rows tr .v-needs_review")
        if await review.count():
            await scroll_into_centre(page, review.first)
            await rec.glide(page, review.first)
            await asyncio.sleep(max(1.5, lines[1] + GAP - 2.0) if len(lines) > 1 else 3.0)

        for i in range(approvals):
            await rec.click(page, approve.nth(i))
            await asyncio.sleep(0.35)
        rec.beat("approved")
        await asyncio.sleep(1.5)
        await rec.click(page, "#submit")
        rec.beat("sent")
        await page.wait_for_function(f"{state}.trim() === 'done'", timeout=900_000)
        rec.beat("done")
        await asyncio.sleep(1)
        await scroll_to(page, "#actions")
        await asyncio.sleep(3)
        video = page.locator("#action-body video")
        preview_check = None
        if await video.count():
            await rec.glide(page, video)
            await video.evaluate(SEEK_AND_PLAY, preview_at)
            rec.beat("preview")
            await asyncio.sleep(hold("app_verify", 2.6, 8.0, least=5.0))
            preview_check = await video.evaluate(
                "(v) => ({readyState: v.readyState, currentTime: v.currentTime, duration: v.duration,"
                " paused: v.paused, error: v.error && v.error.code,"
                " src: v.currentSrc.split('?')[0].split('/').pop()})"
            )
        rec.beat("end")
        await cdp.send("Page.stopScreencast")
        await asyncio.sleep(0.5)
        job = await page.evaluate("new URLSearchParams(location.search).get('job')")
        data = await page.evaluate("(id) => fetch('/api/jobs/' + id).then((r) => r.json())", job)
        await browser.close()

    (out / "beats.json").write_text(json.dumps({"beats": rec.beats, "frames": rec.frames}, indent=1))
    keep = {k: data[k] for k in ("summary", "findings", "trace") if k in data}
    keep["job_id"] = job
    keep["preview_check"] = preview_check  # the repaired preview really played in the browser
    keep["job"] = {k: v for k, v in data["job"].items() if k in ("status", "actions", "pending")}
    (out / "job.json").write_text(json.dumps(keep, indent=1))
    print(f"{len(rec.frames)} frames, {len(rec.beats)} beats -> {out}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--url", required=True)
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--out", default=Path("demo/build/app"), type=Path)
    parser.add_argument("--preview-at", default=29.5, type=float, help="seconds into the repair preview")
    parser.add_argument(
        "--manifest",
        default=Path("demo/build/narration_manifest.json"),
        type=Path,
        help="narration lengths; each step is held as long as its narration",
    )
    args = parser.parse_args()
    asyncio.run(record(args.url, args.clip, args.out, args.preview_at, args.manifest))


if __name__ == "__main__":
    main()
