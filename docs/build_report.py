"""Build docs/REPORT.html (self-contained) and docs/REPORT.pdf (A4) from docs/REPORT.md.

    .venv/bin/python docs/build_report.py

Needs pandoc on PATH and Google Chrome (or Playwright's Chromium) for printing.
"""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path

from playwright.async_api import async_playwright

DOCS = Path(__file__).resolve().parent


async def print_pdf(html: Path, pdf: Path) -> None:
    async with async_playwright() as p:
        try:
            browser = await p.chromium.launch(channel="chrome")
        except Exception:
            browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.goto(html.as_uri())
        await page.pdf(path=str(pdf), prefer_css_page_size=True, print_background=True)
        await browser.close()


def main() -> None:
    subprocess.run(
        [
            "pandoc",
            "REPORT.md",
            "--standalone",
            "--embed-resources",
            "--css",
            "report.css",
            "--metadata",
            "pagetitle=Sightline report",
            "-o",
            "REPORT.html",
        ],
        cwd=DOCS,
        check=True,
    )
    asyncio.run(print_pdf(DOCS / "REPORT.html", DOCS / "REPORT.pdf"))
    print(DOCS / "REPORT.pdf")


if __name__ == "__main__":
    main()
