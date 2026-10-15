"""Zip the judge-facing source tree: everything except what .gitignore excludes.

    .venv/bin/python tools/package_source.py --out dist/frameguard-sightline-src.zip

Footage, generated data, build output and internal notes are left out (see .gitignore). The
script refuses to write the archive if a file looks like it holds an AWS secret key.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SECRET = re.compile(
    rb"(AKIA|ASIA)[0-9A-Z]{16}|aws_secret_access_key\s*=|AWS_SECRET_ACCESS_KEY=[A-Za-z0-9/+]{20}"
)


def ignore_patterns() -> list[str]:
    lines = (ROOT / ".gitignore").read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.startswith("#")]


def ignored(rel: str, patterns: list[str]) -> bool:
    parts = rel.split("/")
    for pattern in patterns:
        if pattern.endswith("/"):
            name = pattern.rstrip("/")
            if "/" in name:
                if rel.startswith(name + "/"):
                    return True
            elif name in parts[:-1]:
                return True
        elif fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(parts[-1], pattern):
            return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=ROOT / "dist" / "frameguard-sightline-src.zip", type=Path)
    args = parser.parse_args()
    patterns = ignore_patterns()
    files = sorted(
        p for p in ROOT.rglob("*") if p.is_file() and not ignored(p.relative_to(ROOT).as_posix(), patterns)
    )
    leaks = [p.relative_to(ROOT).as_posix() for p in files if SECRET.search(p.read_bytes())]
    if leaks:
        raise SystemExit(f"refusing to package, possible secrets in: {leaks}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, Path("frameguard-sightline") / path.relative_to(ROOT))
    print(f"{len(files)} files -> {args.out} ({args.out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
