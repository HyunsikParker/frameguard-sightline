"""Command line: analyze a clip, decide on findings, apply approved actions, evaluate.

python -m sightline.cli analyze INPUT --out jobs/demo
python -m sightline.cli approve jobs/demo flash_frame:39-40 --by editor
python -m sightline.cli act jobs/demo INPUT
python -m sightline.cli synth data/public --seeds 1-24
python -m sightline.cli evaluate data/public --out results/eval.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .actions import ApprovalError, apply_approved, record_decision
from .io import InputRejected
from .pipeline import analyze
from .report import write_report


def _seeds(text: str) -> list[int]:
    if "-" in text:
        first, last = (int(v) for v in text.split("-", 1))
        return list(range(first, last + 1))
    return [int(v) for v in text.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="sightline", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("analyze", help="scan a video file or PNG/JPEG sequence directory")
    p.add_argument("input")
    p.add_argument("--out", required=True)

    p = sub.add_parser("approve", help="approve or reject one pending finding")
    p.add_argument("job_dir")
    p.add_argument("finding", help="finding id, e.g. flash_frame:39-40")
    p.add_argument("--reject", action="store_true")
    p.add_argument("--by", required=True)
    p.add_argument("--note", default="")

    p = sub.add_parser("act", help="apply approved actions (repair preview, re-render list)")
    p.add_argument("job_dir")
    p.add_argument("input")

    p = sub.add_parser("synth", help="write seeded synthetic clips with labels")
    p.add_argument("out")
    p.add_argument("--seeds", default="1-24")

    p = sub.add_parser("evaluate", help="score the agent on a labelled synthetic set")
    p.add_argument("data_dir")
    p.add_argument("--out", required=True)

    args = parser.parse_args(argv)
    try:
        if args.command == "analyze":
            record = write_report(analyze(args.input), Path(args.out))
            print(json.dumps(record["summary"], indent=1))
        elif args.command == "approve":
            finding = record_decision(
                Path(args.job_dir), args.finding, "reject" if args.reject else "approve", args.by, args.note
            )
            print(f"{args.finding}: {finding['approval']}")
        elif args.command == "act":
            print(json.dumps(apply_approved(Path(args.job_dir), args.input), indent=1))
        elif args.command == "synth":
            from .synth import write_set

            paths = write_set(Path(args.out), _seeds(args.seeds))
            print(f"wrote {len(paths)} clips to {args.out}")
        elif args.command == "evaluate":
            from .evaluate import evaluate_set

            result = evaluate_set(Path(args.data_dir))
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(result, indent=1, default=float))
            print(json.dumps({k: v for k, v in result.items() if k != "per_clip"}, indent=1, default=float))
    except (InputRejected, ApprovalError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
