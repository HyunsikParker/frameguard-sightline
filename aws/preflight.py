"""Read-only checks to run before aws/deploy.sh: right user, right region, and every service the stack
needs answers in that region instead of being refused by an organization policy (SCP).

    python aws/preflight.py --expect-region ap-southeast-2 --env-file path/to/aws.env

Each call reads or names a resource that does not exist yet, so "not found" means the service is
reachable. Nothing is created and no paid operation is called. Key values, ARNs and the account ID
are never printed. Exit status 0 only when every check passes.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CREDENTIAL_NAMES = (
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_REGION",
    "AWS_DEFAULT_REGION",
)


def load_env(env_file: Path | None) -> dict:
    env = dict(os.environ)
    if env_file and env_file.exists():
        for raw in env_file.read_text().splitlines():
            line = raw.strip().removeprefix("export ").strip()
            key, sep, value = line.partition("=")
            if sep and key in CREDENTIAL_NAMES:
                env[key] = value.strip().strip('"').strip("'")
    return env


def classify(run: subprocess.CompletedProcess) -> str:
    err = run.stderr
    if run.returncode == 0:
        return "ok"
    if "service control policy" in err:
        return "scp-deny"
    if "identity-based policy" in err or "AccessDenied" in err:
        return "identity-deny"
    if re.search(r"NotFound|does not exist|NoSuchBucket|NoSuchEntity|StateMachineDoesNotExist|\(404\)", err):
        return "not-found"
    return "other"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--expect-region", required=True)
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--stack", default="sightline")
    args = parser.parse_args()
    env = load_env(args.env_file)
    region = env.get("AWS_REGION") or env.get("AWS_DEFAULT_REGION")
    if region != args.expect_region:
        print(f"region: configured {region or 'none'}, expected {args.expect_region}")
        return 1
    env["AWS_REGION"] = env["AWS_DEFAULT_REGION"] = region

    def aws(*cmd: str) -> subprocess.CompletedProcess:
        return subprocess.run(["aws", *cmd], env=env, capture_output=True, text=True, timeout=60)

    who = aws("sts", "get-caller-identity", "--output", "json")
    if who.returncode != 0:
        print("caller: no usable credentials")
        return 1
    identity = json.loads(who.stdout)
    account, user = identity["Account"], identity["Arn"].rsplit("/", 1)[-1]
    print(f"caller: {user} (account id not shown), region {region}")

    s = args.stack
    checks = [
        ("cloudformation", ("cloudformation", "describe-stacks", "--stack-name", s), {"not-found", "ok"}),
        ("lambda", ("lambda", "get-function", "--function-name", f"{s}-analyzer"), {"not-found", "ok"}),
        (
            "dynamodb",
            ("dynamodb", "describe-table", "--table-name", f"{s}-JobTable-preflight"),
            {"not-found"},
        ),
        (
            "states",
            (
                "stepfunctions",
                "describe-state-machine",
                "--state-machine-arn",
                f"arn:aws:states:{region}:{account}:stateMachine:{s}-jobs",
            ),
            {"not-found", "ok"},
        ),
        ("apigateway", ("apigatewayv2", "get-apis", "--query", "length(Items)"), {"ok"}),
        (
            "logs",
            (
                "logs",
                "describe-log-groups",
                "--log-group-name-prefix",
                f"/aws/lambda/{s}",
                "--query",
                "length(logGroups)",
            ),
            {"ok"},
        ),
        ("iam", ("iam", "get-role", "--role-name", f"{s}-preflight-absent"), {"not-found"}),
        # Weak: S3 may answer NoSuchBucket before it evaluates policies; deploy.sh is the real test.
        (
            "s3 (weak)",
            ("s3api", "list-objects-v2", "--bucket", f"{s}-artifacts-{account}-{region}", "--max-items", "1"),
            {"not-found", "ok"},
        ),
    ]
    failed = 0
    for name, cmd, expected in checks:
        outcome = classify(aws(*cmd))
        passed = outcome in expected
        failed += not passed
        print(f"{'pass' if passed else 'FAIL'}  {name:15s} {outcome}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
