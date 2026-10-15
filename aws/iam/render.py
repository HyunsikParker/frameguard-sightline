"""Fill the IAM policy templates for one account and region (offline; nothing is sent to AWS).

    python aws/iam/render.py --account 123456789012 --region us-west-2 --out aws/build/iam

* sightline-runtime-boundary: the permissions boundary every stack role must carry. An
  administrator creates it once with exactly this name; the template references it by name.
* sightline-deployer-stack and sightline-deployer-iam: the two managed policies for the user that
  runs aws/deploy.sh. That user can create roles only with the boundary attached, cannot remove or
  replace a boundary, cannot edit the boundary policy and cannot change its own permissions.

The stack name must stay `sightline`: resource names and ARNs in the policies use that prefix.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = ("sightline-runtime-boundary", "sightline-deployer-stack", "sightline-deployer-iam")
MANAGED_POLICY_LIMIT = 6144  # characters, whitespace excluded


def render(account: str, region: str) -> dict[str, dict]:
    if not re.fullmatch(r"\d{12}", account):
        raise SystemExit("--account must be a 12-digit AWS account ID")
    if not re.fullmatch(r"[a-z]{2}(-[a-z]+)+-\d", region):
        raise SystemExit("--region must look like us-west-2")
    out = {}
    for name in NAMES:
        text = (
            (HERE / f"{name}.json").read_text().replace("<ACCOUNT_ID>", account).replace("<REGION>", region)
        )
        policy = json.loads(text)
        size = len(re.sub(r"\s", "", json.dumps(policy)))
        if size > MANAGED_POLICY_LIMIT:
            raise SystemExit(f"{name} is {size} characters, above the managed-policy limit")
        out[name] = policy
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--account", required=True)
    parser.add_argument("--region", required=True)
    parser.add_argument("--out", default=Path("aws/build/iam"), type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for name, policy in render(args.account, args.region).items():
        (args.out / f"{name}.json").write_text(json.dumps(policy, indent=2) + "\n")
        print(args.out / f"{name}.json")


if __name__ == "__main__":
    main()
