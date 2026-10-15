"""The IAM policy templates and the CloudFormation roles agree on the permissions boundary."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BOUNDARY = "policy/sightline-runtime-boundary"


def _render():
    spec = importlib.util.spec_from_file_location("iam_render", ROOT / "aws" / "iam" / "render.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.render("123456789012", "eu-west-1")


def test_every_stack_role_carries_the_runtime_boundary():
    template = (ROOT / "aws" / "template.yaml").read_text()
    roles = template.count("Type: AWS::IAM::Role")
    bounded = len(
        re.findall(
            r"PermissionsBoundary: !Sub arn:\$\{AWS::Partition\}:iam::\$\{AWS::AccountId\}:"
            + re.escape(BOUNDARY),
            template,
        )
    )
    assert roles == 3 and bounded == roles


def test_rendered_policies_are_complete_and_scoped_to_the_region():
    policies = _render()
    for name, policy in policies.items():
        text = json.dumps(policy)
        assert "<" not in text, f"{name} still has a placeholder"
        assert "us-east-1" not in text
    boundary_arn = f"arn:aws:iam::123456789012:{BOUNDARY}"
    iam = {s["Sid"]: s for s in policies["sightline-deployer-iam"]["Statement"]}
    create = iam["CreateStackRolesOnlyWithTheRuntimeBoundary"]
    assert create["Action"] == "iam:CreateRole"
    assert create["Condition"]["StringEquals"]["iam:PermissionsBoundary"] == boundary_arn
    assert iam["NeverRemoveOrReplaceABoundary"]["Effect"] == "Deny"
    passed = iam["PassStackRolesOnlyToLambdaAndStepFunctions"]["Condition"]["StringEquals"][
        "iam:PassedToService"
    ]
    assert sorted(passed) == ["lambda.amazonaws.com", "states.amazonaws.com"]
    runtime = json.dumps(policies["sightline-runtime-boundary"])
    assert "arn:aws:states:eu-west-1:123456789012:stateMachine:sightline-jobs" in runtime
    assert '"iam:' not in runtime  # the boundary grants no IAM action at all


def test_preflight_tells_an_organization_policy_refusal_from_a_missing_resource():
    import subprocess

    spec = importlib.util.spec_from_file_location("preflight", ROOT / "aws" / "preflight.py")
    preflight = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(preflight)

    def result(code: int, err: str) -> subprocess.CompletedProcess:
        return subprocess.CompletedProcess(["aws"], code, "", err)

    scp = "User: x is not authorized to perform: x with an explicit deny in a service control policy"
    assert preflight.classify(result(254, scp)) == "scp-deny"
    identity = "not authorized to perform: x because no identity-based policy allows the x action"
    assert preflight.classify(result(254, identity)) == "identity-deny"
    missing = "An error occurred (ResourceNotFoundException) when calling the GetFunction operation"
    assert preflight.classify(result(254, missing)) == "not-found"
    assert preflight.classify(result(0, "")) == "ok"
