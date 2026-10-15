"""The two Lambda handlers against mocked S3 and DynamoDB (moto) and a stub Step Functions client.

These tests cover the parts of the AWS deployment that local runs do not: the job states the
page relies on, the task-token handshake, typed S3 objects and the failure paths. No request
leaves the machine: moto intercepts every AWS call and the credentials are fake.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import boto3
import pytest
from botocore.exceptions import ClientError

from sightline.synth import MovingScene, write_video

moto = pytest.importorskip("moto")

ROOT = Path(__file__).resolve().parent.parent
REGION = "ap-northeast-2"
BUCKET = "sightline-jobs-test"
TABLE = "sightline-jobs-test"


class FakeStepFunctions:
    def __init__(self) -> None:
        self.started: list[dict] = []
        self.successes: list[dict] = []
        self.closed_with: str | None = None

    def start_execution(self, **kwargs) -> dict:
        self.started.append(kwargs)
        return {"executionArn": "arn:aws:states:test:execution"}

    def send_task_success(self, **kwargs) -> dict:
        if self.closed_with:
            raise ClientError({"Error": {"Code": self.closed_with, "Message": "closed"}}, "SendTaskSuccess")
        self.successes.append(kwargs)
        return {}


@pytest.fixture
def aws(monkeypatch):
    for name in ("AWS_PROFILE", "AWS_SESSION_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    env = {
        "AWS_REGION": REGION,
        "AWS_DEFAULT_REGION": REGION,
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "JOB_BUCKET": BUCKET,
        "JOB_TABLE": TABLE,
        "STATE_MACHINE_ARN": f"arn:aws:states:{REGION}:123456789012:stateMachine:sightline-jobs",
    }
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    with moto.mock_aws():
        s3 = boto3.client("s3", region_name=REGION)
        s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
        dynamodb = boto3.resource("dynamodb", region_name=REGION)
        dynamodb.create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "job_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "job_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        yield {"s3": s3, "table": dynamodb.Table(TABLE)}


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _api():
    api = _load("sightline_api_handler", "aws/api/handler.py")
    api.sfn = FakeStepFunctions()
    return api


def _http(api, method: str, path: str, body=None) -> tuple[int, dict | str]:
    event = {"requestContext": {"http": {"method": method}}, "rawPath": path}
    if body is not None:
        event["body"] = body if isinstance(body, str) else json.dumps(body)
    response = api.handler(event, None)
    if response["headers"]["content-type"].startswith("application/json"):
        return response["statusCode"], json.loads(response["body"])
    return response["statusCode"], response["body"]


def _status(aws, job_id: str) -> str:
    return aws["table"].get_item(Key={"job_id": job_id})["Item"]["status"]


FLASH = "flash_frame:30-30"
RECORD = {
    "findings": [{"kind": "flash_frame", "start": 30, "end": 30, "approval": "pending", "verdict": "defect"}],
    "trace": [],
    "summary": {"defects": 1},
    "thumbnails": {"29": "thumbs/00029.jpg"},
}


def _job_waiting_for_token(api, aws) -> str:
    status, created = _http(api, "POST", "/api/jobs", {"filename": "clip.mp4", "size": 1000})
    assert status == 201
    job_id = created["job_id"]
    aws["s3"].put_object(Bucket=BUCKET, Key=f"uploads/{job_id}/clip.mp4", Body=b"video")
    assert _http(api, "POST", f"/api/jobs/{job_id}/start", {})[0] == 202
    aws["s3"].put_object(Bucket=BUCKET, Key=f"jobs/{job_id}/findings.json", Body=json.dumps(RECORD))
    return job_id


# -------------------------------------------------------------------------------- API
def test_upload_form_is_regional_and_the_job_expires_with_the_bucket(aws):
    api = _api()
    status, page = _http(api, "GET", "/")
    assert status == 200 and "Sightline review" in page
    status, created = _http(api, "POST", "/api/jobs", {"filename": "../clip.mp4", "size": 1000})
    assert status == 201
    assert created["upload"]["url"] == f"https://{BUCKET}.s3.{REGION}.amazonaws.com/"
    assert created["upload"]["fields"]["key"] == f"uploads/{created['job_id']}/clip.mp4"
    item = aws["table"].get_item(Key={"job_id": created["job_id"]})["Item"]
    assert item["expires_at"] - item["created_at"] == 7 * 86400
    # The workflow does not start before the upload has arrived.
    assert _http(api, "POST", f"/api/jobs/{created['job_id']}/start", {})[0] == 409
    assert api.sfn.started == []


def test_decisions_wait_for_the_token_and_resume_the_workflow_once(aws):
    api = _api()
    job_id = _job_waiting_for_token(api, aws)
    assert api.sfn.started[0]["name"] == job_id
    decision = {"decisions": [{"finding": FLASH, "decision": "approve"}], "by": "editor"}

    # Findings are visible, but nothing can be decided until the workflow's token is stored.
    assert _http(api, "POST", f"/api/jobs/{job_id}/decisions", decision)[0] == 409
    api.handler({"action": "register_token", "job_id": job_id, "token": "token-1"}, None)
    status, view = _http(api, "GET", f"/api/jobs/{job_id}")
    assert view["job"]["status"] == "awaiting_approval"
    assert "task_token" not in view["job"]  # the token never leaves the backend
    assert view["media"]["thumbs"]["29"].startswith(f"https://{BUCKET}.s3.{REGION}.amazonaws.com/")

    unknown = {"decisions": [{"finding": "black_frame:1-1", "decision": "approve"}]}
    assert _http(api, "POST", f"/api/jobs/{job_id}/decisions", unknown)[0] == 400
    assert _http(api, "POST", f"/api/jobs/{job_id}/decisions", decision)[0] == 202
    assert api.sfn.successes[0]["taskToken"] == "token-1"
    sent = json.loads(api.sfn.successes[0]["output"])
    assert sent["decisions"] == [{"finding": FLASH, "decision": "approve", "by": "editor"}]
    assert _status(aws, job_id) == "acting"

    # A second submission cannot act twice.
    assert _http(api, "POST", f"/api/jobs/{job_id}/decisions", decision)[0] == 409
    assert len(api.sfn.successes) == 1


def test_a_closed_review_window_is_reported_and_nothing_changes(aws):
    api = _api()
    job_id = _job_waiting_for_token(api, aws)
    api.handler({"action": "register_token", "job_id": job_id, "token": "token-2"}, None)
    api.sfn.closed_with = "TaskTimedOut"
    decision = {"decisions": [{"finding": FLASH, "decision": "approve"}]}
    status, body = _http(api, "POST", f"/api/jobs/{job_id}/decisions", decision)
    assert status == 409 and "closed" in body["error"]
    assert _status(aws, job_id) == "expired"


@pytest.mark.parametrize(
    ("method", "path", "body", "expected"),
    [
        ("POST", "/api/jobs", "{", 400),
        ("POST", "/api/jobs", "[]", 400),
        ("POST", "/api/jobs", {"filename": "clip.mp4", "size": "big"}, 400),
        ("POST", "/api/jobs", {"filename": "clip.exe", "size": 10}, 400),
        ("POST", "/api/jobs", {"filename": "clip.mp4", "size": 81 * 1024 * 1024}, 400),
        ("GET", "/api/jobs/not-a-job", None, 404),
        ("GET", "/api/jobs/" + "0" * 32, None, 404),
        ("DELETE", "/api/jobs/" + "0" * 32, None, 405),
    ],
)
def test_malformed_requests_are_refused(aws, method, path, body, expected):
    assert _http(_api(), method, path, body)[0] == expected


# --------------------------------------------------------------------------- analyzer
def test_analyzer_stores_typed_results_and_leaves_the_decision_to_the_workflow(aws, tmp_path):
    scene, intruder = MovingScene(10), MovingScene(11)
    frames = [scene.next() for _ in range(30)]
    frames.append(intruder.next())
    scene.next()
    frames += [scene.next() for _ in range(30)]
    clip = write_video(frames, tmp_path / "clip.mp4")
    job_id = "a" * 32
    aws["s3"].upload_file(str(clip), BUCKET, f"uploads/{job_id}/clip.mp4")
    aws["table"].put_item(Item={"job_id": job_id, "status": "queued", "input_name": "clip.mp4"})
    analyzer = _load("sightline_analyzer_handler", "aws/analyzer/handler.py")

    out = analyzer.handler({"action": "analyze", "job_id": job_id, "input_name": "clip.mp4"}, None)
    assert out["pending"] >= 1
    item = aws["table"].get_item(Key={"job_id": job_id})["Item"]
    assert item["status"] == "analyzing"  # awaiting_approval is set when the token arrives
    assert item["summary"]["frames_scanned"] == len(frames)
    head = aws["s3"].head_object(Bucket=BUCKET, Key=f"jobs/{job_id}/findings.json")
    assert head["ContentType"] == "application/json"
    record = json.loads(
        aws["s3"].get_object(Bucket=BUCKET, Key=f"jobs/{job_id}/findings.json")["Body"].read()
    )
    thumb = next(iter(record["thumbnails"].values()))
    assert aws["s3"].head_object(Bucket=BUCKET, Key=f"jobs/{job_id}/{thumb}")["ContentType"] == "image/jpeg"

    flash = next(f for f in record["findings"] if f["kind"] == "flash_frame" and f["approval"] == "pending")
    decisions = [
        {"finding": f"flash_frame:{flash['start']}-{flash['end']}", "decision": "approve", "by": "t"}
    ]
    out = analyzer.handler(
        {"action": "act", "job_id": job_id, "input_name": "clip.mp4", "decisions": decisions}, None
    )
    assert out["verification"]["clean"] is True
    preview = aws["s3"].head_object(Bucket=BUCKET, Key=f"jobs/{job_id}/{out['preview']}")
    assert preview["ContentType"] in ("video/mp4", "video/webm")
    assert _status(aws, job_id) == "done"


def test_analyzer_reports_a_rejected_upload_to_the_person(aws, tmp_path):
    job_id = "b" * 32
    aws["s3"].put_object(Bucket=BUCKET, Key=f"uploads/{job_id}/broken.mp4", Body=b"not a video")
    aws["table"].put_item(Item={"job_id": job_id, "status": "queued", "input_name": "broken.mp4"})
    analyzer = _load("sightline_analyzer_handler", "aws/analyzer/handler.py")
    out = analyzer.handler({"action": "analyze", "job_id": job_id, "input_name": "broken.mp4"}, None)
    assert out["pending"] == 0
    item = aws["table"].get_item(Key={"job_id": job_id})["Item"]
    assert item["status"] == "failed" and item["error"]


def test_state_machine_definition_marks_failures_and_passes_decisions():
    text = (ROOT / "aws" / "template.yaml").read_text()
    start = text.index("DefinitionString: !Sub |") + len("DefinitionString: !Sub |")
    body = text[start : text.index("\n  HttpApi:")]
    definition = json.loads(
        body.replace("${AnalyzerFunction.Arn}", "a")
        .replace("${ApiFunction.Arn}", "b")
        .replace("${JobTable}", "t")
    )
    states = definition["States"]
    for name in ("Analyze", "Act"):
        assert states[name]["Catch"][0]["Next"] == "MarkFailed"
    assert states["WaitForPerson"]["Catch"][0] == {
        "ErrorEquals": ["States.Timeout"],
        "ResultPath": "$.error",
        "Next": "MarkExpired",
    }
    assert states["Act"]["Parameters"]["Payload"]["decisions.$"] == "$.person.decisions"
    assert states["MarkFailed"]["Next"] == "Failed" and states["MarkExpired"]["Next"] == "NoDecision"
    assert all(state.get("Type") for state in states.values())


def test_daily_job_cap_bounds_what_the_public_page_can_spend(aws, monkeypatch):
    monkeypatch.setenv("MAX_JOBS_PER_DAY", "2")
    api = _api()
    day_one = 1_792_000_000.0  # a fixed UTC day
    monkeypatch.setattr(api.time, "time", lambda: day_one)
    request = {"filename": "clip.mp4", "size": 1000}
    assert [_http(api, "POST", "/api/jobs", request)[0] for _ in range(2)] == [201, 201]
    status, body = _http(api, "POST", "/api/jobs", request)
    assert status == 429 and "limit of 2 jobs" in body["error"]
    # The counter is not a job: the job route cannot read it, and it expires on its own.
    day = api.time.strftime("%Y-%m-%d", api.time.gmtime(day_one))
    counter = aws["table"].get_item(Key={"job_id": f"quota-{day}"})["Item"]
    assert counter["count"] == 2 and counter["expires_at"] == int(day_one) + 2 * 86400
    assert _http(api, "GET", f"/api/jobs/quota-{day}")[0] == 404
    monkeypatch.setattr(api.time, "time", lambda: day_one + 86400)
    assert _http(api, "POST", "/api/jobs", request)[0] == 201
