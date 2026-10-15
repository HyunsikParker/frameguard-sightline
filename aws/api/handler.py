"""HTTP API for Sightline: the review page, job creation, job status and human decisions.

Routes (API Gateway HTTP API, Lambda proxy integration):

    GET  /                          review page
    POST /api/jobs                  create a job; returns a presigned S3 upload form
    POST /api/jobs/{id}/start       start the Step Functions workflow after the upload
    GET  /api/jobs/{id}             job state, findings, trace and short-lived media links
    POST /api/jobs/{id}/decisions   approve or reject pending findings; resumes the workflow

The same function also receives the workflow's task token while a job waits for a person
(event ``{"action": "register_token"}``), so the decision endpoint can resume the run. A job
only shows as ``awaiting_approval`` once that token is stored, so a decision cannot arrive
before the workflow can accept it.
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from decimal import Decimal
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

REGION = os.environ["AWS_REGION"]
# Regional, SigV4, virtual-hosted URLs: the global endpoint can redirect a new bucket outside
# us-east-1, which breaks the browser's cross-origin upload.
s3 = boto3.client(
    "s3",
    region_name=REGION,
    endpoint_url=f"https://s3.{REGION}.amazonaws.com",
    config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
)
sfn = boto3.client("stepfunctions")
table = boto3.resource("dynamodb").Table(os.environ["JOB_TABLE"])
BUCKET = os.environ["JOB_BUCKET"]
STATE_MACHINE = os.environ["STATE_MACHINE_ARN"]
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", str(80 * 1024 * 1024)))
RETENTION_SECONDS = int(os.environ.get("RETENTION_DAYS", "7")) * 86400
# Limits how many analyses the public page can start per UTC day. It bounds the job compute
# (one analysis, retried at most three times on Lambda service errors, plus one action step),
# not total spend: page requests and downloads through presigned links are not counted.
MAX_JOBS_PER_DAY = int(os.environ.get("MAX_JOBS_PER_DAY", "10"))
JOB_ID = re.compile(r"^[0-9a-f]{32}$")
ALLOWED_SUFFIXES = {".mp4", ".mov", ".m4v", ".mkv", ".avi"}
PAGE = (Path(__file__).parent / "index.html").read_text()
PAGE_HEADERS = {
    "content-type": "text/html; charset=utf-8",
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
}
CLOSED = ("TaskTimedOut", "TaskDoesNotExist", "InvalidToken")


def _json(status: int, body: dict) -> dict:
    def default(value):
        if isinstance(value, Decimal):
            return int(value) if value == value.to_integral_value() else float(value)
        raise TypeError(type(value))

    return {
        "statusCode": status,
        "headers": {
            "content-type": "application/json",
            "cache-control": "no-store",
            "x-content-type-options": "nosniff",
        },
        "body": json.dumps(body, default=default),
    }


def _presign(key: str) -> str:
    return s3.generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": key}, ExpiresIn=900)


def _take_daily_slot(now: int) -> bool:
    """Count jobs per UTC day in the job table and refuse once the day's cap is reached."""
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    try:
        table.update_item(
            Key={"job_id": f"quota-{day}"},
            UpdateExpression="ADD #c :one SET expires_at = :expires",
            ConditionExpression="attribute_not_exists(#c) OR #c < :cap",
            ExpressionAttributeNames={"#c": "count"},
            ExpressionAttributeValues={":one": 1, ":cap": MAX_JOBS_PER_DAY, ":expires": now + 2 * 86400},
        )
        return True
    except ClientError as error:
        if error.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return False
        raise


def create_job(body: dict) -> dict:
    name = Path(str(body.get("filename", ""))).name
    try:
        size = int(body.get("size", 0))
    except (TypeError, ValueError):
        return _json(400, {"error": "size must be a number of bytes"})
    if Path(name).suffix.lower() not in ALLOWED_SUFFIXES:
        return _json(400, {"error": f"upload a video file ({', '.join(sorted(ALLOWED_SUFFIXES))})"})
    if not 0 < size <= MAX_UPLOAD_BYTES:
        return _json(400, {"error": f"file must be between 1 byte and {MAX_UPLOAD_BYTES} bytes"})
    now = int(time.time())
    if not _take_daily_slot(now):
        return _json(
            429, {"error": f"Today's limit of {MAX_JOBS_PER_DAY} jobs is reached; try again after 00:00 UTC."}
        )
    safe = re.sub(r"[^A-Za-z0-9._-]", "_", name)[:80]
    job_id = uuid.uuid4().hex
    form = s3.generate_presigned_post(
        BUCKET,
        f"uploads/{job_id}/{safe}",
        Conditions=[["content-length-range", 1, MAX_UPLOAD_BYTES]],
        ExpiresIn=600,
    )
    table.put_item(
        Item={
            "job_id": job_id,
            "status": "created",
            "input_name": safe,
            "created_at": now,
            "expires_at": now + RETENTION_SECONDS,  # DynamoDB TTL, matching the bucket's lifecycle
        }
    )
    return _json(201, {"job_id": job_id, "upload": form})


def start_job(job_id: str) -> dict:
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if not item or item["status"] != "created":
        return _json(409, {"error": "job is not waiting to start"})
    try:
        s3.head_object(Bucket=BUCKET, Key=f"uploads/{job_id}/{item['input_name']}")
    except ClientError:
        return _json(409, {"error": "the upload has not arrived"})
    sfn.start_execution(
        stateMachineArn=STATE_MACHINE,
        name=job_id,
        input=json.dumps({"job_id": job_id, "input_name": item["input_name"]}),
    )
    table.update_item(
        Key={"job_id": job_id},
        UpdateExpression="SET #s = :s",
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={":s": "queued"},
    )
    return _json(202, {"job_id": job_id, "status": "queued"})


def get_job(job_id: str) -> dict:
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if not item:
        return _json(404, {"error": "no such job"})
    item.pop("task_token", None)  # the token resumes the workflow; it never leaves the backend
    out = {"job": item}
    try:
        record = json.loads(s3.get_object(Bucket=BUCKET, Key=f"jobs/{job_id}/findings.json")["Body"].read())
    except s3.exceptions.NoSuchKey:
        return _json(200, out)
    out["findings"] = record["findings"]
    out["trace"] = record["trace"]
    out["summary"] = record["summary"]
    out["media"] = {
        "contact_sheet": _presign(f"jobs/{job_id}/contact_sheet.jpg")
        if record.get("contact_sheet")
        else None,
        "thumbs": {k: _presign(f"jobs/{job_id}/{v}") for k, v in record.get("thumbnails", {}).items()},
    }
    if item.get("actions", {}).get("preview"):
        out["media"]["repair_preview"] = _presign(f"jobs/{job_id}/{item['actions']['preview']}")
    return _json(200, out)


def decide(job_id: str, body: dict) -> dict:
    item = table.get_item(Key={"job_id": job_id}).get("Item")
    if not item or item["status"] != "awaiting_approval" or "task_token" not in item:
        return _json(409, {"error": "job is not waiting for a decision"})
    record = json.loads(s3.get_object(Bucket=BUCKET, Key=f"jobs/{job_id}/findings.json")["Body"].read())
    pending = {
        f"{f['kind']}:{f['start']}-{f['end']}" for f in record["findings"] if f["approval"] == "pending"
    }
    by = re.sub(r"[^A-Za-z0-9 ._@-]", "", str(body.get("by", "reviewer")))[:40] or "reviewer"
    decisions = []
    for entry in body.get("decisions", []):
        finding, choice = str(entry.get("finding", "")), entry.get("decision")
        if finding not in pending or choice not in ("approve", "reject"):
            return _json(400, {"error": f"invalid decision for {finding!r}"})
        decisions.append({"finding": finding, "decision": choice, "by": by})
    if not decisions:
        return _json(400, {"error": "no decisions given"})
    try:  # the condition makes a second, concurrent submission fail instead of acting twice
        table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET decisions = :d, #s = :s REMOVE task_token",
            ConditionExpression="#s = :w AND attribute_exists(task_token)",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":d": decisions, ":s": "acting", ":w": "awaiting_approval"},
        )
    except ClientError as error:
        if error.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return _json(409, {"error": "job is not waiting for a decision"})
        raise
    try:  # the decisions travel with the task result, so the act step cannot miss them
        sfn.send_task_success(taskToken=item["task_token"], output=json.dumps({"decisions": decisions}))
    except ClientError as error:
        if error.response["Error"]["Code"] not in CLOSED:
            raise
        table.update_item(
            Key={"job_id": job_id},
            UpdateExpression="SET #s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":s": "expired"},
        )
        return _json(409, {"error": "the review window has closed; nothing was changed"})
    return _json(202, {"job_id": job_id, "status": "acting", "decisions": len(decisions)})


def handler(event: dict, context) -> dict:
    if event.get("action") == "register_token":  # invoked by Step Functions, not by HTTP
        table.update_item(
            Key={"job_id": event["job_id"]},
            UpdateExpression="SET task_token = :t, #s = :s",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={":t": event["token"], ":s": "awaiting_approval"},
        )
        return {"registered": True}

    method = event.get("requestContext", {}).get("http", {}).get("method", "GET")
    path = event.get("rawPath", "/")
    if method == "GET" and path == "/":
        return {"statusCode": 200, "headers": PAGE_HEADERS, "body": PAGE}
    try:
        body = json.loads(event.get("body") or "{}") if method == "POST" else {}
    except json.JSONDecodeError:
        return _json(400, {"error": "the request body is not JSON"})
    if not isinstance(body, dict):
        return _json(400, {"error": "the request body must be a JSON object"})
    if method == "POST" and path == "/api/jobs":
        return create_job(body)
    match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})(/start|/decisions)?", path)
    if not match or not JOB_ID.match(match.group(1)):
        return _json(404, {"error": "not found"})
    job_id, tail = match.group(1), match.group(2)
    if method == "GET" and tail is None:
        return get_job(job_id)
    if method == "POST" and tail == "/start":
        return start_job(job_id)
    if method == "POST" and tail == "/decisions":
        return decide(job_id, body)
    return _json(405, {"error": "method not allowed"})
