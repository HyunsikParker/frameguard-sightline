"""AWS Lambda entry point that runs Sightline (OpenCV 5) on Graviton (arm64).

Step Functions calls it twice per job:

* ``analyze``: download the upload, run the scan and the agent loop, store the report in S3
  and the job state in DynamoDB, and tell the state machine how many findings wait for a person.
* ``act``: replay the person's decisions onto the stored report, run the approved actions
  (repair preview, re-render list), verify the repair, and store the results.

Only numbers and small thumbnails leave the function. Operational metrics are written in
CloudWatch Embedded Metric Format and never contain image content.
"""

from __future__ import annotations

import json
import mimetypes
import os
import shutil
import tempfile
import time
from decimal import Decimal
from pathlib import Path

import boto3

from sightline.actions import ApprovalError, apply_approved, record_decision
from sightline.io import InputRejected
from sightline.pipeline import analyze
from sightline.report import write_report

s3 = boto3.client("s3")
table = boto3.resource("dynamodb").Table(os.environ["JOB_TABLE"])
BUCKET = os.environ["JOB_BUCKET"]


def _upload_dir(local: Path, prefix: str) -> None:
    for path in local.rglob("*"):
        if path.is_file():  # a typed object lets every browser show the thumbnails and play the preview
            ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            key = f"{prefix}/{path.relative_to(local).as_posix()}"
            s3.upload_file(str(path), BUCKET, key, ExtraArgs={"ContentType": ctype})


def _download_prefix(prefix: str, local: Path) -> None:
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix + "/"):
        for item in page.get("Contents", []):
            target = local / item["Key"][len(prefix) + 1 :]
            target.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, item["Key"], str(target))


def _metrics(job_id: str, summary: dict) -> None:
    names = {
        "FramesScanned": ("frames_scanned", "Count"),
        "MsPerFrame": ("ms_per_frame_first_pass", "Milliseconds"),
        "Defects": ("defects", "Count"),
        "NeedsReview": ("needs_review", "Count"),
        "ToolCalls": ("tool_calls", "Count"),
    }
    record = {
        "_aws": {
            "Timestamp": int(time.time() * 1000),
            "CloudWatchMetrics": [
                {
                    "Namespace": "Sightline",
                    "Dimensions": [[]],
                    "Metrics": [{"Name": n, "Unit": unit} for n, (_, unit) in names.items()],
                }
            ],
        },
        "job_id": job_id,
    }
    record.update({n: summary[key] for n, (key, _) in names.items()})
    print(json.dumps(record))


def _set_state(job_id: str, **fields) -> None:
    expressions = ", ".join(f"#{k} = :{k}" for k in fields)
    table.update_item(
        Key={"job_id": job_id},
        UpdateExpression=f"SET {expressions}",
        ExpressionAttributeNames={f"#{k}": k for k in fields},
        ExpressionAttributeValues={
            f":{k}": v for k, v in json.loads(json.dumps(fields), parse_float=Decimal).items()
        },
    )


def handler(event: dict, context) -> dict:
    job_id = event["job_id"]
    input_name = Path(event["input_name"]).name
    work = Path(tempfile.mkdtemp(dir="/tmp"))
    source = work / input_name
    job_dir = work / "job"
    try:
        s3.download_file(BUCKET, f"uploads/{job_id}/{input_name}", str(source))
        if event["action"] == "analyze":
            _set_state(job_id, status="analyzing")
            try:
                job = analyze(str(source))
            except InputRejected as error:  # the person sees why; there is nothing to review
                _set_state(job_id, status="failed", error=str(error))
                return {"job_id": job_id, "pending": 0}
            record = write_report(job, job_dir)
            _upload_dir(job_dir, f"jobs/{job_id}")
            pending = [f for f in record["findings"] if f["approval"] == "pending"]
            # With pending findings the workflow's next step stores its task token and moves the
            # job to awaiting_approval; until then the page shows the findings without choices.
            fields = {"summary": record["summary"], "pending": len(pending)}
            if not pending:
                fields["status"] = "done"
            _set_state(job_id, **fields)
            _metrics(job_id, record["summary"])
            return {"job_id": job_id, "pending": len(pending)}

        if event["action"] == "act":
            _download_prefix(f"jobs/{job_id}", job_dir)
            decisions = event.get("decisions")
            if decisions is None:  # a manual re-run without the task result uses the job record
                decisions = table.get_item(Key={"job_id": job_id}).get("Item", {}).get("decisions", [])
            for decision in decisions:
                try:
                    record_decision(job_dir, decision["finding"], decision["decision"], decision["by"])
                except ApprovalError as error:
                    print(json.dumps({"job_id": job_id, "skipped_decision": str(error)}))
            result = apply_approved(job_dir, str(source))
            _upload_dir(job_dir, f"jobs/{job_id}")
            _set_state(job_id, status="done", actions=result)
            return {"job_id": job_id, **result}

        raise ValueError(f"unknown action {event['action']}")
    finally:
        shutil.rmtree(work, ignore_errors=True)
