#!/usr/bin/env bash
# Remove every resource the deployment created (job data, stack, artifacts).
set -euo pipefail
cd "$(dirname "$0")"
ENV_FILE="${SIGHTLINE_ENV:-../../.env}"
# Credentials: AWS_* lines from the env file when it has them, otherwise the CLI's own chain
# (aws login, SSO or a profile). Values are exported for the AWS CLI only and never printed.
if [ -f "$ENV_FILE" ]; then
  while IFS= read -r line || [ -n "$line" ]; do
    line="${line#export }"
    case "$line" in
      AWS_ACCESS_KEY_ID=*|AWS_SECRET_ACCESS_KEY=*|AWS_SESSION_TOKEN=*|AWS_REGION=*|AWS_DEFAULT_REGION=*)
        key="${line%%=*}"; value="${line#*=}"; value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
        export "$key=$value" ;;
    esac
  done < "$ENV_FILE"
fi
STACK="${STACK:-sightline}"
REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-$(aws configure get region 2>/dev/null || true)}}"
if [ -z "$REGION" ]; then  # no silent default: the account may allow regional services in one region only
  echo "Set AWS_REGION (env file or environment) to the region this account deploys to." >&2
  exit 1
fi
export AWS_REGION="$REGION" AWS_DEFAULT_REGION="$REGION"
if ! ACCOUNT="$(aws sts get-caller-identity --query Account --output text 2>/dev/null)"; then
  echo "No usable AWS credentials. Run 'aws login', or put AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY in $ENV_FILE." >&2
  exit 1
fi
BUCKET="$(aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs[?OutputKey=='JobBucketName'].OutputValue" --output text 2>/dev/null || true)"
[ -n "$BUCKET" ] && [ "$BUCKET" != "None" ] && aws s3 rm "s3://${BUCKET}" --recursive --only-show-errors
aws cloudformation delete-stack --stack-name "$STACK"
aws cloudformation wait stack-delete-complete --stack-name "$STACK"
aws s3 rb "s3://sightline-artifacts-${ACCOUNT}-${REGION}" --force >/dev/null 2>&1 || true
echo "removed stack ${STACK}"
