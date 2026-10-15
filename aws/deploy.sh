#!/usr/bin/env bash
# Deploy Sightline with the credentials in an env file (AWS_* lines) or the AWS CLI's own chain.
# The deploying user needs the policies in aws/iam/; the stack name must stay 'sightline'.
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
# Read-only checks first: right user and region, and no organization policy refusing a service.
python3 preflight.py --expect-region "$REGION" --stack "$STACK"
ARTIFACTS="sightline-artifacts-${ACCOUNT}-${REGION}"
if ! aws s3api head-bucket --bucket "$ARTIFACTS" >/dev/null 2>&1; then
  if [ "$REGION" = "us-east-1" ]; then aws s3api create-bucket --bucket "$ARTIFACTS" >/dev/null
  else aws s3api create-bucket --bucket "$ARTIFACTS" --create-bucket-configuration LocationConstraint="$REGION" >/dev/null; fi
  aws s3api put-public-access-block --bucket "$ARTIFACTS" --public-access-block-configuration \
    BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
fi
./build.sh
VERSION="$(date -u +%Y%m%dT%H%M%SZ)"
aws s3 cp build/analyzer.zip "s3://${ARTIFACTS}/analyzer-${VERSION}.zip" --only-show-errors
aws s3 cp build/api.zip "s3://${ARTIFACTS}/api-${VERSION}.zip" --only-show-errors
aws cloudformation deploy --stack-name "$STACK" --template-file template.yaml --capabilities CAPABILITY_IAM \
  --parameter-overrides ArtifactBucket="$ARTIFACTS" AnalyzerKey="analyzer-${VERSION}.zip" ApiKey="api-${VERSION}.zip"
aws cloudformation describe-stacks --stack-name "$STACK" --query "Stacks[0].Outputs" --output table
