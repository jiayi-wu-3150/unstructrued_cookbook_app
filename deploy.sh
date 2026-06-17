#!/bin/bash
# Deploy the Streamlit app to Databricks Apps
# Usage: ./deploy.sh

set -e

PROFILE="fevm-serverless-stable-r4umw1"
APP_NAME="unstructured-cookbook"
LOCAL_PATH="$(dirname "$0")/app"
WORKSPACE_PATH="/Workspace/Users/jiayi.wu@databricks.com/unstructured_parsing/app"

echo "==> Syncing local app/ to workspace..."
databricks workspace import-dir "$LOCAL_PATH" "$WORKSPACE_PATH" \
  --overwrite \
  --profile "$PROFILE"

echo "==> Deploying $APP_NAME..."
databricks apps deploy "$APP_NAME" \
  --source-code-path "$WORKSPACE_PATH" \
  --profile "$PROFILE"

echo "==> Done! App URL: https://unstructured-cookbook-7474653681240163.aws.databricksapps.com"
