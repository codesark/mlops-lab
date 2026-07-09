#!/usr/bin/env bash
# One-time Azure bootstrap for this repo (already executed 2026-07-10; kept for
# reproducibility). Creates the ML workspace and the OIDC trust that lets
# GitHub Actions log runs and manage the model registry with no stored secrets.
#
# Prereqs: az login (Owner on the subscription), gh auth login.
set -euo pipefail

RG=mlops-lab-rg
WS=mlops-lab-ws
LOCATION=centralindia
REPO=codesark/mlops-lab
APP_NAME=mlops-lab-gha

az extension add -n ml --only-show-errors || true
for p in Microsoft.MachineLearningServices Microsoft.Storage Microsoft.KeyVault microsoft.insights; do
  az provider register -n "$p"
done
az provider show -n Microsoft.MachineLearningServices --query registrationState -o tsv

az group create -n "$RG" -l "$LOCATION" --query properties.provisioningState -o tsv
az ml workspace create -n "$WS" -g "$RG"

TRACKING_URI=$(az ml workspace show -n "$WS" -g "$RG" --query mlflow_tracking_uri -o tsv)
WS_ID=$(az ml workspace show -n "$WS" -g "$RG" --query id -o tsv)

APP_ID=$(az ad app create --display-name "$APP_NAME" --query appId -o tsv)
az ad sp create --id "$APP_ID" --query id -o tsv || true

# Federated credentials: one per OIDC subject GitHub Actions presents.
for SUBJECT in "ref:refs/heads/main" "pull_request" "environment:staging" "environment:production"; do
  NAME=$(echo "$SUBJECT" | tr ':/' '--')
  az ad app federated-credential create --id "$APP_ID" --parameters "{
    \"name\": \"gha-$NAME\",
    \"issuer\": \"https://token.actions.githubusercontent.com\",
    \"subject\": \"repo:$REPO:$SUBJECT\",
    \"audiences\": [\"api://AzureADTokenExchange\"]
  }" --query name -o tsv
done

az role assignment create --assignee "$APP_ID" --role "AzureML Data Scientist" --scope "$WS_ID" \
  --query roleDefinitionName -o tsv

gh variable set AZURE_CLIENT_ID --repo "$REPO" --body "$APP_ID"
gh variable set AZURE_TENANT_ID --repo "$REPO" --body "$(az account show --query tenantId -o tsv)"
gh variable set AZURE_SUBSCRIPTION_ID --repo "$REPO" --body "$(az account show --query id -o tsv)"
gh variable set MLFLOW_TRACKING_URI --repo "$REPO" --body "$TRACKING_URI"

echo "done. tracking URI: $TRACKING_URI"
