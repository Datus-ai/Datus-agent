#!/usr/bin/env bash
# Prepare only the disposable CodeBuild host for SAP HANA Express Docker.
# SAP's server-only Docker requirements:
# https://developers.sap.com/tutorials/hxe-ua-install-using-docker
set -euo pipefail

if [[ -z "${CODEBUILD_BUILD_ID:-}" || "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
  echo "Host preparation requires an ephemeral CodeBuild Linux x86_64 environment" >&2
  exit 1
fi

# These documented CodeBuild identifiers protect against accidentally preparing
# a local host or another project's host. They are not an authentication API.
# https://docs.aws.amazon.com/codebuild/latest/userguide/build-env-ref-env-vars.html
if [[ ! "$CODEBUILD_BUILD_ID" =~ ^datus-agent-nightly:[[:xdigit:]]{8}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{4}-[[:xdigit:]]{12}$ ]] ||
   [[ ! "${CODEBUILD_BUILD_ARN:-}" =~ ^arn:([a-z-]+):codebuild:([a-z0-9-]+):([0-9]{12}):build/(.+)$ ]]; then
  echo "Host preparation requires valid datus-agent-nightly CodeBuild identifiers" >&2
  exit 1
fi
build_region="${BASH_REMATCH[2]}"
build_id="${BASH_REMATCH[4]}"
expected_project_arn="${CODEBUILD_BUILD_ARN%:build/*}:project/datus-agent-nightly"
if [[ "$build_id" != "$CODEBUILD_BUILD_ID" ]] ||
   [[ -n "${AWS_REGION:-}" && "$AWS_REGION" != "$build_region" ]] ||
   [[ -n "${CODEBUILD_PROJECT_ARN:-}" && "$CODEBUILD_PROJECT_ARN" != "$expected_project_arn" ]]; then
  echo "CodeBuild build, project, or region identifiers do not match" >&2
  exit 1
fi

read_parameter() {
  local current
  current="$(sysctl -n "$1")" || return 1
  if [[ ! "$current" =~ ^[0-9]+$ ]]; then
    echo "Invalid numeric kernel parameter: $1" >&2
    return 1
  fi
  printf '%s\n' "$current"
}

prepare_parameter() {
  local key="$1" required="$2" comparison="$3" current verified
  current="$(read_parameter "$key")"
  if [[ "$comparison" == minimum && "$current" -ge "$required" ]] ||
     [[ "$comparison" == exact && "$current" -eq "$required" ]]; then
    printf 'Verified %s=%s\n' "$key" "$current"
    return
  fi
  if [[ "$(id -u)" == 0 ]]; then
    sysctl -w "${key}=${required}"
  else
    sudo -n sysctl -w "${key}=${required}"
  fi
  verified="$(read_parameter "$key")"
  if [[ "$comparison" == minimum && "$verified" -lt "$required" ]] ||
     [[ "$comparison" == exact && "$verified" -ne "$required" ]]; then
    echo "Kernel parameter verification failed: $key=$verified; required $comparison $required" >&2
    return 1
  fi
  printf 'Verified %s=%s\n' "$key" "$verified"
}

# Raise limits only; do not overwrite a host's larger limits. The early-kill
# switch is an exact SAP requirement rather than a numeric resource limit.
prepare_parameter vm.max_map_count 135217728 minimum
prepare_parameter fs.aio-max-nr 262144 minimum
prepare_parameter fs.file-max 20000000 minimum
prepare_parameter vm.memory_failure_early_kill 1 exact
# Do not alter host networking; Docker Compose sets the container port range.
# The adapter's full Linux/RAM/disk/bootstrap checks still run before HANA up.
