#!/usr/bin/env bash
set -uo pipefail

run_id="${DATUS_TEST_RUN_ID:-}"
if [ -z "$run_id" ]; then
  echo "DATUS_TEST_RUN_ID is not set; skipping Testcontainers cleanup"
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "Docker is not available; skipping Testcontainers cleanup"
  exit 0
fi

if ! container_output="$(
  docker ps -aq \
    --filter "label=org.testcontainers=true" \
    --filter "label=com.datus.ci.run-id=${run_id}"
)"; then
  echo "Failed to list Testcontainers for run ${run_id}" >&2
  exit 1
fi

container_ids=()
while IFS= read -r container_id; do
  [ -n "$container_id" ] && container_ids+=("$container_id")
done <<< "$container_output"

if [ "${#container_ids[@]}" -eq 0 ]; then
  echo "No Testcontainers found for run ${run_id}"
  exit 0
fi

cleanup_failed=0
for container_id in "${container_ids[@]}"; do
  container_run_id="$(docker inspect --format '{{index .Config.Labels "com.datus.ci.run-id"}}' "$container_id" 2>/dev/null || true)"
  if [ "$container_run_id" != "$run_id" ]; then
    echo "Skipping container ${container_id}: run label does not match"
    continue
  fi
  docker rm -fv "$container_id" || cleanup_failed=1
done

if ! remaining="$(
  docker ps -aq \
    --filter "label=org.testcontainers=true" \
    --filter "label=com.datus.ci.run-id=${run_id}" \
    | wc -l
)"; then
  echo "Failed to list remaining Testcontainers for run ${run_id}" >&2
  cleanup_failed=1
fi
if [ "$remaining" -ne 0 ]; then
  echo "Failed to remove ${remaining} Testcontainers for run ${run_id}" >&2
  cleanup_failed=1
fi

exit "$cleanup_failed"
