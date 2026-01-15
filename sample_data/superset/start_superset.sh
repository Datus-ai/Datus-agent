#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

helm repo add superset https://apache.github.io/superset

# install & start superset
helm upgrade --install superset superset/superset -n default -f "${SCRIPT_DIR}/examples-values.yaml"

# Directly tunnel one pod's port into your localhost
kubectl port-forward -n default service/superset 8088:8088 > /dev/null &

kubectl port-forward -n default svc/superset-postgresql 15432:5432 > /dev/null &
