#!/usr/bin/env bash
# Deploy a Ray cluster for lakehouse_ann into a Kubernetes (or OpenShift) namespace.
#
#   deploy/kubernetes/deploy.sh up       render + apply, wait for pods, print `ray status`
#   deploy/kubernetes/deploy.sh code     refresh the lakehouse_ann code configmap, restart pods
#   deploy/kubernetes/deploy.sh down     delete everything this script created
#   deploy/kubernetes/deploy.sh run X.py [args]   copy a script to the head and run it there
#
# Settings (env): NS (default lakehouse-ann), IMAGE (default python:3.12-slim),
# WORKERS (5), WORKER_CPUS (8), WORKER_MEM (32Gi), KUBECTL (kubectl; `oc` works too).
set -euo pipefail
NS=${NS:-lakehouse-ann}
export IMAGE=${IMAGE:-python:3.12-slim} WORKERS=${WORKERS:-5} WORKER_CPUS=${WORKER_CPUS:-8} WORKER_MEM=${WORKER_MEM:-32Gi}
K="${KUBECTL:-kubectl} -n $NS"
cd "$(dirname "$0")/../.."
HERE=deploy/kubernetes

render() { envsubst '${IMAGE} ${WORKERS} ${WORKER_CPUS} ${WORKER_MEM}' < $HERE/ray-cluster.yaml; }
code() { $K create configmap lakehouse-ann-code --from-file=lakehouse_ann/ --dry-run=client -o yaml | $K apply -f -; }
head_pod() { $K get pod -l app=ray,component=head -o name | head -1; }
wait_ready() {
  $K rollout status deploy/ray-head --timeout=15m
  $K rollout status deploy/ray-worker --timeout=15m
  $K exec "$(head_pod)" -c ray-head -- ray status
}

case "${1:-up}" in
  up)   code; render | $K apply -f -; wait_ready ;;
  code) code; $K rollout restart deploy/ray-head deploy/ray-worker; wait_ready ;;
  down) render | $K delete -f - --ignore-not-found; $K delete configmap lakehouse-ann-code --ignore-not-found ;;
  run)  shift; P=$(head_pod); P=${P#pod/}
        $K cp "$1" "$P:/tmp/$(basename "$1")" -c ray-head
        S=/tmp/$(basename "$1"); shift
        $K exec "$P" -c ray-head -- python "$S" "$@" ;;
  *)    echo "usage: $0 up|code|down|run script.py [args]" >&2; exit 2 ;;
esac
