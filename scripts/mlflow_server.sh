#!/usr/bin/env bash
# Ephemeral MLflow server with proxied artifacts.
#
# Usage: mlflow_server.sh start [dev|registry]   (default: registry)
#        mlflow_server.sh stop
#
# Always serve artifacts through the server (--serve-artifacts) so the sqlite DB
# stores portable mlflow-artifacts:/ URIs instead of machine-specific file: paths.
set -euo pipefail
cd "$(dirname "$0")/.."

PID_FILE=.mlflow-server.pid

start() {
  local mode=${1:-registry}
  local port db_path artifacts
  if [[ "$mode" == "dev" ]]; then
    port=5000; db_path=mlflow-dev.db; artifacts=mlruns-dev
  else
    port=5001; db_path=registry/mlflow.db; artifacts=registry/mlruns
    mkdir -p registry
  fi

  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "mlflow server already running (pid $(cat "$PID_FILE"))" >&2
    exit 1
  fi

  nohup uv run mlflow server \
    --host 127.0.0.1 --port "$port" \
    --backend-store-uri "sqlite:///$db_path" \
    --artifacts-destination "./$artifacts" \
    --serve-artifacts \
    > mlflow-server.log 2>&1 &
  echo $! > "$PID_FILE"

  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
      echo "mlflow server ($mode) ready at http://127.0.0.1:$port (pid $(cat "$PID_FILE"))"
      echo "export MLFLOW_TRACKING_URI=http://127.0.0.1:$port"
      exit 0
    fi
    sleep 1
  done
  echo "mlflow server failed to become healthy; see mlflow-server.log" >&2
  exit 1
}

stop() {
  if [[ -f "$PID_FILE" ]]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "mlflow server stopped"
  else
    echo "no pid file; nothing to stop"
  fi
}

case "${1:-}" in
  start) start "${2:-registry}" ;;
  stop) stop ;;
  *) echo "usage: $0 {start [dev|registry]|stop}" >&2; exit 2 ;;
esac
