#!/usr/bin/env bash
# Local throwaway MLflow server for offline development. The real tracking
# server + registry is the Azure ML workspace (get its URI via
# `az ml workspace show -n mlops-lab-ws -g mlops-lab-rg --query mlflow_tracking_uri -o tsv`).
#
# Usage: mlflow_server.sh start | stop
set -euo pipefail
cd "$(dirname "$0")/.."

PID_FILE=.mlflow-server.pid
PORT=5000

start() {
  if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
    echo "mlflow server already running (pid $(cat "$PID_FILE"))" >&2
    exit 1
  fi

  nohup uv run mlflow server \
    --host 127.0.0.1 --port "$PORT" \
    --backend-store-uri sqlite:///mlflow-dev.db \
    --artifacts-destination ./mlruns-dev \
    --serve-artifacts \
    > mlflow-server.log 2>&1 &
  echo $! > "$PID_FILE"

  for _ in $(seq 1 60); do
    if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "dev mlflow server ready at http://127.0.0.1:$PORT (pid $(cat "$PID_FILE"))"
      echo "export MLFLOW_TRACKING_URI=http://127.0.0.1:$PORT"
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
  start) start ;;
  stop) stop ;;
  *) echo "usage: $0 {start|stop}" >&2; exit 2 ;;
esac
