#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_NAME="${RUN_NAME:-mamba31_fresh_seed_0_v5}"
RUN_DIR="${RUN_DIR:-$ROOT_DIR/experiment/checkpoints/$RUN_NAME}"
RESULT_DIR="${RESULT_DIR:-$ROOT_DIR/experiment/results/$RUN_NAME}"
LOG_DIR="${LOG_DIR:-$ROOT_DIR/experiment/logs}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/${RUN_NAME}.log}"
PID_FILE="${PID_FILE:-$RUN_DIR/training.pid}"
TOTAL_ENV_STEPS="${TOTAL_ENV_STEPS:-1000000}"
CUDA_DEVICES="${CUDA_DEVICES:-0,1}"

is_running() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  [[ "$pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$pid" 2>/dev/null
}

start_training() {
  if is_running; then
    echo "Training is already running (PID $(<"$PID_FILE"))."
    exit 0
  fi

  mkdir -p "$RUN_DIR" "$RESULT_DIR" "$LOG_DIR"
  local resume_mode=0
  if [[ -f "$RUN_DIR/checkpoint.pth" ]]; then
    resume_mode=1
  elif [[ -n "$(find "$RUN_DIR" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Run directory is non-empty but has no checkpoint: $RUN_DIR" >&2
    exit 2
  fi

  cd "$ROOT_DIR"
  nohup setsid env \
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
    RUN_DIR="$RUN_DIR" \
    RESULT_DIR="$RESULT_DIR" \
    RESUME_FROM_LEGACY="$resume_mode" \
    TOTAL_ENV_STEPS="$TOTAL_ENV_STEPS" \
    ./scripts/run_mamba31_split_experiment.sh \
    >>"$LOG_FILE" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$PID_FILE"
  sleep 2
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "Training failed to start. Inspect $LOG_FILE" >&2
    exit 1
  fi
  echo "Training started (PID $pid, resume=$resume_mode)."
  echo "Log: $LOG_FILE"
}

stop_training() {
  if ! is_running; then
    echo "Training is not running."
    exit 0
  fi
  local pid
  pid="$(<"$PID_FILE")"
  echo "Sending SIGINT to training process group $pid..."
  kill -INT -- "-$pid"
  for _ in $(seq 1 60); do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "Training stopped after saving checkpoint."
      exit 0
    fi
    sleep 1
  done
  echo "Training is still shutting down; check status again shortly." >&2
  exit 1
}

case "${1:-}" in
  start|resume) start_training ;;
  stop) stop_training ;;
  status)
    if is_running; then
      ps -p "$(<"$PID_FILE")" -o pid,ppid,sid,tty,stat,etime,args
    else
      echo "Training is not running."
      exit 1
    fi
    ;;
  log) tail -f "$LOG_FILE" ;;
  *)
    echo "Usage: $0 {start|resume|stop|status|log}" >&2
    exit 2
    ;;
esac
