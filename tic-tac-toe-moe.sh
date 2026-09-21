#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HOST="127.0.0.1"
PORT="18777"
PID_FILE="$ROOT/tic-tac-toe-moe.pid"
LOG_FILE="$ROOT/tic-tac-toe-moe.log"

running() {
  [[ -f "$PID_FILE" ]] && kill -0 "$(<"$PID_FILE")" 2>/dev/null
}

case "${1:-}" in
  start)
    if running; then printf 'Already running (PID %s) at http://%s:%s/\n' "$(<"$PID_FILE")" "$HOST" "$PORT"; exit 0; fi
    rm -f "$PID_FILE"
    cd "$ROOT"
    nohup python3 -m http.server "$PORT" --bind "$HOST" >"$LOG_FILE" 2>&1 &
    pid=$!; printf '%s\n' "$pid" >"$PID_FILE"
    for _ in {1..30}; do
      if python3 - "$PORT" <<'PY'
import sys
from urllib.request import urlopen
try:
    body=urlopen(f'http://127.0.0.1:{sys.argv[1]}/',timeout=1).read()
    raise SystemExit(0 if b'Tic' in body else 1)
except Exception:
    raise SystemExit(1)
PY
      then printf 'Started (PID %s): http://%s:%s/\n' "$pid" "$HOST" "$PORT"; exit 0; fi
      sleep 0.1
    done
    printf 'Server did not become healthy; see %s\n' "$LOG_FILE" >&2; kill "$pid" 2>/dev/null || true; rm -f "$PID_FILE"; exit 1
    ;;
  stop)
    if running; then kill "$(<"$PID_FILE")"; rm -f "$PID_FILE"; printf 'Stopped.\n'; else rm -f "$PID_FILE"; printf 'Not running.\n'; fi
    ;;
  status)
    if running; then printf 'Running (PID %s): http://%s:%s/\n' "$(<"$PID_FILE")" "$HOST" "$PORT"; else printf 'Not running.\n'; exit 1; fi
    ;;
  *) printf 'Usage: %s {start|stop|status}\n' "$0" >&2; exit 64;;
esac
