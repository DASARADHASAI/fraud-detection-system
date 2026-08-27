#!/usr/bin/env bash
#
# Take this machine from cold to demo-ready with one command.
#
# Everything here exists because of something that can go wrong while a panel
# is watching: a stale database making the feed look wrong, a service that is
# listening but not yet ready, or a first Ollama call that takes eight seconds
# in the middle of a sentence.
#
#   ./demo.sh              reset, start both services, seed, open the dashboard
#   ./demo.sh --no-reset    keep the existing database (rehearsing mid-session)
#   ./demo.sh --stop        stop both services and exit
#
# It NEVER runs features.py. That is a ~50 second, ~4.5 GB pass over 6.3M rows
# and has no place on demo day -- the models are already trained and on disk.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML="$ROOT/ml-service"
BACKEND="$ROOT/backend"
PY="$ML/venv/bin/python"
LOGS="$ROOT/.demo-logs"

ML_URL="http://localhost:5001"
APP_URL="http://localhost:8080"

RESET=1
[[ "${1:-}" == "--no-reset" ]] && RESET=0

say()  { printf '\033[36m  %s\033[0m\n' "$*"; }
ok()   { printf '\033[32m  ok  %s\033[0m\n' "$*"; }
die()  { printf '\033[31m\n  STOPPED: %s\033[0m\n\n' "$*" >&2; exit 1; }

stop_services() {
  # lsof rather than a pidfile: the services are often started by hand during
  # rehearsal, and a pidfile would not know about those.
  for port in 5001 8080; do
    local pids
    pids="$(lsof -ti tcp:"$port" 2>/dev/null || true)"
    [[ -n "$pids" ]] && kill $pids 2>/dev/null && say "stopped whatever held port $port"
  done
  # Spring Boot's shutdown releases the H2 file lock a moment after the port.
  sleep 2
}

if [[ "${1:-}" == "--stop" ]]; then
  stop_services; ok "both services stopped"; exit 0
fi

# --- Fail fast, with an instruction rather than a stack trace -----------------

[[ -x "$PY" ]] || die "no venv at $PY
    fix:  cd ml-service && python3 -m venv venv && ./venv/bin/pip install -r requirements.txt"

for artefact in random_forest.joblib isolation_forest.joblib scaler.joblib global_stats.json; do
  [[ -f "$ML/models/$artefact" ]] || die "missing model file: models/$artefact
    fix:  cd ml-service && ./venv/bin/python train_model.py
    NOTE: this takes minutes. Do not discover this on demo day."
done

command -v mvn >/dev/null || die "maven not on PATH"

mkdir -p "$LOGS"

# --- Reset ------------------------------------------------------------------

stop_services

if (( RESET )); then
  # H2 is file-backed, so every rehearsal accumulates in the feed and skews
  # /api/stats. Delete it while the backend is down; Spring recreates it.
  rm -f "$BACKEND"/data/frauddb*.db && say "cleared the H2 database"
fi

# --- Start the ML service ---------------------------------------------------

say "starting the ML service..."
( cd "$ML" && nohup "$PY" app.py > "$LOGS/ml-service.log" 2>&1 & )

for i in {1..60}; do
  if curl -sf "$ML_URL/health" >/dev/null 2>&1; then break; fi
  (( i == 60 )) && die "ML service never became healthy. Last lines:
$(tail -n 15 "$LOGS/ml-service.log" 2>/dev/null)"
  sleep 1
done
ok "ML service healthy on 5001"

# --- Start the backend ------------------------------------------------------

say "starting the backend (Maven is slow the first time)..."
( cd "$BACKEND" && nohup mvn -q spring-boot:run > "$LOGS/backend.log" 2>&1 & )

# Poll for readiness rather than sleeping a fixed duration. A fixed sleep is
# either too short on a cold Maven cache or wasted time on a warm one.
for i in {1..180}; do
  if curl -sf "$APP_URL/api/health" >/dev/null 2>&1; then break; fi
  (( i == 180 )) && die "backend never became healthy. Last lines:
$(tail -n 20 "$LOGS/backend.log" 2>/dev/null)"
  sleep 1
done
ok "backend healthy on 8080"

# --- Clear the ML service's in-memory account history -----------------------

if (( RESET )); then
  curl -sf -X POST "$ML_URL/reset" >/dev/null 2>&1 && say "cleared in-memory account history"
fi

# --- Warm Ollama ------------------------------------------------------------

# The first call after idle loads the model and can take several seconds. Doing
# that now costs nothing; doing it mid-demo looks like the system hanging.
# Best effort only: the system is designed to run without Ollama.
say "warming the language model..."
if curl -sf -m 90 -X POST "$ML_URL/explain" \
     -H 'Content-Type: application/json' \
     -d '{"step":254,"type":"PAYMENT","amount":1000,"nameOrig":"C-WARMUP",
          "oldbalanceOrg":5000,"newbalanceOrig":4000,"nameDest":"M-WARMUP",
          "oldbalanceDest":2000,"newbalanceDest":3000}' >/dev/null 2>&1; then
  ok "language model warm"
else
  say "Ollama not responding -- fine, narration falls back to templates"
fi

# --- Seed and open ----------------------------------------------------------

if (( RESET )); then
  say "seeding ordinary traffic..."
  ( cd "$ML" && "$PY" demo_seed.py ) || die "seeding failed"
fi

printf '\033[32m\n  READY\033[0m  %s\n\n' "$APP_URL"
printf '  before you speak, confirm all three health dots are green.\n'
printf '  logs:   %s\n' "$LOGS"
printf '  stop:   ./demo.sh --stop\n\n'

command -v open >/dev/null && open "$APP_URL"
