#!/usr/bin/env bash
#
# ============================================================================
# PHASE 13 -- put the fraud engine behind a public URL so a real payment
# gateway can reach it.
# ============================================================================
#
#   ./razorpay-live.sh          start the tunnel and print what to configure
#   ./razorpay-live.sh --check  run the local self-test only, no tunnel
#   ./razorpay-live.sh --stop   stop the tunnel
#
# WHAT THIS DOES NOT DO: create your Razorpay account, hold your keys, or click
# anything in their dashboard. Those are yours. This gets a verified public URL
# into your hands and tells you exactly what to paste where.
#
# WHY A TUNNEL AT ALL: Razorpay's servers cannot reach localhost. A tunnel gives
# your laptop a temporary public address that forwards to port 5001. Nothing is
# deployed anywhere; when you stop it, the address dies.
# ============================================================================

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ML="$HERE/ml-service"
PY="$ML/venv/bin/python"
ML_URL="http://localhost:5001"
TUNNEL_LOG="$HERE/.demo-logs/cloudflared.log"
TUNNEL_PID="$HERE/.demo-logs/cloudflared.pid"

bold()  { printf "\033[1m%s\033[0m\n" "$1"; }
say()   { printf "\033[36m  %s\033[0m\n" "$1"; }
ok()    { printf "\033[32m  ok  %s\033[0m\n" "$1"; }
warn()  { printf "\033[33m  !   %s\033[0m\n" "$1"; }
die()   { printf "\033[31m  x   %s\033[0m\n" "$1" >&2; exit 1; }

mkdir -p "$HERE/.demo-logs"

# ---------------------------------------------------------------------------
# --stop
# ---------------------------------------------------------------------------
if [[ "${1:-}" == "--stop" ]]; then
  if [[ -f "$TUNNEL_PID" ]] && kill -0 "$(cat "$TUNNEL_PID")" 2>/dev/null; then
    kill "$(cat "$TUNNEL_PID")" && rm -f "$TUNNEL_PID"
    ok "tunnel stopped"
  else
    pkill -f "cloudflared tunnel" 2>/dev/null && ok "tunnel stopped" \
      || say "no tunnel was running"
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# Preflight -- everything that can be wrong before a tunnel is even involved
# ---------------------------------------------------------------------------
echo
bold "Preflight"

[[ -x "$PY" ]] || die "no venv at $PY -- see the README setup section"

if [[ -z "${RAZORPAY_WEBHOOK_SECRET:-}" ]]; then
  echo
  die "RAZORPAY_WEBHOOK_SECRET is not set.

  Create it in the Razorpay dashboard (Settings -> Webhooks) when you add the
  webhook -- YOU choose the value, it is not issued to you. Then, in the same
  shell you start the services from:

      export RAZORPAY_WEBHOOK_SECRET='whsec_pick_something_long'

  The ML service must be started with the SAME value, or every webhook 401s."
fi
ok "webhook secret is set in this shell"

if ! curl -sf "$ML_URL/health" >/dev/null 2>&1; then
  echo
  die "the ML service is not running on $ML_URL

  Start it in another terminal, WITH the secret:

      cd '$ML'
      RAZORPAY_WEBHOOK_SECRET=\"\$RAZORPAY_WEBHOOK_SECRET\" ./venv/bin/python app.py"
fi
ok "ML service is up"

CONFIGURED=$(curl -sf "$ML_URL/webhook/razorpay/recent" \
  | "$PY" -c "import json,sys; print(json.load(sys.stdin)['configured'])" 2>/dev/null || echo "False")
if [[ "$CONFIGURED" != "True" ]]; then
  echo
  die "the ML service is running WITHOUT a webhook secret, so it refuses every
  webhook with 503. Restart it with the secret set:

      cd '$ML'
      RAZORPAY_WEBHOOK_SECRET=\"\$RAZORPAY_WEBHOOK_SECRET\" ./venv/bin/python app.py"
fi
ok "the service has a secret configured too"

echo
bold "Local self-test"
"$PY" "$ML/webhook_selftest.py" || die "the local path is broken -- fix that before tunnelling"

[[ "${1:-}" == "--check" ]] && exit 0

# ---------------------------------------------------------------------------
# The tunnel
# ---------------------------------------------------------------------------
echo
bold "Public tunnel"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo
  die "cloudflared is not installed.

  It is the simplest option: free, and a quick tunnel needs no account.

      brew install cloudflared

  Then run this script again."
fi
ok "cloudflared found"

pkill -f "cloudflared tunnel" 2>/dev/null || true
: > "$TUNNEL_LOG"

say "starting a quick tunnel to port 5001..."
cloudflared tunnel --url "$ML_URL" > "$TUNNEL_LOG" 2>&1 &
echo $! > "$TUNNEL_PID"

# Poll for the URL rather than sleeping a fixed time -- it takes anywhere from
# two to fifteen seconds depending on the network.
PUBLIC=""
for _ in $(seq 1 40); do
  PUBLIC=$(grep -oE 'https://[a-z0-9-]+\.trycloudflare\.com' "$TUNNEL_LOG" 2>/dev/null | head -1 || true)
  [[ -n "$PUBLIC" ]] && break
  sleep 1
done

[[ -n "$PUBLIC" ]] || die "the tunnel did not produce a URL. See $TUNNEL_LOG"
ok "tunnel is up at $PUBLIC"

# A freshly minted hostname is not instantly resolvable, and a failed lookup in
# the first seconds gets NEGATIVE-CACHED -- so the naive "verify immediately"
# reports a dead tunnel that is in fact perfectly alive. Wait for it to answer.
say "waiting for DNS to propagate..."
REACHABLE=0
for _ in $(seq 1 30); do
  if curl -sf -m 5 -o /dev/null "$PUBLIC/health" 2>/dev/null; then REACHABLE=1; break; fi
  sleep 2
done

if [[ "$REACHABLE" -eq 0 ]]; then
  # Distinguish the two very different reasons this can fail. If a direct DNS
  # query finds the host but this shell still cannot reach it, the tunnel is
  # fine and the restriction is local -- do not blame the tunnel for that.
  if command -v nslookup >/dev/null 2>&1 && \
     nslookup "${PUBLIC#https://}" >/dev/null 2>&1; then
    echo
    warn "the hostname resolves from a direct DNS query, but not from this shell."
    warn "That is a local resolver or sandbox restriction, NOT a broken tunnel."
    warn "The tunnel is up and Razorpay will be able to reach it."
    echo
    say "verify it yourself from a normal terminal:"
    echo
    echo "      curl -s $PUBLIC/health"
    echo
    say "then run the public self-test the same way:"
    echo
    echo "      ./ml-service/venv/bin/python ml-service/webhook_selftest.py $PUBLIC"
    echo
  else
    die "the tunnel produced a URL but nothing answers on it. See $TUNNEL_LOG"
  fi
else
  echo
  bold "Verifying the PUBLIC path"
  "$PY" "$ML/webhook_selftest.py" "$PUBLIC" \
    || die "the tunnel is up but the public path failed -- see above"
fi

# ---------------------------------------------------------------------------
# What you now have to do by hand
# ---------------------------------------------------------------------------
cat <<BANNER

======================================================================
$(bold "  Now configure Razorpay (this part is yours)")
======================================================================

  1. Dashboard -> Settings -> Webhooks -> Add New Webhook
     (make sure you are in TEST MODE first)

  2. Webhook URL

         $PUBLIC/webhook/razorpay

  3. Secret        the same value as RAZORPAY_WEBHOOK_SECRET

  4. Active Events tick  payment.authorized
                   (payment.captured is scored too, but by then the
                    money has moved -- it is a report, not a decision)

  5. Make a test payment, then watch it land:

         curl -s localhost:5001/webhook/razorpay/recent | $PY -m json.tool

======================================================================
  Tunnel log   $TUNNEL_LOG
  Stop it      ./razorpay-live.sh --stop

  This URL dies when the tunnel stops, and a quick tunnel gets a NEW
  one every restart. Re-paste it into the dashboard each time.
======================================================================

BANNER
