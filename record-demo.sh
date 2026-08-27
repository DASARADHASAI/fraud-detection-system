#!/usr/bin/env bash
#
# Record the backup demonstration video.
#
# The video exists for one moment: the laptop dies, or a service will not come
# up, and you have four minutes left. You play this and narrate over it. That is
# why silent is acceptable and why one unbroken take matters more than polish --
# a cut version invites "what happened in the gap?"
#
#   ./record-demo.sh                 silent (you narrate live over it)
#   ./record-demo.sh --with-audio    capture the microphone too
#
# Press q (or Ctrl-C) in this terminal to stop. Output lands in demo-video/.
#
# FIRST RUN WILL FAIL until Terminal has Screen Recording permission:
#   System Settings -> Privacy & Security -> Screen & System Audio Recording
# Grant it, then fully quit and reopen Terminal. Discover this now, not on the
# day.

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT_DIR="$ROOT/demo-video"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/backup-demo-$(date +%Y%m%d-%H%M).mp4"

command -v ffmpeg >/dev/null || {
  printf '\033[31m  ffmpeg not found:  brew install ffmpeg\033[0m\n' >&2; exit 1; }

# "Capture screen 0" -- confirm the index on this machine rather than assuming:
#   ffmpeg -f avfoundation -list_devices true -i ""
SCREEN="$(ffmpeg -f avfoundation -list_devices true -i "" 2>&1 \
          | sed -n 's/.*\[\([0-9]*\)\] Capture screen 0.*/\1/p' | head -1)"
[[ -n "$SCREEN" ]] || { printf '\033[31m  no screen capture device found\033[0m\n' >&2; exit 1; }

INPUT="$SCREEN"
AUDIO_ARGS=()
if [[ "${1:-}" == "--with-audio" ]]; then
  INPUT="$SCREEN:0"                       # :0 = default microphone
  AUDIO_ARGS=(-c:a aac -b:a 192k)
  printf '\033[36m  recording screen + microphone\033[0m\n'
else
  printf '\033[36m  recording screen only (narrate live over it)\033[0m\n'
fi

cat <<'BANNER'

  Before you start:
    - demo.sh has been run and all three health dots are green
    - the run-sheet is open on a second screen or printed, NOT on camera
    - notifications are off, and the terminal you will Ctrl-C in beat 5 is ready
    - one take. Do not stop for a stumble; keep going.

BANNER

read -r -p "  press Enter to start recording (q to stop) ... "

# -r 30 is plenty for a UI walkthrough and keeps the file small enough to email.
# yuv420p because anything else may not play in PowerPoint or QuickTime.
ffmpeg -hide_banner -loglevel warning -stats \
  -f avfoundation -capture_cursor 1 -framerate 30 -i "$INPUT" \
  -c:v libx264 -preset fast -crf 20 -pix_fmt yuv420p -r 30 \
  "${AUDIO_ARGS[@]}" \
  "$OUT"

if [[ -f "$OUT" ]]; then
  printf '\033[32m\n  saved  %s  (%s)\033[0m\n\n' "$OUT" "$(du -h "$OUT" | cut -f1)"
  printf '  watch it back before you trust it. Check: the decision feed is\n'
  printf '  legible, the OTP modal is on screen long enough to read, and the\n'
  printf '  degraded-mode badge in beat 5 is visible.\n\n'
else
  printf '\033[31m\n  nothing was written -- almost always the Screen Recording\n'
  printf '  permission. See the note at the top of this script.\033[0m\n\n'
fi
