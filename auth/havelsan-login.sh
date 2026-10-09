#!/usr/bin/env bash
# Havelsan overnight link keeper. Runs every 30 min (19:00-07:00) via cron.
#
# Each run is a CHECK:
#   - probe the connection
#   - internet UP   -> do nothing (skip)
#   - internet DOWN -> re-authenticate using the captive-portal link
#
# Notifications are transition-based (no spam):
#   - UP -> DOWN : "Havelsan link DOWN"
#   - DOWN -> UP : "Havelsan link RESTORED"
#   - same state repeated : silent
set -u

CRED_FILE="$HOME/.config/havelsan-creds"
LOG_FILE="$HOME/.config/havelsan-login.log"
STATE_FILE="$HOME/.config/havelsan-state"
UA="Mozilla/5.0 (Windows NT 10.0; Win64; x64)"

# --- Connectivity probe (env-overridable; change host if your network blocks it) ---
PROBE_URL="${PROBE_URL:-http://www.msftconnecttest.com/connecttest.txt}"
UP_MARKER="${UP_MARKER:-Microsoft Connect Test}"

# --- Captive-portal login link (re-copy fresh token URL if it ever expires) ---
PORTAL_URL="${PORTAL_URL:-https://kullanicidogrulama.havelsan.com.tr:6082/php/uid.php?vsys=1&rule=7&token=MFWxYN4SS1XXILEJx4gaIC_iN9o=&url=http://firefox-portal-detection.com%2fgenerate_204}"

log() { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" >> "$LOG_FILE"; }

notify() { # $1=title $2=body  (works from cron on a Wayland/X session)
  local uid; uid=$(id -u)
  export XDG_RUNTIME_DIR="/run/user/${uid}"
  export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/${uid}/bus"
  export DISPLAY="${DISPLAY:-:0}"
  export WAYLAND_DISPLAY="${WAYLAND_DISPLAY:-wayland-0}"
  command -v notify-send >/dev/null 2>&1 && \
    notify-send -u critical -a "Havelsan Login" "$1" "$2" 2>/dev/null || true
}

prev=$(cat "$STATE_FILE" 2>/dev/null || echo unknown)

# 1) CHECK: is the internet up?
probe_body=$(mktemp)
pcode=$(curl -s --max-time 10 -A "$UA" -o "$probe_body" -w '%{http_code}' "$PROBE_URL" 2>/dev/null)
if [ "$pcode" = "200" ] && grep -qF "$UP_MARKER" "$probe_body"; then
  conn="UP"
else
  conn="DOWN"
fi
rm -f "$probe_body"

if [ "$conn" = "UP" ]; then
  # 2a) Internet up -> do nothing
  log "SKIP: internet up (probe http=$pcode)"
  if [ "$prev" = "DOWN" ]; then
    notify "Havelsan link RESTORED" "Connection is up again."
  fi
else
  # 2b) Internet down -> re-authenticate via the portal link
  username=$(grep '^username=' "$CRED_FILE" 2>/dev/null | head -1 | cut -d= -f2-)
  password=$(grep '^password=' "$CRED_FILE" 2>/dev/null | head -1 | cut -d= -f2-)

  if [ ! -f "$CRED_FILE" ] || [ -z "${username:-}" ] || [ -z "${password:-}" ]; then
    result="no credentials (check $CRED_FILE)"
  else
    # Credentials go to curl through private temp files (name@file), not as
    # command-line arguments, which any local user could read with `ps`.
    secrets=$(umask 077; mktemp -d)
    trap 'rm -rf "$secrets"' EXIT
    printf '%s' "$username" | sed 's/\\/\\\\/g' > "$secrets/escape_user"
    printf '%s' "$username" > "$secrets/user"
    printf '%s' "$password" > "$secrets/passwd"
    cj="$secrets/cookies"
    lcode=$(curl -s --max-time 30 -c "$cj" -b "$cj" -A "$UA" \
      --data-urlencode "inputStr=" \
      --data-urlencode "escapeUser@$secrets/escape_user" \
      --data-urlencode "preauthid=" \
      --data-urlencode "user@$secrets/user" \
      --data-urlencode "passwd@$secrets/passwd" \
      --data-urlencode 'ok=Login' \
      -o /dev/null -w '%{http_code}' "$PORTAL_URL")
    rm -rf "$secrets"
    if [ "$lcode" = "302" ]; then
      result="re-authenticated (login 302)"
    else
      result="login HTTP $lcode (token expired or bad creds - internet probe http=$pcode)"
    fi
  fi

  log "DOWN: internet down -> $result"
  if [ "$prev" != "DOWN" ]; then
    notify "Havelsan link DOWN" "$result"
  fi
fi

echo "$conn" > "$STATE_FILE"
exit $([ "$conn" = "UP" ] && echo 0 || echo 1)
