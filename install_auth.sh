#!/usr/bin/env bash
# Installs the Havelsan captive-portal link keeper (same as the central box):
#  - script  -> ~/.local/bin/havelsan-login.sh
#  - cron    -> every 30 min overnight (19:00-07:00), hourly in the day
# Usage:
#   ./install_auth.sh <username> <password>   # installs + writes creds
#   ./install_auth.sh                          # installs only (creds later)
# Idempotent: safe to re-run.
set -e
mkdir -p ~/.local/bin ~/.config
install -m 755 "$(dirname "$0")/auth/havelsan-login.sh" ~/.local/bin/havelsan-login.sh

if [ -n "$1" ]; then
  [ -n "$2" ] || { echo "Password also required: ./install_auth.sh <username> <password>"; exit 1; }
  printf 'username=%s\npassword=%s\n' "$1" "$2" > ~/.config/havelsan-creds
  chmod 600 ~/.config/havelsan-creds
  echo "Credentials written to ~/.config/havelsan-creds (mode 600)."
else
  if [ ! -f ~/.config/havelsan-creds ]; then
    echo "NOTE: no credentials yet. Either re-run:  ./install_auth.sh <username> <password>"
    echo "      or create ~/.config/havelsan-creds with username=/password= lines"
  else
    echo "Credentials already present, leaving untouched."
  fi
fi

# "|| true": with set -e, an empty crontab (grep -v matches nothing) would
# otherwise end the subshell before the echo lines and install an empty crontab
( { crontab -l 2>/dev/null | grep -v "havelsan-login.sh"; } || true
  echo "*/30 0-6,19-23 * * * $HOME/.local/bin/havelsan-login.sh"
  echo "0 7-18 * * * $HOME/.local/bin/havelsan-login.sh"
) | crontab -

echo "Cron now:"
crontab -l | grep havelsan
echo ""
echo "If re-auth ever fails with 'token expired', open the portal page in a"
echo "browser once, copy the fresh token URL, and set PORTAL_URL in the script."
