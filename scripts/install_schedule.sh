#!/usr/bin/env bash
#
# Install the AMC Phase 7.1 scheduling: user systemd timers that run the
# collectors and the DB backups, so the non-backfillable daily captures stop
# bleeding and the irreplaceable data gets mirrored off the WSL disk.
#
# WSL2 note: user systemd timers only fire while the WSL instance is running,
# and only across logouts if lingering is enabled (this script enables it).
# All timers use Persistent=true, so a run missed while the laptop was off
# fires shortly after the next boot instead of being skipped.
#
# Idempotent: re-running rewrites the unit files and re-enables the timers.
#
# Usage:
#   scripts/install_schedule.sh            # install + enable + start
#   scripts/install_schedule.sh --status   # just show timer status
#   scripts/install_schedule.sh --uninstall # stop, disable, remove units

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="$(command -v uv || true)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
UNITS=(
  amc-collectors-daily
  amc-collectors-weekly
  amc-backup-tables
  amc-backup-full
)

if [[ "${1:-}" == "--status" ]]; then
  systemctl --user list-timers --all 'amc-*' || true
  exit 0
fi

if [[ "${1:-}" == "--uninstall" ]]; then
  for u in "${UNITS[@]}"; do
    systemctl --user disable --now "${u}.timer" 2>/dev/null || true
    rm -f "$UNIT_DIR/${u}.timer" "$UNIT_DIR/${u}.service"
  done
  systemctl --user daemon-reload
  echo "Removed AMC timers and services."
  exit 0
fi

if [[ -z "$UV_BIN" ]]; then
  echo "ERROR: 'uv' not found on PATH. Install uv or fix PATH, then re-run." >&2
  exit 1
fi

mkdir -p "$UNIT_DIR"

# --- collectors: daily set (coin_premiums, cme_daily, consensus) + gap audit ---
cat >"$UNIT_DIR/amc-collectors-daily.service" <<EOF
[Unit]
Description=AMC Phase 7.1 daily collectors + staleness audit

[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
ExecStart=$UV_BIN run python scripts/run_collectors.py --skip trends,jm_pgm
ExecStartPost=$UV_BIN run python scripts/run_collectors.py --check-gaps
EOF

cat >"$UNIT_DIR/amc-collectors-daily.timer" <<EOF
[Unit]
Description=Run AMC daily collectors every morning

[Timer]
OnCalendar=*-*-* 03:30:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF

# --- collectors: weekly set (trends, jm_pgm) ---
cat >"$UNIT_DIR/amc-collectors-weekly.service" <<EOF
[Unit]
Description=AMC Phase 7.1 weekly collectors (trends, jm_pgm)

[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
ExecStart=$UV_BIN run python scripts/run_collectors.py --only trends,jm_pgm
EOF

cat >"$UNIT_DIR/amc-collectors-weekly.timer" <<EOF
[Unit]
Description=Run AMC weekly collectors on Mondays

[Timer]
OnCalendar=Mon *-*-* 03:40:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF

# --- backup: daily small-table Parquet snapshot ---
cat >"$UNIT_DIR/amc-backup-tables.service" <<EOF
[Unit]
Description=AMC daily capture/ledger table backup (Parquet)

[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
ExecStart=$UV_BIN run python scripts/backup_db.py --tables
EOF

cat >"$UNIT_DIR/amc-backup-tables.timer" <<EOF
[Unit]
Description=Back up AMC capture tables every morning

[Timer]
OnCalendar=*-*-* 04:00:00
Persistent=true
RandomizedDelaySec=300

[Install]
WantedBy=timers.target
EOF

# --- backup: weekly full DB snapshot ---
cat >"$UNIT_DIR/amc-backup-full.service" <<EOF
[Unit]
Description=AMC weekly full DuckDB snapshot

[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
ExecStart=$UV_BIN run python scripts/backup_db.py --full
EOF

cat >"$UNIT_DIR/amc-backup-full.timer" <<EOF
[Unit]
Description=Snapshot the whole AMC DuckDB weekly

[Timer]
OnCalendar=Sun *-*-* 04:15:00
Persistent=true
RandomizedDelaySec=600

[Install]
WantedBy=timers.target
EOF

# Enable lingering so the timers fire without an interactive login session.
if ! loginctl show-user "$USER" 2>/dev/null | grep -q "Linger=yes"; then
  if loginctl enable-linger "$USER" 2>/dev/null; then
    echo "Enabled linger for $USER."
  else
    echo "NOTE: could not enable linger unprivileged. Run once:" >&2
    echo "      sudo loginctl enable-linger $USER" >&2
  fi
fi

systemctl --user daemon-reload
for u in "${UNITS[@]}"; do
  systemctl --user enable --now "${u}.timer"
done

echo
echo "Installed and started AMC timers:"
systemctl --user list-timers --all 'amc-*' || true
echo
echo "Manage with:"
echo "  systemctl --user list-timers 'amc-*'          # next/last fire times"
echo "  journalctl --user -u amc-collectors-daily -n 50   # recent run output"
echo "  systemctl --user start amc-backup-full.service    # run a leg now"
echo "  scripts/install_schedule.sh --uninstall           # remove"
