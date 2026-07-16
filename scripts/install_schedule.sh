#!/usr/bin/env bash
#
# Install the AMC Phase 7.1 scheduling: user systemd timers for the DB backups,
# so the irreplaceable data gets mirrored off the WSL disk.
#
# COLLECTOR TIMERS ARE PAUSED (2026-07-16). The ToU audit found every Phase 7.1
# collector source bars AMC's use or moved to manual import (journal.md):
# coin_premiums (APMEX/JM Bullion), consensus (ForexFactory), jm_pgm (Johnson
# Matthey) are barred pending licence; trends is now an operator-run CSV importer.
# So this installer generates the collector units but leaves them DISABLED, and
# enables only the backup timers. Re-enable a collector's timer by hand once its
# source is licensed (systemctl --user enable --now amc-collectors-*.timer) and
# restore its real ExecStart (kept as a comment in the unit).
#
# WSL2 note: user systemd timers only fire while the WSL instance is running,
# and only across logouts if lingering is enabled (this script enables it).
# All timers use Persistent=true, so a run missed while the laptop was off
# fires shortly after the next boot instead of being skipped.
#
# Idempotent: re-running rewrites the unit files, enables backups, and keeps
# the collector timers disabled.
#
# Usage:
#   scripts/install_schedule.sh            # install; enable backups only
#   scripts/install_schedule.sh --status   # just show timer status
#   scripts/install_schedule.sh --uninstall # stop, disable, remove units

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UV_BIN="$(command -v uv || true)"
UNIT_DIR="${XDG_CONFIG_HOME:-$HOME/.config}/systemd/user"
# Collector timers are generated but left DISABLED (sources barred/manual — see
# the header). Only the backup timers are enabled.
COLLECTOR_UNITS=(
  amc-collectors-daily
  amc-collectors-weekly
)
BACKUP_UNITS=(
  amc-backup-tables
  amc-backup-full
)
UNITS=("${COLLECTOR_UNITS[@]}" "${BACKUP_UNITS[@]}")

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

# --- collectors: daily set (PAUSED — see header) ---
# Left DISABLED by the enable step below: consensus (in the daily set) scrapes a
# barred source, so this unit must not auto-run. The real command is preserved in
# a comment for when a licence lands; the active ExecStart is a no-op notice so a
# manual `systemctl start` does nothing harmful.
#   Real (restore when licensed): run_collectors.py --skip jm_pgm  (+ --check-gaps)
cat >"$UNIT_DIR/amc-collectors-daily.service" <<EOF
[Unit]
Description=AMC Phase 7.1 daily collectors (PAUSED 2026-07-16 — sources barred)

[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
ExecStart=/usr/bin/env bash -c 'echo "amc daily collectors are PAUSED (2026-07-16): sources barred by ToU. See journal.md." >&2'
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

# --- collectors: weekly set (PAUSED — see header) ---
# jm_pgm (barred, pending JM licence) was the only remaining weekly collector after
# trends became a manual CSV importer (2026-07-16). Left DISABLED; no-op ExecStart.
#   Real (restore when licensed): run_collectors.py --only jm_pgm
# trends is now run by hand: uv run python -m metals.data.trends <multiTimeline.csv>
cat >"$UNIT_DIR/amc-collectors-weekly.service" <<EOF
[Unit]
Description=AMC Phase 7.1 weekly collectors (PAUSED 2026-07-16 — sources barred/manual)

[Service]
Type=oneshot
WorkingDirectory=$REPO_ROOT
ExecStart=/usr/bin/env bash -c 'echo "amc weekly collectors are PAUSED (2026-07-16): jm_pgm barred, trends is manual. See journal.md." >&2'
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
# Enable only the backup timers. Collector timers are generated but kept disabled
# because every collector source is currently barred/manual (see header).
for u in "${BACKUP_UNITS[@]}"; do
  systemctl --user enable --now "${u}.timer"
done
for u in "${COLLECTOR_UNITS[@]}"; do
  systemctl --user disable --now "${u}.timer" 2>/dev/null || true
done
echo "Collector timers left DISABLED (sources barred/manual, 2026-07-16). Backups enabled."

echo
echo "Installed AMC timers (backups enabled; collectors paused):"
systemctl --user list-timers --all 'amc-*' || true
echo
echo "Manage with:"
echo "  systemctl --user list-timers 'amc-*'          # next/last fire times"
echo "  journalctl --user -u amc-collectors-daily -n 50   # recent run output"
echo "  systemctl --user start amc-backup-full.service    # run a leg now"
echo "  scripts/install_schedule.sh --uninstall           # remove"
