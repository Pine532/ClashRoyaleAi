#!/bin/bash
# Watch a mirror-mode run (CLASH_PHASE1_DECK_POOL=0) and turn the 16-deck pool on
# once it reaches a rung, by restarting it through tools/mac_training.sh.
#
#   nohup tools/mac_training_pool_switch.sh [rung] >> runs/livesync_pool_switch.log 2>&1 &
#
# Why: a fresh net cannot learn against the meta-deck pool (the documented cold
# start; docs/live_sync/rung0_stall_diagnosis.md), but climbs in the mirror. The
# pool is the point of phase 1 (deck variety), so the run starts in the mirror and
# switches once the lower rungs are behind it. The switch must happen while the
# run is still in the "mirror" phase: that phase ends at rung 8
# (PHASE2_MIN_CURRICULUM_STAGE), so the default rung is 4.
#
# The restart is `stop` (SIGINT, saves) then `start` without
# CLASH_PHASE1_DECK_POOL. The trainer reports the changed setting at the top of
# the new log and continues. Any other exported CLASH_* is passed through
# unchanged.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNG="${1:-4}"
LOGDIR="${CLASH_LOGDIR:-$REPO/runs/livesync}"
LOGLINK="$(dirname "$LOGDIR")/$(basename "$LOGDIR").current_log"
echo "$(date): watching $LOGLINK for rung >= $RUNG in the mirror phase"
while true; do
    LOG="$(cat "$LOGLINK" 2>/dev/null)"
    line="$(grep '^Episodes: ' "$LOG" 2>/dev/null | tail -1)"
    stage="$(echo "$line" | sed -n 's/.*Stage: \([0-9]*\).*/\1/p')"
    phase="$(echo "$line" | sed -n 's/.*Phase: \([a-z_]*\).*/\1/p')"
    if [ -n "$phase" ] && [ "$phase" != "mirror" ]; then
        echo "$(date): the run left the mirror phase ($phase) before rung $RUNG; not switching"
        exit 1
    fi
    if [ -n "$stage" ] && [ "$stage" -ge "$RUNG" ]; then
        echo "$(date): rung $stage reached in the mirror phase; switching the deck pool on"
        "$REPO/tools/mac_training.sh" stop || exit 1
        sleep 5
        env -u CLASH_PHASE1_DECK_POOL "$REPO/tools/mac_training.sh" start
        exit $?
    fi
    sleep 300
done
