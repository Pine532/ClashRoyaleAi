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
# The restart is `stop` (SIGINT, saves), a copy of the checkpoint the stop
# wrote, then `start` without CLASH_PHASE1_DECK_POOL. The copy is the last
# mirror-only state, kept as <weights>.pre_pool_switch_<time>: the saves after
# the switch replace the live file, and on 2026-11-19 they replaced the only
# copy. If the copy fails, the run restarts unchanged and nothing switches. The
# trainer reports the changed setting at the top of the new log and continues.
# Any other exported CLASH_* is passed through unchanged.
set -u
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNG="${1:-4}"
WEIGHTS="${CLASH_WEIGHTS:-$REPO/runs/checkpoints/livesync.pth}"  # mac_training.sh's default
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
        KEPT="$WEIGHTS.pre_pool_switch_$(date +%Y%m%d-%H%M%S)"
        if ! cp -p "$WEIGHTS" "$KEPT"; then
            echo "$(date): could not copy $WEIGHTS; restarting unchanged, not switching"
            "$REPO/tools/mac_training.sh" start
            exit 1
        fi
        echo "$(date): kept the pre-switch checkpoint as $KEPT"
        env -u CLASH_PHASE1_DECK_POOL "$REPO/tools/mac_training.sh" start
        exit $?
    fi
    sleep 300
done
