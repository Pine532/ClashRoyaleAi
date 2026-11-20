#!/bin/bash
# Run the phase-1 -> phase-2 training pipeline in the background on macOS.
#
#   tools/mac_training.sh start    start, or resume (same CLASH_WEIGHTS -> "Resumed from ...")
#   tools/mac_training.sh status   process, phase, latest progress and loss lines, non-finite counts
#   tools/mac_training.sh row      one markdown table row for an hourly log (header: row-header)
#   tools/mac_training.sh stop     SIGINT to the trainer's main process only; it saves, then exits
#
# Defaults (export a variable first to override it): the 2.6 Hog Cycle deck,
# checkpoint runs/checkpoints/livesync.pth (runs/ is gitignored whole, the .prev
# copy included), TensorBoard runs/livesync, a save every 250 episodes, and the
# network on MPS where the Mac has it (CLASH_DEVICE=cpu opts out). Stdout goes to runs/livesync_<timestamp>.log.
#
# MPS since 2026-11-19: 376 vs 184 updates/hour from the same checkpoint (rung 3,
# full pool), no non-finite fault in 63 updates, with the rollout's draws made
# on the CPU (rl/sampling.py; MPS's sampler draws zero-probability actions).
# docs/TODO.md L11.
#
# Why this script and not a bare `nohup python ... &`:
# - A job started with `&` from a non-interactive shell inherits SIGINT as
#   IGNORED, and Python then never installs its KeyboardInterrupt handler, so
#   "stop = Ctrl-C saves" silently does nothing. The trainer is exec'd through a
#   one-line shim that resets SIGINT to its default first.
# - `stop` signals the main process only. A terminal Ctrl-C reaches the
#   AsyncVectorEnv workers too; they die first and the save on the way out can
#   be lost (the next resume falls back to the last periodic save).
# - Phase 1 starts phase 2 as a subprocess and exits, so `caffeinate -i` wraps
#   a waiter that outlives phase 1 and holds the no-sleep assertion until
#   phase 2 ends too.
# - Everything below is one { ... } group ending in `exit`, which bash parses
#   whole before running. bash otherwise reads a script as it goes: the `_run`
#   launcher waits hours inside it, then reads on from the same byte offset, so
#   an edit made in place meanwhile would run from the middle of a line.
set -u
{
REPO="$(cd "$(dirname "$0")/.." && pwd)"
PY="$REPO/venv/bin/python"
export CLASH_DECK="${CLASH_DECK:-15,6,25,40,24,72,33,7}"
export CLASH_WEIGHTS="${CLASH_WEIGHTS:-$REPO/runs/checkpoints/livesync.pth}"
export CLASH_LOGDIR="${CLASH_LOGDIR:-$REPO/runs/livesync}"
export CLASH_SAVE_EVERY="${CLASH_SAVE_EVERY:-250}"
if [ -z "${CLASH_DEVICE:-}" ] && [ "${1:-}" = start ] \
   && "$PY" -c 'import sys, torch; sys.exit(0 if torch.backends.mps.is_available() else 1)' 2>/dev/null; then
    export CLASH_DEVICE=mps
fi
export PYTHONDONTWRITEBYTECODE=1
LOGROOT="$(dirname "$CLASH_LOGDIR")"
NAME="$(basename "$CLASH_LOGDIR")"
PIDFILE="$LOGROOT/$NAME.pid"        # the trainer's main process (phase 1, later phase 2)
LOGLINK="$LOGROOT/$NAME.current_log" # path of the current stdout log
SELFPLAY_WEIGHTS="${CLASH_WEIGHTS%.pth}_selfplay.pth"
# The phase-2 process `_run` adopts, matched machine-wide by pgrep. Tests point
# it at a string nothing matches, or their launcher would adopt a real phase-2
# trainer and never exit.
P2_PATTERN="${CLASH_P2_PATTERN:-trainers/train_selfplay.py}"
SHIM='import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execv(sys.argv[1], sys.argv[1:])'

alive() { [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }
# Lines in the current log matching $1 (0 when there is no log).
count() { local n; n=$(grep -c -- "$1" "$LOG" 2>/dev/null); echo "${n:-0}"; }
# Both must stay 0 (docs/TODO.md L11): updates that dropped a minibatch to a
# non-finite gradient, and sampled actions whose log-prob was not finite.
SKIP_LINE="non-finite minibatch"
SAMPLE_LINE="non-finite log-prob for a sampled action"

case "${1:-}" in
start)
    if alive; then echo "already running: pid $(cat "$PIDFILE")"; exit 1; fi
    mkdir -p "$LOGROOT" "$(dirname "$CLASH_WEIGHTS")"
    LOG="$LOGROOT/${NAME}_$(date +%Y%m%d-%H%M%S).log"
    echo "$LOG" > "$LOGLINK"
    cd "$REPO" || exit 1
    nohup caffeinate -i /bin/bash "$0" _run >"$LOG" 2>&1 &
    sleep 3
    echo "started: trainer pid $(cat "$PIDFILE" 2>/dev/null), log $LOG"
    echo "checkpoint $CLASH_WEIGHTS, TensorBoard $CLASH_LOGDIR"
    ;;
_run)  # internal: runs under caffeinate
    if [ -f "$SELFPLAY_WEIGHTS" ]; then
        echo ">>> mac_training.sh: $SELFPLAY_WEIGHTS exists, resuming phase 2 directly"
        "$PY" -c "$SHIM" "$PY" -u python_ai/trainers/train_selfplay.py &
    else
        "$PY" -c "$SHIM" "$PY" -u -m python_ai.trainers.train &
    fi
    echo $! > "$PIDFILE"
    wait $!
    echo ">>> mac_training.sh: trainer exited with code $? at $(date)"
    # Phase 2, if phase 1 launched it, outlives phase 1: track it and keep
    # caffeinate's assertion alive until it ends.
    while p2=$(pgrep -f -- "$P2_PATTERN" | head -1) && [ -n "$p2" ]; do
        echo "$p2" > "$PIDFILE"; sleep 60
    done
    ;;
stop)
    if ! alive; then echo "not running"; exit 1; fi
    pid=$(cat "$PIDFILE")
    kill -INT "$pid"
    echo "SIGINT sent to $pid; waiting for the checkpoint save (up to 3 min)..."
    for _ in $(seq 1 180); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
    if kill -0 "$pid" 2>/dev/null; then echo "still running after 3 min; check the log"; exit 1; fi
    LOG="$(cat "$LOGLINK" 2>/dev/null)"
    grep -h "Checkpoint saved\|Interrupted" "$LOG" "$LOGROOT/training_selfplay_pfsp.log" 2>/dev/null | tail -2
    ;;
status)
    LOG="$(cat "$LOGLINK" 2>/dev/null)"
    if alive; then
        pid=$(cat "$PIDFILE"); echo "running: pid $pid ($(ps -o etime= -p "$pid" | tr -d ' ') elapsed)"
        ps -o command= -p "$pid" | grep -q train_selfplay && echo "phase 2 (PFSP self-play), log $LOGROOT/training_selfplay_pfsp.log"
    else
        echo "not running"
    fi
    echo "log: $LOG"
    grep -h "Resumed from" "$LOG" 2>/dev/null | tail -1
    grep -h "^Episodes: " "$LOG" 2>/dev/null | tail -1
    grep -h "Update @" "$LOG" 2>/dev/null | tail -1
    grep -h "Checkpoint saved" "$LOG" 2>/dev/null | tail -1
    grep -h "FLOOR\|ALARM\|Traceback\|Error" "$LOG" 2>/dev/null | tail -3
    skips=$(count "$SKIP_LINE"); samples=$(count "$SAMPLE_LINE")
    echo "non-finite: $skips update(s) with dropped minibatches, $samples bad sample(s) in this log"
    if [ "$skips" != 0 ] || [ "$samples" != 0 ]; then
        echo "WARNING: non-finite counts must be 0 (docs/TODO.md L11); grep the log for [WARN]"
    fi
    ;;
row-header)
    echo "| Time | Trainer | Last progress line | Advances | Demotions | [FLOOR] | Skip updates | Bad samples | Non-finite |"
    echo "|---|---|---|---|---|---|---|---|---|"
    ;;
row)
    LOG="$(cat "$LOGLINK" 2>/dev/null)"
    state="stopped"; alive && state="running"
    # The progress line has pipes of its own; escaped, or they split the cells.
    last="$(grep -h "^Episodes: " "$LOG" 2>/dev/null | tail -1 | cut -c1-80 | sed 's/|/\\|/g')"
    skips=$(count "$SKIP_LINE"); samples=$(count "$SAMPLE_LINE")
    flag="ok"; { [ "$skips" != 0 ] || [ "$samples" != 0 ]; } && flag="**WARNING**"
    echo "| $(date +%H:%M) | $state | \`$last\` | $(count "Curriculum advanced") | $(count DEMOTED) | $(count "\[FLOOR\]") | $skips | $samples | $flag |"
    ;;
*)
    sed -n 2,12p "$0"; exit 2 ;;
esac
exit
}
