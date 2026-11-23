"""The tracker A/B harness: run both arms to the update budget, freeze copies, judge by the rule.

    python -m python_ai.tools.run_tracker_ab --root runs/ab_tracker [--dry-run]
    python -m python_ai.tools.run_tracker_ab --root runs/ab_tracker --extend

The arms come from `setup_tracker_ab.py` (ROOT/ab_manifest.json). The rule is the
user's (docs/DECISIONS.md, "2026-11-22 (late)"):

  main measure   the tracker arm T against the control arm C, 400 games, sides swapped
                 (eval/arm_h2h.py), each net with its own observation layout
  adopt          T scores >= 0.55 AND its rung-10 score (bench.py --rung 10, 200 games, on
                 each arm's frozen copy) is no more than 10/200 below C's
  inconclusive   0.50 <= score < 0.55: extend both arms ONCE by 2,000 updates (--extend)
                 and judge again
  reject         < 0.50
  >= 0.55 with the guard failed: not adopted.
  A second inconclusive after the extension: not adopted. (PROPOSED, not yet confirmed
  by the user: DECISIONS.md lists it as open.)

Running an arm
--------------
`nice -n N python -c <SIGINT shim> python -u -m python_ai.trainers.train_selfplay`, cwd and
PYTHONPATH the worktree (--src), the arm's env file as its environment, stdout and stderr to
ROOT/<arm>/train_s<stage>.log. The shim resets SIGINT to its default first: a job started from a
non-interactive shell inherits SIGINT as ignored and would never save on it (tools/mac_training.sh).
SIGINT goes to the trainer's main process only; it saves, then exits.

The update counter
------------------
A PPO update is one pass of BaseTrainer.run (collect, update, log_update), and log_update prints
exactly one line `  >> Update @ ep N | ...` for each. That LOG LINE is the counter: no checkpoint
field counts updates (the optimizer's step is per minibatch and varies with skipped ones), and the
episode count is not a clock for updates. Counted per launch (segment) from the launch's log;
the totals across --extend segments are kept in ROOT/ab_state.json.
The budget is met at the first `>>> Checkpoint saved to ... (episode N)` line printed after the
budget-th update line (the periodic save follows its update's log line), so each arm stops at its
own first save at or after the budget. The frozen copy is that save's generation
(`<weights>_history/epN.pth`, a hard link kept by keep_in_history, exact even if the trainer saves
again before this process copies) or, without history, the live file; it is checked against N.
Then SIGINT; the arm's live checkpoint is a few updates newer (the interrupt's own save), and
--extend resumes from it.

Judging
-------
arm_h2h (T vs C, --h2h-games) and bench.py --rung 10 (--bench-games) on each frozen copy, with
SRC = --src and that arm's CLASH_OPP_TRACKER. bench.py and h2h.py of runs/tools_scratch are not
modified. Writes ROOT/judge/s<stage>/..., ROOT/verdict_s<stage>.json and ROOT/verdict.json.
--dry-run prints every command and runs nothing.
"""
import argparse
import dataclasses
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from typing import Optional

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

UPDATE_RE = re.compile(r"^\s*>> Update @ ep (\d+)\b")
SAVE_RE = re.compile(r"^>>> Checkpoint saved to (.+) \(episode (\d+)\)\s*$")
#: Resets SIGINT to its default, then replaces itself with the trainer (tools/mac_training.sh).
SHIM = "import os, signal, sys; signal.signal(signal.SIGINT, signal.SIG_DFL); os.execv(sys.argv[1], sys.argv[1:])"
TRAINER_MODULE = "python_ai.trainers.train_selfplay"
DEFAULT_BENCH = "/Users/owenzhang/ClashRoyaleAi/runs/tools_scratch/bench.py"

ADOPT_SCORE = (11, 20)          # 0.55 as a fraction
REJECT_SCORE = (1, 2)           # 0.50
GUARD = (1, 20)                 # 10/200: the tracker arm's rung-10 wins may trail the control's by this share


@dataclasses.dataclass
class Args:
    root: str
    src: str = _REPO_ROOT
    python: str = sys.executable
    nice: int = 10
    budget: int = 2000
    h2h_games: int = 400
    bench_games: int = 200
    bench_rung: int = 10
    bench_script: str = DEFAULT_BENCH
    skip_bench: bool = False
    poll: float = 5.0
    stop_wait: float = 600.0
    extend: bool = False
    dry_run: bool = False
    trainer_module: str = TRAINER_MODULE
    h2h_module: str = "python_ai.eval.arm_h2h"


# ---- the update counter -------------------------------------------------------
class UpdateCounter:
    """Counts `>> Update @` lines; `feed` returns the save that meets the budget, once."""

    def __init__(self, budget):
        self.budget, self.updates, self.fired = int(budget), 0, False

    def feed(self, line) -> Optional[dict]:
        if UPDATE_RE.match(line):
            self.updates += 1
            return None
        m = SAVE_RE.match(line)
        if m and not self.fired and self.updates >= self.budget:
            self.fired = True
            return {"path": m.group(1), "episode": int(m.group(2)), "updates": self.updates}
        return None


class LogTail:
    """Whole new lines of a growing log file."""

    def __init__(self, path):
        self.path, self.offset, self._partial = path, 0, ""

    def read_lines(self):
        if not os.path.exists(self.path):
            return []
        with open(self.path, "r", errors="replace") as f:
            f.seek(self.offset)
            chunk = f.read()
            self.offset = f.tell()
        text = self._partial + chunk
        lines = text.split("\n")
        self._partial = lines.pop()
        return lines


# ---- the verdict rule ---------------------------------------------------------
def decide(h2h, r10_t, r10_c, stage):
    """The user's rule. All comparisons in integers (half points), so 0.55 is exactly 220/400.

    h2h: {wins, draws, n} for T against C. r10_*: {wins, n} on bench.py --rung 10, or None.
    """
    n = int(h2h["n"])
    twice = 2 * int(h2h["wins"]) + int(h2h["draws"])               # 2 x the score times n
    score = twice / (2 * n)
    ge_adopt = twice * ADOPT_SCORE[1] >= 2 * n * ADOPT_SCORE[0]
    below_reject = twice * REJECT_SCORE[1] < 2 * n * REJECT_SCORE[0]
    out = {"stage": stage, "score": score, "h2h": dict(h2h), "rung10_tracker": r10_t, "rung10_control": r10_c,
           "rule": "adopt: score >= 0.55 and rung-10 T >= C - 10/200; 0.50 <= score < 0.55 inconclusive "
                   "(extend once by 2,000 updates); < 0.50 reject"}
    if below_reject:
        return dict(out, decision="reject", next="stop", reason=f"score {score:.4f} < 0.50")
    if not ge_adopt:
        if stage >= 2:
            return dict(out, decision="not_adopted", next="stop",
                        reason=f"score {score:.4f} in [0.50, 0.55) after the one extension: a second "
                               "inconclusive is not adopted (proposed rule, unconfirmed by the user)")
        return dict(out, decision="inconclusive", next="extend",
                    reason=f"score {score:.4f} in [0.50, 0.55): extend both arms once by 2,000 updates")
    if r10_t is None or r10_c is None:
        return dict(out, decision="incomplete", next="run the rung-10 benchmark",
                    reason=f"score {score:.4f} >= 0.55 but the rung-10 guard has no data")
    deficit = int(r10_c["wins"]) - int(r10_t["wins"])
    out["guard_deficit_wins"] = deficit
    if deficit * GUARD[1] > int(r10_t["n"]) * GUARD[0]:
        return dict(out, decision="not_adopted", next="stop",
                    reason=f"score {score:.4f} >= 0.55 but the rung-10 guard failed: T trails C by "
                           f"{deficit}/{r10_t['n']} (limit {GUARD[0]}/{GUARD[1]} of the games)")
    return dict(out, decision="adopt", next="adopt",
                reason=f"score {score:.4f} >= 0.55 and rung-10 T trails C by {deficit}/{r10_t['n']} "
                       "(within the guard)")


# ---- commands -----------------------------------------------------------------
def read_env_file(path):
    env = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


def _base_env(args, extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([args.src] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep)
                                                       if p and p != args.src])
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra or {})
    return env


def arm_launch(args, arm):
    """(command, environment, cwd) of one arm's trainer."""
    env = _base_env(args, read_env_file(arm["env"]))
    cmd = ["nice", "-n", str(args.nice), args.python, "-c", SHIM, args.python, "-u", "-m", args.trainer_module]
    return cmd, env, args.src


def h2h_command(args, t_ckpt, c_ckpt, deck, work, out_md=None):
    cmd = ["nice", "-n", "19", args.python, "-u", "-m", args.h2h_module,
           "--a", t_ckpt, "--a-mode", "on", "--b", c_ckpt, "--b-mode", "zero",
           "--deck", deck, "--n", str(args.h2h_games), "--work", work]
    if out_md:
        cmd += ["--out-md", out_md]
    return cmd, _base_env(args, {"CLASH_DEVICE": "cpu"}), args.src


def bench_command(args, arm, ckpt, work, deck, games, rung, script):
    """bench.py SRC CKPT DECK N WORK --rung R, with the arm's CLASH_OPP_TRACKER."""
    cmd = ["nice", "-n", "19", args.python, "-u", script, args.src, ckpt, deck, str(games), work,
           "--rung", str(rung)]
    return cmd, _base_env(args, {"CLASH_OPP_TRACKER": arm["mode"]}), args.src


def fmt(cmd, env=None, cwd=None, keys=()):
    pre = " ".join(f"{k}={env[k]}" for k in keys if env and k in env)
    return (pre + " " if pre else "") + " ".join(cmd) + (f"   (cwd {cwd})" if cwd else "")


# ---- state --------------------------------------------------------------------
def load_state(root):
    p = os.path.join(root, "ab_state.json")
    return json.load(open(p)) if os.path.exists(p) else {"arms": {}, "stages": {}}


def save_state(root, st):
    tmp = os.path.join(root, ".ab_state.tmp")
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, os.path.join(root, "ab_state.json"))


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def frozen_source(weights_path, episode):
    """The exact generation of the save at `episode`: the history link, else the live file."""
    from python_ai.rl.checkpointing import history_dir
    gen = os.path.join(history_dir(weights_path), f"ep{episode:08d}.pth")
    return gen if os.path.exists(gen) else weights_path


def freeze(root, arm_name, hit, stage, updates_total):
    import torch
    src = frozen_source(hit["path"], hit["episode"])
    d = os.path.join(root, "frozen")
    os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, f"{arm_name}_s{stage}_u{updates_total}.pth")
    shutil.copy2(src, dst)
    ck = torch.load(dst, map_location="cpu", weights_only=False)
    got = int(ck.get("episodes_completed", -1))
    rec = {"path": dst, "source": src, "save_episode": hit["episode"], "frozen_episode": got,
           "updates_at_save": updates_total, "exact": got == hit["episode"]}
    if not rec["exact"]:
        print(f"!! [{arm_name}] the frozen copy is episode {got}, the save was episode {hit['episode']}: "
              "another save intervened; the copy is a later valid save", flush=True)
    return rec


# ---- running the arms ----------------------------------------------------------
def _stop_live(live, state, root, wait):
    """SIGINT every arm still running (it saves, then exits) and wait for it, so a failure of one
    arm, or Ctrl-C of the harness, never leaves a trainer behind (they run in their own session)."""
    for name, L in live.items():
        if L["proc"].poll() is None and not L["stopped_at"]:      # not one already told to save and exit
            print(f"[{name}] stopping pid {L['proc'].pid} (SIGINT)", flush=True)
            try:
                L["proc"].send_signal(signal.SIGINT)
            except OSError:
                pass
    deadline = time.time() + wait
    for name, L in live.items():
        try:
            L["proc"].wait(timeout=max(0.0, deadline - time.time()))
        except subprocess.TimeoutExpired:
            print(f"!! [{name}] pid {L['proc'].pid} did not exit after SIGINT", flush=True)
            continue
        L["out"].close()
        state["arms"][name]["pid"] = None
    save_state(root, state)


def run_arms(args, manifest, state, stage):
    """Launch every arm that has not frozen this stage, watch the logs, freeze, SIGINT."""
    root = manifest["root"]
    key = f"s{stage}"
    live = {}
    for name, arm in manifest["arms"].items():
        rec = state["arms"].setdefault(name, {})
        if key in rec.get("frozen", {}):
            print(f"[{name}] stage {stage} already frozen: {rec['frozen'][key]['path']}")
            continue
        if rec.get("pid") and pid_alive(rec["pid"]):
            raise SystemExit(f"!! arm {name} is still running as pid {rec['pid']}; stop it first")
        cmd, env, cwd = arm_launch(args, arm)
        log = os.path.join(arm["dir"], f"train_{key}.log")
        if os.path.exists(log):
            raise SystemExit(f"!! {log} exists; refusing to overwrite a previous launch's log")
        out = open(log, "w")
        proc = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=out, stderr=subprocess.STDOUT,
                                start_new_session=True)
        rec.update(pid=proc.pid, log=log, mode=arm["mode"])
        save_state(root, state)
        print(f"[{name}] launched pid {proc.pid} ({arm['mode']}), log {log}", flush=True)
        live[name] = dict(proc=proc, out=out, tail=LogTail(log), counter=UpdateCounter(args.budget),
                          hit=None, stopped_at=None, log=log)
    try:
        _watch(args, state, root, stage, key, live)
    except BaseException:
        _stop_live(live, state, root, args.stop_wait)
        raise


def _watch(args, state, root, stage, key, live):
    while live:
        time.sleep(args.poll)
        for name in list(live):
            L = live[name]
            for line in L["tail"].read_lines():
                hit = L["counter"].feed(line)
                if hit and L["hit"] is None:
                    L["hit"] = hit
                    total = state["arms"][name].get("updates_total", 0) + hit["updates"]
                    state["arms"][name].setdefault("frozen", {})[key] = freeze(
                        root, name, hit, stage, total)
                    save_state(root, state)
                    print(f"[{name}] budget met ({hit['updates']} updates, save at episode "
                          f"{hit['episode']}); frozen {state['arms'][name]['frozen'][key]['path']}; "
                          "SIGINT", flush=True)
                    os.kill(L["proc"].pid, signal.SIGINT)
                    L["stopped_at"] = time.time()
            rc = L["proc"].poll()
            if rc is not None:
                for line in L["tail"].read_lines():
                    L["counter"].feed(line)
                L["out"].close()
                rec = state["arms"][name]
                rec["pid"] = None
                if L["hit"] is None:
                    raise SystemExit(f"!! arm {name} exited (rc {rc}) before its budget "
                                     f"({L['counter'].updates}/{args.budget} updates); see {L['log']}")
                rec["updates_total"] = rec.get("updates_total", 0) + L["counter"].updates
                rec.setdefault("segments", []).append({"stage": stage, "log": L["log"], "rc": rc,
                                                       "updates": L["counter"].updates})
                save_state(root, state)
                print(f"[{name}] stopped (rc {rc}); {L['counter'].updates} updates this segment, "
                      f"{rec['updates_total']} in all", flush=True)
                del live[name]
            elif L["stopped_at"] and time.time() - L["stopped_at"] > args.stop_wait:
                raise SystemExit(f"!! arm {name} (pid {L['proc'].pid}) did not exit {args.stop_wait:.0f} s "
                                 "after SIGINT; check its log")


# ---- judging --------------------------------------------------------------------
def _run(cmd, env, cwd, log):
    print("  $ " + fmt(cmd, env, cwd, ("CLASH_OPP_TRACKER",)), flush=True)
    with open(log, "w") as f:
        rc = subprocess.run(cmd, env=env, cwd=cwd, stdout=f, stderr=subprocess.STDOUT).returncode
    if rc:
        raise SystemExit(f"!! {cmd[4:7]} failed (rc {rc}); see {log}")


def _reuse(path, n):
    """A finished result of the same size from an earlier judge() of this stage, else None.

    Lets a re-run (e.g. rung-10 data arriving after --skip-bench) compute the verdict again
    without replaying the 400 head-to-head games."""
    if not os.path.exists(path):
        return None
    try:
        res = json.load(open(path))
    except ValueError:
        return None
    if int(res.get("n", -1)) != int(n):
        return None
    print(f"  reusing {path}", flush=True)
    return res


def judge(args, manifest, state, stage):
    root = manifest["root"]
    key = f"s{stage}"
    deck = read_env_file(manifest["arms"]["T"]["env"])["CLASH_DECK"]
    frozen = {n: state["arms"][n]["frozen"][key]["path"] for n in manifest["arms"]}
    jd = os.path.join(root, "judge", key)
    os.makedirs(jd, exist_ok=True)
    h2h_dir = os.path.join(jd, "h2h")
    cmd, env, cwd = h2h_command(args, frozen["T"], frozen["C"], deck, h2h_dir,
                                os.path.join(root, "H2H.md"))
    h2h = _reuse(os.path.join(h2h_dir, "result.json"), args.h2h_games)
    if h2h is None:
        _run(cmd, env, cwd, os.path.join(jd, "h2h.log"))
        h2h = json.load(open(os.path.join(h2h_dir, "result.json")))
    r10 = {}
    if not args.skip_bench:
        for name, arm in manifest["arms"].items():
            work = os.path.join(jd, f"bench_{name}")
            cmd, env, cwd = bench_command(args, arm, frozen[name], work, deck, args.bench_games,
                                          args.bench_rung, args.bench_script)
            res = _reuse(os.path.join(work, "result.json"), args.bench_games)
            if res is None:
                _run(cmd, env, cwd, os.path.join(jd, f"bench_{name}.log"))
                res = json.load(open(os.path.join(work, "result.json")))
            r10[name] = {"wins": int(res["wins"]), "n": int(res["n"])}
    verdict = decide(h2h, r10.get("T"), r10.get("C"), stage)
    verdict.update(frozen=frozen, deck=deck, budget=args.budget,
                   updates_total={n: state["arms"][n].get("updates_total") for n in manifest["arms"]},
                   updates_at_save={n: state["arms"][n]["frozen"][key]["updates_at_save"]
                                    for n in manifest["arms"]},
                   frozen_episode={n: state["arms"][n]["frozen"][key]["frozen_episode"]
                                   for n in manifest["arms"]},
                   h2h_result=os.path.join(h2h_dir, "result.json"), at=time.strftime("%Y-%m-%d %H:%M"))
    for name in (f"verdict_{key}.json", "verdict.json"):
        with open(os.path.join(root, name), "w") as f:
            json.dump(verdict, f, indent=1)
    state["stages"][key] = {"decision": verdict["decision"], "next": verdict["next"]}
    save_state(root, state)
    return verdict


# ---- dry run --------------------------------------------------------------------
def print_plan(args, manifest, stage):
    root = manifest["root"]
    key = f"s{stage}"
    deck = read_env_file(manifest["arms"]["T"]["env"])["CLASH_DECK"]
    keys = ("CLASH_OPP_TRACKER", "CLASH_WEIGHTS", "CLASH_LOGDIR", "CLASH_ARTIFACT_DIR", "PYTHONPATH")
    print(f"DRY RUN: stage {stage} of the tracker A/B in {root}; nothing is run or written.\n")
    print(f"1. launch both arms ({args.budget} updates each), logs under <arm>/train_{key}.log:")
    for name, arm in manifest["arms"].items():
        cmd, env, cwd = arm_launch(args, arm)
        print("   " + fmt(cmd, env, cwd, keys))
    print(f"2. each arm: count '>> Update @' lines; at the first '>>> Checkpoint saved' at or after "
          f"{args.budget} updates copy the generation to {root}/frozen/<arm>_{key}_u<updates>.pth, "
          "then SIGINT the trainer (it saves)")
    jd = os.path.join(root, "judge", key)
    frozen = {n: os.path.join(root, "frozen", f"{n}_{key}_u<updates>.pth") for n in manifest["arms"]}
    print("3. head-to-head, tracker vs control:")
    cmd, env, cwd = h2h_command(args, frozen["T"], frozen["C"], deck, os.path.join(jd, "h2h"),
                                os.path.join(root, "H2H.md"))
    print("   " + fmt(cmd, env, cwd))
    if args.skip_bench:
        print("4. rung-10 benchmark: skipped (--skip-bench)")
    else:
        print("4. rung-10 benchmark on each frozen copy:")
        for name, arm in manifest["arms"].items():
            cmd, env, cwd = bench_command(args, arm, frozen[name], os.path.join(jd, f"bench_{name}"), deck,
                                          args.bench_games, args.bench_rung, args.bench_script)
            print("   " + fmt(cmd, env, cwd, ("CLASH_OPP_TRACKER",)))
    print(f"5. verdict by the rule -> {root}/verdict_{key}.json and {root}/verdict.json "
          "(inconclusive at stage 1: rerun with --extend)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    d = Args(root="")
    ap.add_argument("--src", default=d.src, help="the worktree: PYTHONPATH, cwd and bench.py's SRC")
    ap.add_argument("--python", default=d.python)
    ap.add_argument("--nice", type=int, default=d.nice, help="nice level of the arms (judging runs at 19)")
    ap.add_argument("--budget", type=int, default=d.budget, help="PPO updates per arm (and per --extend)")
    ap.add_argument("--h2h-games", type=int, default=d.h2h_games)
    ap.add_argument("--bench-games", type=int, default=d.bench_games)
    ap.add_argument("--bench-rung", type=int, default=d.bench_rung)
    ap.add_argument("--bench-script", default=d.bench_script)
    ap.add_argument("--skip-bench", action="store_true",
                    help="no rung-10 benchmark: a score >= 0.55 then ends 'incomplete', never adopt")
    ap.add_argument("--poll", type=float, default=d.poll)
    ap.add_argument("--stop-wait", type=float, default=d.stop_wait)
    ap.add_argument("--extend", action="store_true",
                    help="continue both arms by --budget more updates after an inconclusive stage 1")
    ap.add_argument("--dry-run", action="store_true")
    ns = ap.parse_args(argv)
    args = Args(**{**vars(ns), "root": os.path.abspath(ns.root)})
    args.src = os.path.abspath(args.src)

    mpath = os.path.join(args.root, "ab_manifest.json")
    if not os.path.exists(mpath):
        raise SystemExit(f"!! {mpath} not found: run python -m python_ai.tools.setup_tracker_ab first")
    manifest = json.load(open(mpath))
    state = load_state(args.root)
    stage = 2 if args.extend else 1
    if args.extend:
        s1 = state["stages"].get("s1", {})
        if s1.get("decision") != "inconclusive":
            raise SystemExit(f"!! --extend needs an inconclusive stage 1 (state says {s1 or 'none'})")
        if "s2" in state["stages"]:
            raise SystemExit("!! the one extension has been run already")
    if args.dry_run:
        print_plan(args, manifest, stage)
        return None
    run_arms(args, manifest, state, stage)
    verdict = judge(args, manifest, state, stage)
    print(json.dumps({k: verdict[k] for k in ("stage", "score", "decision", "next", "reason")}, indent=1))
    return verdict


if __name__ == "__main__":
    main()
