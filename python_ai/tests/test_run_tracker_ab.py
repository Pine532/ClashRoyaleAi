"""tools/run_tracker_ab.py: the update counter, the verdict rule at its
boundaries, the commands the harness would run, and the dry run.

Nothing here trains: logs are strings, the verdict inputs are numbers.
"""
import json
import os

import pytest
import torch

from python_ai.models.net import MicroRoyaleNet
from python_ai.tools import run_tracker_ab as ab
from python_ai.tools import setup_tracker_ab as setup

UPD = "  >> Update @ ep {ep} | Actor: 0.1 | Critic: 0.2 | Entropy: 1.0 | ClipFrac: 0.01 | NextCardCE: 1 Acc: 0.2 | CycleId: 1 Acc: 0.3"
SAVE = ">>> Checkpoint saved to /x/T_selfplay.pth (episode {ep})"


def feed(counter, lines):
    return [counter.feed(l) for l in lines]


# ---- the update counter ------------------------------------------------------
def test_counts_only_update_lines():
    c = ab.UpdateCounter(budget=100)
    feed(c, [UPD.format(ep=1), "     H(place|card): a 0.1", "  [WARN] x", ">>> Interrupted", UPD.format(ep=2),
             "Update @ ep 3 (not a trainer line, no >>)", "  >> Updated @ ep 4"])
    assert c.updates == 2


def test_a_save_before_the_budget_does_not_trigger_and_the_first_after_does():
    c = ab.UpdateCounter(budget=3)
    out = feed(c, [UPD.format(ep=1), UPD.format(ep=2), SAVE.format(ep=5),        # 2 updates: no
                   UPD.format(ep=3), SAVE.format(ep=9),                          # 3 updates: yes
                   UPD.format(ep=4), SAVE.format(ep=14)])                        # already fired
    assert out[2] is None
    assert out[4] == {"path": "/x/T_selfplay.pth", "episode": 9, "updates": 3}
    assert out[6] is None
    assert c.fired and c.updates == 4


def test_the_save_in_the_budgets_own_update_counts():
    # the periodic save follows the update's log line, so budget updates then a save fires
    c = ab.UpdateCounter(budget=1)
    assert feed(c, [UPD.format(ep=1), SAVE.format(ep=2)])[1]["updates"] == 1


def test_a_budget_met_between_saves_waits_for_the_next_save():
    c = ab.UpdateCounter(budget=2)
    out = feed(c, [UPD.format(ep=1), SAVE.format(ep=1), UPD.format(ep=2), UPD.format(ep=3), UPD.format(ep=4),
                   SAVE.format(ep=40)])
    assert out[1] is None and out[5]["updates"] == 4


def test_the_log_tail_returns_whole_lines_only(tmp_path):
    p = tmp_path / "l.log"
    t = ab.LogTail(str(p))
    assert t.read_lines() == []                  # no file yet
    p.write_text("a\nb")
    assert t.read_lines() == ["a"]
    with open(p, "a") as f:
        f.write("c\nd\n")
    assert t.read_lines() == ["bc", "d"]


# ---- the verdict rule --------------------------------------------------------
def v(wins, draws=0, n=400, c10=100, t10=100, n10=200, stage=1):
    return ab.decide({"wins": wins, "draws": draws, "n": n},
                     {"wins": t10, "n": n10}, {"wins": c10, "n": n10}, stage)


def test_exactly_055_with_the_guard_met_adopts():
    r = v(220)                                    # 220/400 = 0.55
    assert r["score"] == 0.55 and r["decision"] == "adopt" and r["next"] == "adopt"


def test_just_below_055_is_inconclusive():
    r = v(219)
    assert r["decision"] == "inconclusive" and r["next"] == "extend"
    assert v(219, draws=2)["score"] == 0.55 and v(219, draws=2)["decision"] == "adopt"   # 219 + 1 = 220


def test_exactly_050_is_inconclusive_and_below_it_rejects():
    assert v(200)["decision"] == "inconclusive"
    assert v(199)["decision"] == "reject" and v(199)["next"] == "stop"
    assert v(199, draws=2)["decision"] == "inconclusive"      # 200/400 by half points


def test_the_guard_at_exactly_10_below_passes_and_11_fails():
    assert v(240, c10=130, t10=120)["decision"] == "adopt"                    # 10/200 below
    r = v(240, c10=130, t10=119)                                              # 11/200 below
    assert r["decision"] == "not_adopted" and "guard" in r["reason"]
    assert v(240, c10=100, t10=150)["decision"] == "adopt"                    # above the control


def test_the_guard_scales_with_the_bench_size():
    assert v(240, c10=2, t10=1, n10=20)["decision"] == "adopt"                # 10/200 = 1 of 20
    assert v(240, c10=3, t10=1, n10=20)["decision"] == "not_adopted"


def test_a_missing_rung_10_never_adopts():
    r = ab.decide({"wins": 300, "draws": 0, "n": 400}, None, None, 1)
    assert r["decision"] == "incomplete" and r["next"] == "run the rung-10 benchmark"
    # but a rejection or an inconclusive score does not depend on it
    assert ab.decide({"wins": 100, "draws": 0, "n": 400}, None, None, 1)["decision"] == "reject"


def test_a_second_inconclusive_is_not_adopted():
    r = v(210, stage=2)
    assert r["decision"] == "not_adopted" and r["next"] == "stop" and "second" in r["reason"]
    assert v(230, stage=2)["decision"] == "adopt"


# ---- commands and dry run ----------------------------------------------------
@pytest.fixture
def arms(tmp_path):
    hog = [15, 6, 25, 40, 24, 72, 33, 7]
    net = MicroRoyaleNet(num_ability_slots=0)
    start = tmp_path / "start_selfplay.pth"
    torch.save({"model": net.state_dict(), "optimizer": torch.optim.Adam(net.parameters()).state_dict(),
                "episodes_completed": 10, "deck": hog, "lineage_started_at": 1.0}, start)
    pool = tmp_path / "shared"
    pool.mkdir()
    torch.save({"model": net.state_dict()}, pool / "1_pipeline1_ep00000001.pth")
    os.utime(pool / "1_pipeline1_ep00000001.pth", (5, 5))
    tpl = tmp_path / "t.env"
    tpl.write_text("CLASH_DECK=15,6,25,40,24,72,33,7\nCLASH_NUM_ENVS=1\n")
    root = tmp_path / "ab"
    setup.main(["--root", str(root), "--start", str(start), "--shared-pool", str(pool),
                "--env-template", str(tpl)])
    return root


def test_dry_run_prints_every_command_and_runs_nothing(arms, capsys, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("launched"))
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("ran"))
    before = sorted(os.listdir(arms))
    ab.main(["--root", str(arms), "--dry-run"])
    out = capsys.readouterr().out
    assert sorted(os.listdir(arms)) == before                    # nothing written
    assert out.count("train_selfplay") == 2                      # one launch per arm
    assert "CLASH_OPP_TRACKER=zero" in out and "CLASH_OPP_TRACKER=on" in out
    assert "nice" in out and "PYTHONPATH" in out
    assert "arm_h2h" in out and "--n 400" in out and "--a-mode on" in out and "--b-mode zero" in out
    assert out.count("bench.py") == 2 and "--rung 10" in out and " 200 " in out
    assert "verdict" in out


def test_commands_use_the_arms_env_and_the_worktree(arms):
    m = json.load(open(arms / "ab_manifest.json"))
    a = ab.Args(root=str(arms), src="/work/tree", python="/py", nice=7)
    cmd, env, cwd = ab.arm_launch(a, m["arms"]["T"])
    assert cmd[:3] == ["nice", "-n", "7"] and cmd[-2:] == ["-m", "python_ai.trainers.train_selfplay"]
    assert env["CLASH_OPP_TRACKER"] == "on" and env["PYTHONPATH"].split(os.pathsep)[0] == "/work/tree"
    assert env["CLASH_WEIGHTS"].endswith("T/checkpoints/T.pth") and cwd == "/work/tree"
    bc, benv, _ = ab.bench_command(a, m["arms"]["C"], "/f/C.pth", "/w", "15,6", 200, 10, "/b/bench.py")
    assert benv["CLASH_OPP_TRACKER"] == "zero"
    assert bc == ["nice", "-n", "19", "/py", "-u", "/b/bench.py", "/work/tree", "/f/C.pth", "15,6", "200",
                  "/w", "--rung", "10"]


def test_extend_refuses_without_an_inconclusive_stage_1(arms):
    with pytest.raises(SystemExit):
        ab.main(["--root", str(arms), "--extend", "--dry-run"])


# ---- run_arms / freeze / judge end to end, with a fake trainer and fake benchmarks ----------
FAKE_TRAINER = '''
import os, signal, sys, time
import torch
from python_ai.rl.checkpointing import atomic_save, keep_in_history
w = os.environ["CLASH_WEIGHTS"]
path = os.path.join(os.path.dirname(w), os.path.splitext(os.path.basename(w))[0] + "_selfplay.pth")
mode = os.environ["CLASH_OPP_TRACKER"]
ep = [100]
def save():
    atomic_save({"model": {}, "episodes_completed": ep[0]}, path)
    keep_in_history(path, ep[0], 5)
    print(f">>> Checkpoint saved to {path} (episode {ep[0]})", flush=True)
def stop(*_):
    ep[0] += 1
    save()
    sys.exit(0)
signal.signal(signal.SIGINT, stop)
n = 0
while True:
    n += 1
    ep[0] += 10
    print(f"  >> Update @ ep {ep[0]} | Actor: 0.1", flush=True)
    if os.environ.get("FAKE_FAIL_MODE") == mode and n == 2:
        sys.exit(3)
    if n % 2 == 0:
        save()
    time.sleep(0.05)
'''
FAKE_H2H = '''
import json, os, sys
a = sys.argv
work, n = a[a.index("--work") + 1], int(a[a.index("--n") + 1])
os.makedirs(work, exist_ok=True)
open(os.path.join(os.path.dirname(work), "h2h_calls"), "a").write("x")
json.dump({"wins": int(os.environ["FAKE_H2H_WINS"]), "draws": 0, "n": n}, open(os.path.join(work, "result.json"), "w"))
'''
FAKE_BENCH = '''
import json, os, sys
work, n = sys.argv[5], int(sys.argv[4])
os.makedirs(work, exist_ok=True)
wins = int(os.environ["FAKE_BENCH_" + os.environ["CLASH_OPP_TRACKER"].upper()])
json.dump({"wins": wins, "n": n}, open(os.path.join(work, "result.json"), "w"))
'''


@pytest.fixture
def fakes(tmp_path, monkeypatch):
    d = tmp_path / "fakes"
    d.mkdir()
    (d / "fake_trainer.py").write_text(FAKE_TRAINER)
    (d / "fake_h2h.py").write_text(FAKE_H2H)
    (d / "fake_bench.py").write_text(FAKE_BENCH)
    monkeypatch.setenv("PYTHONPATH", str(d))
    for k, v in (("FAKE_H2H_WINS", "240"), ("FAKE_BENCH_ZERO", "100"), ("FAKE_BENCH_ON", "100")):
        monkeypatch.setenv(k, v)
    return d


def _args(arms, fakes, **kw):
    return ab.Args(root=str(arms), budget=3, h2h_games=400, bench_games=200, poll=0.1, stop_wait=60,
                   trainer_module="fake_trainer", h2h_module="fake_h2h",
                   bench_script=str(fakes / "fake_bench.py"), nice=0, **kw)


def _go(arms, args, stage=1):
    m = json.load(open(arms / "ab_manifest.json"))
    st = ab.load_state(str(arms))
    ab.run_arms(args, m, st, stage)
    return m, st, ab.judge(args, m, st, stage)


def test_run_arms_freezes_both_arms_and_the_verdict_is_written(arms, fakes):
    m, st, v = _go(arms, _args(arms, fakes))
    assert v["decision"] == "adopt" and v["score"] == 0.6
    for name in ("C", "T"):
        rec = st["arms"][name]
        fz = rec["frozen"]["s1"]
        assert os.path.exists(fz["path"]) and fz["exact"] and fz["updates_at_save"] >= 3
        assert rec["pid"] is None
        assert rec["segments"][0]["rc"] == 0                    # the SIGINT made it save and exit
        ck = torch.load(fz["path"], map_location="cpu", weights_only=False)
        assert ck["episodes_completed"] == fz["save_episode"]    # the frozen generation, not the SIGINT save
    assert os.path.exists(arms / "verdict.json") and os.path.exists(arms / "verdict_s1.json")
    saved = json.load(open(arms / "verdict.json"))
    assert saved["updates_at_save"] == {n: st["arms"][n]["frozen"]["s1"]["updates_at_save"] for n in "CT"}
    assert saved["frozen_episode"]["T"] == st["arms"]["T"]["frozen"]["s1"]["frozen_episode"]
    assert json.load(open(arms / "ab_state.json"))["stages"]["s1"]["decision"] == "adopt"


def test_an_arm_that_dies_does_not_orphan_the_other(arms, fakes, monkeypatch):
    monkeypatch.setenv("FAKE_FAIL_MODE", "zero")
    args = _args(arms, fakes)
    m = json.load(open(arms / "ab_manifest.json"))
    st = ab.load_state(str(arms))
    with pytest.raises(SystemExit, match="exited"):
        ab.run_arms(args, m, st, 1)
    t_pid = st["arms"]["T"].get("pid")
    assert t_pid is None or not ab.pid_alive(t_pid)
    for name in ("C", "T"):
        assert st["arms"][name]["pid"] is None
    import subprocess
    out = subprocess.run(["pgrep", "-f", "fake_trainer"], capture_output=True, text=True).stdout.split()
    assert not [p for p in out if p != str(os.getpid())]


def test_a_skip_bench_incomplete_stage_is_completed_by_a_rerun_without_replaying_h2h(arms, fakes):
    args = _args(arms, fakes, skip_bench=True)
    m, st, v = _go(arms, args)
    assert v["decision"] == "incomplete"
    assert open(arms / "judge" / "s1" / "h2h_calls").read() == "x"
    full = _args(arms, fakes)
    st2 = ab.load_state(str(arms))
    ab.run_arms(full, m, st2, 1)                              # both frozen: launches nothing
    v2 = ab.judge(full, m, st2, 1)
    assert v2["decision"] == "adopt"
    assert open(arms / "judge" / "s1" / "h2h_calls").read() == "x"       # not replayed
    assert json.load(open(arms / "verdict.json"))["decision"] == "adopt"


def test_the_extension_runs_stage_2_and_totals_the_updates(arms, fakes, monkeypatch):
    monkeypatch.setenv("FAKE_H2H_WINS", "210")                # 0.525: inconclusive
    m, st, v1 = _go(arms, _args(arms, fakes))
    assert v1["decision"] == "inconclusive" and v1["next"] == "extend"
    monkeypatch.setenv("FAKE_H2H_WINS", "230")
    args = _args(arms, fakes, extend=True)
    st = ab.load_state(str(arms))
    ab.run_arms(args, m, st, 2)
    v2 = ab.judge(args, m, st, 2)
    assert v2["stage"] == 2 and v2["decision"] == "adopt"
    for n in "CT":
        rec = st["arms"][n]
        assert set(rec["frozen"]) == {"s1", "s2"} and len(rec["segments"]) == 2
        assert v2["updates_total"][n] == rec["segments"][0]["updates"] + rec["segments"][1]["updates"]
        assert v2["updates_at_save"][n] == rec["frozen"]["s2"]["updates_at_save"]
        assert rec["frozen"]["s2"]["updates_at_save"] <= v2["updates_total"][n]
