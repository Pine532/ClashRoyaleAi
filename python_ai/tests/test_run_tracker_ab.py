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
