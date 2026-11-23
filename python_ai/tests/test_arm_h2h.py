"""eval/arm_h2h.py: a head-to-head where each net gets its own observation layout.

Checkpoints are tiny random nets in tmp_path; nothing real is read or written.
"""
import json

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

import clash_royale_env as E

from python_ai.eval import arm_h2h
from python_ai.models.net import MicroRoyaleNet
from python_ai.rl import abilities as ability_mod
from python_ai.rl import hand_slots
from python_ai.tracking import config as tcfg

CE = E.ClashRoyaleEnv
HOG = [15, 6, 25, 40, 24, 72, 33, 7]
TAIL = tcfg.tail_width("on")


def _ckpt(path, tracker, seed=0, nonzero_branch=False):
    torch.manual_seed(seed)
    net = MicroRoyaleNet(num_ability_slots=0, tracker_size=TAIL if tracker else 0)
    if tracker and nonzero_branch:
        with torch.no_grad():
            net.tracker_branch[-1].weight.normal_(0, 0.5)
    torch.save({"model": net.state_dict(), "deck": HOG, "episodes_completed": 7}, path)
    return str(path)


@pytest.fixture
def old(tmp_path):
    return _ckpt(tmp_path / "old.pth", False)


@pytest.fixture
def trk(tmp_path):
    return _ckpt(tmp_path / "trk.pth", True, nonzero_branch=True)


def _game(seed=3):
    g = CE(HOG, HOG, 3600, E.TowerTroopType.NONE, E.TowerTroopType.NONE)
    g.seed(seed)
    return g


def test_widths_follow_each_sides_own_layout(old, trk):
    engine = _game().observation_size()
    so = arm_h2h.Side(old, "off", HOG)
    st = arm_h2h.Side(trk, "on", HOG)
    sz = arm_h2h.Side(trk, "zero", HOG)
    assert so.net.obs_size == engine and so.tail_width == 0
    assert st.net.obs_size == engine + TAIL and st.tail_width == TAIL
    assert sz.net.obs_size == engine + TAIL
    g = _game()
    for side, want in ((so, engine), (st, engine + TAIL), (sz, engine + TAIL)):
        tail = side.make_tail(g, 0)
        obs = arm_h2h.observation(g, 0, tail)
        assert obs.shape == (want,)
        if side.net.expects_tail:
            side.net.require_tail(torch.tensor(obs)[None])


def test_a_zero_side_gets_an_all_zero_tail(trk):
    g = _game()
    tail = arm_h2h.Side(trk, "zero", HOG).make_tail(g, 1)
    tail.reset((HOG, HOG))
    for _ in range(20):
        g.step_self_play(0, 9.0, 5.0, 0, 9.0, 5.0, 10)
        obs = arm_h2h.observation(g, 1, tail)
        assert not obs[g.observation_size():].any()
        assert obs[g.observation_size():].shape == (TAIL,)


def test_layout_must_match_the_mode(old, trk):
    with pytest.raises(SystemExit):
        arm_h2h.Side(trk, "off", HOG)      # a tracker net cannot be played untailed
    with pytest.raises(SystemExit):
        arm_h2h.Side(old, "on", HOG)       # and an old net has no branch to feed
    with pytest.raises(SystemExit):
        arm_h2h.Side(old, "zero", HOG)


def test_n_must_be_even(old, tmp_path):
    with pytest.raises(SystemExit):
        arm_h2h.run(old, "off", old, "off", HOG, 3, str(tmp_path / "w"))


def _old_act(net, games, teams, hx, cx, slots):
    """runs/tools_scratch/h2h.py's act, copied verbatim (the live watchdog's)."""
    obs = torch.tensor(np.stack([np.asarray(g.get_observation_for_team(t), dtype=np.float32)
                                 for g, t in zip(games, teams)]))
    hand_ready = torch.cat([hand_slots.ready_from_engine(g, t, net.hand_size) for g, t in zip(games, teams)])
    card_mask = net.affordability_mask(obs, hand_ready)
    feats, emb, spatial, hires = net.extract_features_hires(obs)
    logits, ab1, ab2, _, (hx, cx) = net.step_lstm_and_card(feats, (hx, cx), card_mask)
    card = Categorical(logits=logits).sample()
    place = net.placement_given_card(hx, emb, card, obs, spatial, hires_map=hires)
    cell = Categorical(logits=place).sample()
    x, y = net.cell_to_xy(cell)
    flags = np.zeros((len(games), 3), dtype=bool)
    ab_logits = [l for l in (ab1, ab2) if l is not None]
    if slots and ab_logits:
        ready = torch.tensor([[bool(g.is_champion_ability_ready(t, s)) for s in slots]
                              for g, t in zip(games, teams)])
        acts, _, _ = ability_mod.sample(ab_logits, ready)
        for k, s in enumerate(slots):
            flags[:, s] = acts[:, k].numpy().astype(bool)
    return (card.numpy().reshape(-1), x.numpy().reshape(-1), y.numpy().reshape(-1), flags, hx, cx)


def test_an_old_layout_net_acts_as_in_the_old_h2h(old):
    net = arm_h2h.Side(old, "off", HOG).net
    games = [_game(11), _game(12)]
    for g in games:                      # a mid-game state, not the opening
        for _ in range(12):
            g.step_self_play(0, 9.0, 5.0, 1, 9.0, 5.0, 10)
    teams = [0, 1]
    z = lambda: torch.zeros(2, net.LSTM_HIDDEN)
    for seed in (1, 2, 3):
        torch.manual_seed(seed)
        ref = _old_act(net, games, teams, z(), z(), [])
        torch.manual_seed(seed)
        new = arm_h2h.act(net, games, teams, z(), z(), [], tails=[None, None])
        for r, n in zip(ref[:4], new[:4]):
            assert np.array_equal(r, n)
        assert torch.equal(ref[4], new[4]) and torch.equal(ref[5], new[5])


def test_the_null_runs_and_records_per_side_scores(trk, tmp_path):
    out = arm_h2h.run(trk, "on", trk, "zero", HOG, 2, str(tmp_path / "w"),
                      out_md=str(tmp_path / "row.md"), wave=2)
    saved = json.load(open(tmp_path / "w" / "result.json"))
    assert saved["n"] == 2 and saved["unfinished"] == 0
    assert saved["a_mode"] == "on" and saved["b_mode"] == "zero"
    assert saved["a_tail"] == TAIL and saved["b_tail"] == TAIL
    assert saved["a_score"] + saved["b_score"] == pytest.approx(1.0)
    assert saved["wins"] + saved["draws"] + saved["losses"] == 2
    assert len(saved["ci95"]) == 2
    assert out["a_score"] == saved["a_score"]
    assert "|" in (tmp_path / "row.md").read_text()


def test_reproducible_with_fixed_seeds(trk, old, tmp_path):
    r1 = arm_h2h.run(trk, "on", old, "off", HOG, 2, str(tmp_path / "w1"), wave=2)
    r2 = arm_h2h.run(trk, "on", old, "off", HOG, 2, str(tmp_path / "w2"), wave=2)
    for k in ("wins", "draws", "losses", "score", "mean_decisions", "a_tracker_stats"):
        assert r1[k] == r2[k], k


def test_no_side_sees_the_others_press_in_the_same_step(trk, monkeypatch, tmp_path):
    """Per decision: both tails are polled (observations built) before any
    note_flags of that step, and all note_flags precede the step."""
    from python_ai.tracking import tail as tail_mod
    log = []
    real_note, real_feat = tail_mod.TrackerTail.note_flags, tail_mod.TrackerTail.features

    def note(self, team, flags):
        log.append("note")
        return real_note(self, team, flags)

    def feat(self):
        log.append("poll")
        return real_feat(self)

    monkeypatch.setattr(tail_mod.TrackerTail, "note_flags", note)
    monkeypatch.setattr(tail_mod.TrackerTail, "features", feat)
    monkeypatch.setattr(arm_h2h, "MAX_STEPS", 3)
    arm_h2h.run(trk, "on", trk, "on", HOG, 2, str(tmp_path / "w"), wave=1)
    # each decision, per game: 2 polls then 4 notes (2 tails x 2 teams)
    per_decision = ["poll"] * 2 + ["note"] * 4
    assert log == per_decision * (3 * 2)


OLD_SCRIPT = "/Users/owenzhang/ClashRoyaleAi/runs/tools_scratch/h2h.py"


@pytest.mark.slow
@pytest.mark.skipif(not __import__("os").path.exists(OLD_SCRIPT),
                    reason="the live watchdog's h2h.py is not on this machine")
def test_off_off_reproduces_the_old_script_end_to_end(tmp_path):
    """Two old-layout nets: arm_h2h and runs/tools_scratch/h2h.py (run as a
    subprocess, read-only) give the same games for the same seeds."""
    import subprocess
    import sys
    a = _ckpt(tmp_path / "a.pth", False, seed=1)
    b = _ckpt(tmp_path / "b.pth", False, seed=2)
    root = str(__import__("pathlib").Path(arm_h2h.__file__).parents[2])
    subprocess.run([sys.executable, OLD_SCRIPT, root, a, b, ",".join(map(str, HOG)), "4",
                    str(tmp_path / "old")], check=True, capture_output=True,
                   env={**__import__("os").environ, "PYTHONPATH": root})
    old = json.load(open(tmp_path / "old" / "result.json"))
    new = arm_h2h.run(a, "off", b, "off", HOG, 4, str(tmp_path / "new"), wave=20)
    for k in ("wins", "draws", "losses", "score", "mean_decisions"):
        assert new[k] == old[k], k
