"""tools/setup_tracker_ab.py: two A/B arms that link the shared pool read-only.

Everything is in tmp_path: a fake shared pool, a tiny starting checkpoint.
"""
import json
import os
import stat
import time

import pytest
import torch

from python_ai.models.net import MicroRoyaleNet
from python_ai.tools import setup_tracker_ab as setup
from python_ai.trainers import league

HOG = [15, 6, 25, 40, 24, 72, 33, 7]
LINEAGE = 1_000_000.0
EPISODE = 50_000


def _write(path, mtime, tracker=False):
    torch.save({"model": MicroRoyaleNet(num_ability_slots=0).state_dict()}, path)
    os.utime(path, (mtime, mtime))
    return str(path)


@pytest.fixture
def start(tmp_path):
    net = MicroRoyaleNet(num_ability_slots=0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    path = tmp_path / "start_selfplay.pth"
    torch.save({"model": net.state_dict(), "optimizer": opt.state_dict(),
                "episodes_completed": EPISODE, "deck": HOG,
                "lineage_started_at": LINEAGE,
                "clash_settings": {"CLASH_DECK": "x", "CLASH_OPP_TRACKER": None}}, path)
    return str(path)


@pytest.fixture
def pool(tmp_path):
    """4 eligible-or-not snapshots: older than the lineage, a pipeline-1 one, an
    old pipeline-2 one, and a pipeline-2 one younger than the age gate."""
    d = tmp_path / "shared"
    d.mkdir()
    _write(d / "100_pipeline1_ep00000100.pth", LINEAGE - 500)                 # before the lineage
    _write(d / "200_pipeline1_ep00040000.pth", LINEAGE + 100)
    _write(d / "300_pipeline2_ep00001000.pth", LINEAGE + 300)
    _write(d / "400_pipeline2_ep00049999.pth", LINEAGE + 400)                 # too young
    return str(d)


@pytest.fixture
def template(tmp_path):
    p = tmp_path / "tpl.env"
    p.write_text("# a comment\nCLASH_DECK=15,6,25,40,24,72,33,7\n"
                 "CLASH_WEIGHTS=/live/run.pth\nCLASH_LOGDIR=/live/runs\n"
                 "CLASH_ARTIFACT_DIR=/live/art\nCLASH_OPP_TRACKER=off\n"
                 "CLASH_NUM_ENVS=16\nCLASH_UPDATE_TIMESTEP=250\n")
    return str(p)


def _env(path):
    out = {}
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out.setdefault(k, []).append(v)
    return out


def _snapshot_tree(d):
    return {f: (os.stat(os.path.join(d, f)).st_mtime_ns, os.stat(os.path.join(d, f)).st_mode,
                open(os.path.join(d, f), "rb").read()) for f in sorted(os.listdir(d))}


def _run(tmp_path, start, pool, template, extra=()):
    root = tmp_path / "runs" / "ab_x"
    setup.main(["--root", str(root), "--start", start, "--shared-pool", pool,
                "--env-template", template, *extra])
    return root


def test_creates_both_arms_without_touching_the_shared_pool(tmp_path, start, pool, template):
    before = _snapshot_tree(pool)
    dir_mode = os.stat(pool).st_mode
    os.chmod(pool, 0o555)                       # a write into the shared dir would raise
    try:
        root = _run(tmp_path, start, pool, template)
    finally:
        os.chmod(pool, dir_mode)
    assert _snapshot_tree(pool) == before
    assert not [f for f in os.listdir(pool) if f.startswith(".")]

    for arm, mode in (("C", "zero"), ("T", "on")):
        env = _env(root / f"{arm}.env")
        assert env["CLASH_OPP_TRACKER"] == [mode]
        assert env["CLASH_WEIGHTS"] == [str(root / arm / "checkpoints" / f"{arm}.pth")]
        assert env["CLASH_LOGDIR"] == [str(root / arm / "runs")]
        assert env["CLASH_ARTIFACT_DIR"] == [str(root / arm)]
        assert env["CLASH_NUM_ENVS"] == ["16"] and env["CLASH_UPDATE_TIMESTEP"] == ["250"]
        assert env["CLASH_DECK"] == ["15,6,25,40,24,72,33,7"]      # the template's own
        # each key once: the template's CLASH_WEIGHTS etc. are replaced, not duplicated
        assert all(len(v) == 1 for v in env.values())
        # the checkpoint train_selfplay resumes from
        resume = root / arm / "checkpoints" / f"{arm}_selfplay.pth"
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        assert any(k.startswith("tracker_branch.") for k in ck["model"])
        assert ck["episodes_completed"] == EPISODE
        assert ck["clash_settings"]["CLASH_OPP_TRACKER"] == mode
        assert ck["lineage_started_at"] == LINEAGE


def test_the_resume_path_is_the_one_train_selfplay_computes(tmp_path, start, pool, template, monkeypatch):
    root = _run(tmp_path, start, pool, template)
    from python_ai.trainers.train_selfplay import selfplay_paths
    env = _env(root / "T.env")
    monkeypatch.setenv("CLASH_WEIGHTS", env["CLASH_WEIGHTS"][0])
    monkeypatch.setenv("CLASH_LOGDIR", env["CLASH_LOGDIR"][0])
    assert selfplay_paths()[0] == str(root / "T" / "checkpoints" / "T_selfplay.pth")


def test_both_arms_start_from_the_same_weights(tmp_path, start, pool, template):
    root = _run(tmp_path, start, pool, template)
    c = torch.load(root / "C/checkpoints/C_selfplay.pth", weights_only=False)
    t = torch.load(root / "T/checkpoints/T_selfplay.pth", weights_only=False)
    assert c["model"].keys() == t["model"].keys()
    for k in c["model"]:
        assert torch.equal(c["model"][k], t["model"][k]), k


def test_links_are_absolute_and_follow_mtime(tmp_path, start, pool, template):
    root = _run(tmp_path, start, pool, template)
    shared = sorted(f for f in os.listdir(pool) if f.endswith(".pth"))
    for arm in ("C", "T"):
        d = root / arm / "historical_checkpoints"
        assert sorted(os.listdir(d)) == shared
        for f in shared:
            link = d / f
            assert os.path.islink(link)
            target = os.readlink(link)
            assert os.path.isabs(target) and target == os.path.join(pool, f)
            # the link was made now; the lineage filter must see the TARGET's time
            assert os.lstat(link).st_mtime > os.path.getmtime(os.path.join(pool, f)) + 1000
            assert os.path.getmtime(link) == os.path.getmtime(os.path.join(pool, f))


def test_the_eligible_set_matches_the_shared_pool(tmp_path, start, pool, template):
    root = _run(tmp_path, start, pool, template)
    want = [os.path.realpath(p) for p in league.discover_historical_checkpoints(
        EPISODE, directory=pool, since=LINEAGE)]
    assert [os.path.basename(p) for p in want] == [
        "200_pipeline1_ep00040000.pth", "300_pipeline2_ep00001000.pth"]
    for arm in ("C", "T"):
        got = [os.path.realpath(p) for p in league.discover_historical_checkpoints(
            EPISODE, directory=str(root / arm / "historical_checkpoints"), since=LINEAGE)]
        assert got == want


def test_verify_pool_catches_a_missing_link(tmp_path, pool):
    arm_pool = tmp_path / "armpool"
    arm_pool.mkdir()
    for f in sorted(os.listdir(pool))[:1] + sorted(os.listdir(pool))[2:]:    # an ELIGIBLE one missing
        os.symlink(os.path.join(pool, f), arm_pool / f)
    with pytest.raises(SystemExit):
        setup.verify_pool(pool, str(arm_pool), LINEAGE, EPISODE)


def test_refuses_an_existing_root(tmp_path, start, pool, template):
    root = tmp_path / "runs" / "ab_x"
    root.mkdir(parents=True)
    with pytest.raises(SystemExit):
        setup.main(["--root", str(root), "--start", start, "--shared-pool", pool,
                    "--env-template", template])
    assert os.listdir(root) == []


def test_refuses_an_empty_shared_pool(tmp_path, start, template):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit):
        setup.main(["--root", str(tmp_path / "r"), "--start", start,
                    "--shared-pool", str(empty), "--env-template", template])
    assert not (tmp_path / "r").exists()


def test_manifest_lists_the_arms(tmp_path, start, pool, template):
    root = _run(tmp_path, start, pool, template)
    m = json.load(open(root / "ab_manifest.json"))
    assert set(m["arms"]) == {"C", "T"}
    assert m["arms"]["C"]["mode"] == "zero" and m["arms"]["T"]["mode"] == "on"
    assert m["arms"]["T"]["checkpoint"] == str(root / "T/checkpoints/T_selfplay.pth")
    assert m["arms"]["T"]["env"] == str(root / "T.env")


def test_a_new_snapshot_in_an_arm_pool_leaves_the_shared_files_alone(tmp_path, start, pool, template):
    """The trainer's writer (save_historical_snapshot) adds a new regular file;
    the linked shared files stay byte-identical."""
    from python_ai.rl.checkpointing import save_historical_snapshot
    root = _run(tmp_path, start, pool, template)
    before = _snapshot_tree(pool)
    arm_pool = str(root / "T" / "historical_checkpoints")
    path = save_historical_snapshot(MicroRoyaleNet(num_ability_slots=0), 123, "pipeline2", directory=arm_pool)
    assert os.path.isfile(path) and not os.path.islink(path)
    assert _snapshot_tree(pool) == before
    assert sorted(os.listdir(pool)) == sorted(before)


def _with_settings(tmp_path, start, settings):
    ck = torch.load(start, map_location="cpu", weights_only=False)
    ck["clash_settings"] = settings
    torch.save(ck, start)


def test_the_live_runs_extra_settings_reach_both_arms(tmp_path, start, pool, template, capsys):
    """Keys the live run had (from the checkpoint's clash_settings) but the template lacks."""
    _with_settings(tmp_path, start, {
        "CLASH_DECK_POOL": "/live/pools/batch_4.json", "CLASH_DEVICE": "mps", "CLASH_SAVE_EVERY": "250",
        "CLASH_NUM_ENVS": "99",                       # the template says 16: the template wins
        "CLASH_WEIGHTS": "/live/run.pth", "CLASH_OPP_TRACKER": "off",   # owned: never carried
        "CLASH_UNSET": None})
    root = _run(tmp_path, start, pool, template)
    for arm, mode in (("C", "zero"), ("T", "on")):
        env = _env(root / f"{arm}.env")
        assert env["CLASH_DECK_POOL"] == ["/live/pools/batch_4.json"]
        assert env["CLASH_DEVICE"] == ["mps"] and env["CLASH_SAVE_EVERY"] == ["250"]
        assert env["CLASH_NUM_ENVS"] == ["16"]
        assert env["CLASH_OPP_TRACKER"] == [mode]
        assert "CLASH_UNSET" not in env
        assert all(len(v) == 1 for v in env.values())
    m = json.load(open(root / "ab_manifest.json"))
    assert m["extra_env"] == {"CLASH_DECK_POOL": "/live/pools/batch_4.json", "CLASH_DEVICE": "mps",
                              "CLASH_SAVE_EVERY": "250"}
    out = capsys.readouterr().out
    assert "CLASH_DECK_POOL=/live/pools/batch_4.json" in out      # the final env is printed


def test_the_env_flag_overrides_and_can_choose_meta_decks(tmp_path, start, pool, template):
    _with_settings(tmp_path, start, {"CLASH_DECK_POOL": "/live/pools/batch_4.json"})
    root = _run(tmp_path, start, pool, template,
                extra=("--env", "CLASH_DECK_POOL=/other/meta.json", "--env", "CLASH_SAVE_EVERY=50"))
    for arm in ("C", "T"):
        env = _env(root / f"{arm}.env")
        assert env["CLASH_DECK_POOL"] == ["/other/meta.json"] and env["CLASH_SAVE_EVERY"] == ["50"]


def test_the_env_flag_cannot_set_an_owned_key(tmp_path, start, pool, template):
    with pytest.raises(SystemExit):
        _run(tmp_path, start, pool, template, extra=("--env", "CLASH_OPP_TRACKER=on"))
    assert not (tmp_path / "runs" / "ab_x").exists()
