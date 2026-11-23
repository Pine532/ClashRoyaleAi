"""Set up the tracker A/B's two arms (docs/DECISIONS.md, "2026-11-22 (late)").

    python -m python_ai.tools.setup_tracker_ab --root runs/ab_tracker \\
        --start runs/checkpoints/minigiant_p2_selfplay.pth \\
        --shared-pool historical_checkpoints --env-template runs/minigiant_p2.env

Creates, for arm C (CLASH_OPP_TRACKER=zero, the control) and arm T (=on):

  ROOT/<arm>.env                     the template's settings, with CLASH_WEIGHTS,
                                     CLASH_LOGDIR, CLASH_ARTIFACT_DIR and
                                     CLASH_OPP_TRACKER set for the arm
  ROOT/<arm>/checkpoints/<arm>_selfplay.pth
                                     the starting checkpoint, migrated to the
                                     arm's layout (tools/migrate_checkpoint_tracker),
                                     at the path train_selfplay.py resumes from
                                     (selfplay_paths(): the CLASH_WEIGHTS stem +
                                     "_selfplay")
  ROOT/<arm>/historical_checkpoints/ one ABSOLUTE symlink per *.pth of the shared
                                     pool as of now
  ROOT/ab_manifest.json              what the harness (run_tracker_ab.py) reads

Why links: `discover_historical_checkpoints` reads the arm's own folder once
CLASH_ARTIFACT_DIR is set, so without them an arm would start phase 2 with no
past copies. The arm's NEW snapshots are regular files in its own folder.

The shared pool is only READ here: nothing in it is written, chmodded or
removed. The trainers cannot modify it through a link either: the only writers
of the pool folder (base_trainer.periodic -> save_historical_snapshot, and the
disabled exploiter) create NEW timestamped files via atomic_save (temp file +
os.replace onto a new name), and nothing deletes or renames a pool file (the
rmtree in BaseTrainer.setup is of the run's log dir; keep_in_history prunes only
the arm's own `<weights>_history`). os.replace onto a symlink replaces the link,
never its target.

The lineage filter (`since=lineage_started_at`) uses os.path.getmtime, which
follows a link, so it sees the TARGET's mtime. setup verifies that the arm
folder yields the same eligible set, in the same order, as the shared pool, and
refuses otherwise. Refuses if ROOT exists.
"""
import argparse
import glob
import json
import os
import sys

# Put the repo root on sys.path, for script and -m invocation alike.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch

import python_ai  # noqa: F401  -- finds the engine
from python_ai.rl.checkpointing import atomic_save
from python_ai.tools import migrate_checkpoint_tracker as migrate
from python_ai.tracking import config as tracker_config

ARMS = (("C", "zero"), ("T", "on"))
#: Template keys the arm sets itself; the template's copies are dropped.
OWNED = ("CLASH_WEIGHTS", "CLASH_LOGDIR", "CLASH_ARTIFACT_DIR", tracker_config.ENV_VAR)


def _eligible(directory, since, episode):
    from python_ai.trainers import league
    return [os.path.realpath(p) for p in league.discover_historical_checkpoints(
        episode, directory=directory, since=since)]


def verify_pool(shared, arm_pool, since, episode):
    """The arm's folder must give the same eligible checkpoints (by target, in
    order) as the shared pool. Raises SystemExit otherwise."""
    want, got = _eligible(shared, since, episode), _eligible(arm_pool, since, episode)
    if want != got:
        raise SystemExit(
            f"!! the arm pool {arm_pool} does not give the shared pool's eligible set.\n"
            f"   shared: {len(want)} {[os.path.basename(p) for p in want][:5]}...\n"
            f"   arm:    {len(got)} {[os.path.basename(p) for p in got][:5]}...")
    return want


def link_pool(shared, arm_pool):
    """One absolute symlink per *.pth of `shared`. Only reads `shared`."""
    os.makedirs(arm_pool)
    names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(shared, "*.pth")))
    for n in names:
        os.symlink(os.path.join(shared, n), os.path.join(arm_pool, n))
    return names


def env_text(template_text, arm_dir, arm, mode):
    kept = []
    for line in template_text.splitlines():
        key = line.split("=", 1)[0].strip()
        if "=" in line and not line.lstrip().startswith("#") and key in OWNED:
            continue
        kept.append(line)
    kept += ["",
             f"# arm {arm} of the tracker A/B (python_ai/tools/setup_tracker_ab.py)",
             f"CLASH_WEIGHTS={os.path.join(arm_dir, 'checkpoints', arm + '.pth')}",
             f"CLASH_LOGDIR={os.path.join(arm_dir, 'runs')}",
             f"CLASH_ARTIFACT_DIR={arm_dir}",
             f"{tracker_config.ENV_VAR}={mode}", ""]
    return "\n".join(kept)


def resume_path(arm_dir, arm):
    """Where train_selfplay.py resumes from for this arm's CLASH_WEIGHTS."""
    from python_ai.trainers.train_selfplay import selfplay_paths
    saved = {k: os.environ.get(k) for k in ("CLASH_WEIGHTS", "CLASH_LOGDIR")}
    os.environ["CLASH_WEIGHTS"] = os.path.join(arm_dir, "checkpoints", arm + ".pth")
    os.environ["CLASH_LOGDIR"] = os.path.join(arm_dir, "runs")
    try:
        return selfplay_paths()[0]
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="the A/B folder to create (runs/ab_<name>)")
    ap.add_argument("--start", required=True, help="the phase-2 checkpoint both arms start from")
    ap.add_argument("--shared-pool", required=True, help="the shared historical_checkpoints/ folder")
    ap.add_argument("--env-template", required=True, help="an env file whose settings both arms copy")
    ap.add_argument("--seed", type=int, default=0, help="seed of the tracker branch's first layer")
    a = ap.parse_args(argv)

    root, start = os.path.abspath(a.root), os.path.abspath(a.start)
    shared, template = os.path.abspath(a.shared_pool), os.path.abspath(a.env_template)
    if os.path.exists(root):
        raise SystemExit(f"!! {root} exists; refusing to reuse it.")
    for p, what in ((start, "starting checkpoint"), (template, "env template")):
        if not os.path.isfile(p):
            raise SystemExit(f"!! no such {what}: {p}")
    if not glob.glob(os.path.join(shared, "*.pth")):
        raise SystemExit(f"!! no *.pth in the shared pool {shared}")

    ck = torch.load(start, map_location="cpu", weights_only=False)
    lineage = float(ck.get("lineage_started_at", 0.0)) or None
    episode = int(ck.get("episodes_completed", 0))
    template_text = open(template).read()

    os.makedirs(root)
    manifest = {"root": root, "start": start, "shared_pool": shared, "template": template,
                "lineage_started_at": lineage, "start_episode": episode, "arms": {}}
    for arm, mode in ARMS:
        arm_dir = os.path.join(root, arm)
        os.makedirs(os.path.join(arm_dir, "checkpoints"))
        env_path = os.path.join(root, arm + ".env")
        with open(env_path, "w") as f:
            f.write(env_text(template_text, arm_dir, arm, mode))

        dst = resume_path(arm_dir, arm)
        print(f"[{arm}] migrating to the {mode} layout -> {dst}")
        atomic_save(migrate.migrate(ck, mode, a.seed), dst)
        stats = migrate.verify(start, dst, mode)
        print(f"[{arm}] verified: {stats}")

        arm_pool = os.path.join(arm_dir, "historical_checkpoints")
        names = link_pool(shared, arm_pool)
        eligible = verify_pool(shared, arm_pool, lineage, episode)
        print(f"[{arm}] {len(names)} links; {len(eligible)} eligible for the lineage filter "
              "(same set as the shared pool)")
        manifest["arms"][arm] = {"mode": mode, "dir": arm_dir, "env": env_path,
                                 "checkpoint": dst, "pool": arm_pool, "links": names}
    with open(os.path.join(root, "ab_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    print(f"OK: {root}")
    return manifest


if __name__ == "__main__":
    main()
