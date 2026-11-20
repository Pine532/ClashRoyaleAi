"""tools/redeck_checkpoint.py: a new lineage on another deck from trained weights.

The Hog checkpoint (no Champion/Hero) is fine-tuned onto a deck with a Hero at
deck index 1, which adds an ability head. The head shifts the positions of the
parameters after it, so Adam's state must be carried by NAME, or the moments of
one tensor land on another.
"""
import hashlib
import importlib.util
import os

import torch

from python_ai.models.net import MicroRoyaleNet
from python_ai.rl.config import PHASE1_ENTROPY
from python_ai.rl.curriculum import CURRICULUM_STAGES

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HOG = [15, 6, 25, 40, 24, 72, 33, 7]
NEW = [129, 184, 126, 183, 0, 31, 24, 101]


def _tool():
    spec = importlib.util.spec_from_file_location("redeck_checkpoint",
                                                  os.path.join(REPO, "tools", "redeck_checkpoint.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _hog_checkpoint(path):
    torch.manual_seed(0)
    net = MicroRoyaleNet(num_ability_slots=0)
    opt = torch.optim.Adam(net.parameters(), lr=3e-4)
    for p in net.parameters():
        p.grad = torch.randn_like(p)
    opt.step()
    torch.save({
        "model": net.state_dict(), "optimizer": opt.state_dict(),
        "episodes_completed": 142124, "outcome_history": [1, -1, 1], "deck": HOG,
        "clash_settings": {"CLASH_DECK": ",".join(map(str, HOG))},
        "lineage_started_at": 1.0, "ent_coef_card": 0.0286, "ent_coef_place": 0.01,
        "curriculum_stage": 3, "curriculum_demotions": 24, "curriculum_plateau_advances": 23,
        "teacher_table_size": len(CURRICULUM_STAGES), "stage_start_episode": 138603,
        "phase": "mirror", "deck_curriculum_stage": 0, "phase_deck_episode_start": 0,
        "random_phase_episode_start": 0, "current_random_deck": None,
        "rung_history": [0, 1, 1], "best_rung_mean": 0.65, "best_rung_episode": 141800,
        "rung_entry_progress": 0.59, "progress_signal": 0.65,
        "deck_pool_stats": {"hog_26_mirror": 0.88}, "deck_pool_counts": {"hog_26_mirror": 142},
    }, path)
    return net, opt


def test_redeck_keeps_weights_and_adam_by_name_and_resets_the_lineage(tmp_path):
    src, dst = tmp_path / "hog.pth", tmp_path / "new.pth"
    old_net, old_opt = _hog_checkpoint(src)
    src_hash = hashlib.sha256(src.read_bytes()).hexdigest()

    _tool().redeck(str(src), str(dst), NEW)
    out = torch.load(dst, map_location="cpu", weights_only=False)

    # Weights: strict load into the new architecture; shared tensors unchanged.
    new_net = MicroRoyaleNet(num_ability_slots=1)
    new_net.load_state_dict(out["model"])
    old_sd = old_net.state_dict()
    fresh = set(out["redeck"]["fresh_params"])
    assert fresh == {"ability_slot1_head.weight", "ability_slot1_head.bias"}
    for name, t in new_net.state_dict().items():
        if name not in fresh:
            assert torch.equal(t, old_sd[name]), name

    # Adam: by name, every old moment on its own tensor; none for the new head.
    adam = torch.optim.Adam(new_net.parameters(), lr=3e-4)
    adam.load_state_dict(out["optimizer"])
    old_by_name = {n: old_opt.state[p] for n, p in old_net.named_parameters()}
    for name, p in new_net.named_parameters():
        if name in fresh:
            assert p not in adam.state or not adam.state[p], name
        else:
            assert torch.equal(adam.state[p]["exp_avg"], old_by_name[name]["exp_avg"]), name
            assert torch.equal(adam.state[p]["exp_avg_sq"], old_by_name[name]["exp_avg_sq"]), name

    # The lineage restarts.
    assert out["deck"] == NEW
    assert out["episodes_completed"] == 0
    assert out["outcome_history"] == []
    assert out["curriculum_stage"] == 0
    assert out["teacher_table_size"] == len(CURRICULUM_STAGES)
    assert (out["curriculum_demotions"], out["curriculum_plateau_advances"]) == (0, 0)
    assert out["stage_start_episode"] == 0 and out["rung_history"] == []
    assert out["ent_coef_card"] == PHASE1_ENTROPY.initial_coef_card
    assert out["ent_coef_place"] == PHASE1_ENTROPY.initial_coef_placement
    for gone in ("deck_pool_stats", "deck_pool_counts", "clash_settings", "best_rung_mean"):
        assert gone not in out, gone
    assert out["redeck"]["from_deck"] == HOG and out["redeck"]["from_episodes"] == 142124
    assert out["lineage_started_at"] > 1.0

    # The source is only read.
    assert hashlib.sha256(src.read_bytes()).hexdigest() == src_hash
