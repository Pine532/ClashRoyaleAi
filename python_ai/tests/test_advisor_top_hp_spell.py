"""The advisor's placement rule for a spell that hits only the highest-HP few.

Lightning strikes the three highest-HP enemies whose bodies overlap its area,
towers and buildings included (AreaSpell's top-HP rule), and deals a tower only
its Crown Tower damage. Scoring it like a Fireball (every enemy in the area)
overvalues a clump of four or more and ignores the slot a tower takes. Every
expected value below is what the engine deals when the spell is cast there.
"""
import numpy as np
import pytest

import clash_royale_env as E
from python_ai import deck as D
from python_ai.advisors import advisor_target as AT
from python_ai.advisors import card_probes, tactics

CE = E.ClashRoyaleEnv
HAND = CE.HAND_SIZE
HOLD = 300                     # deploy ticks: the targets stand still


def _card(name):
    return next(c for c in E.get_all_card_ids() if E.get_card_info(c)["name"] == name)


LIGHTNING = _card("Lightning")
KNIGHT = _card("Knight")
# The Minion Giant deck, whose Lightning this rule was written for.
DECK = D.parse_deck("129,184,126,183,0,31,24,101")


def test_the_target_cap_is_measured_not_keyed():
    """Lightning and Vines hit three; Fireball and Poison hit everyone; Void,
    whose damage per target falls as the crowd grows, hits everyone too.
    """
    assert card_probes.spell_max_targets(LIGHTNING) == 3
    assert card_probes.spell_max_targets(_card("Vines")) == 3
    for name in ("Fireball", "Poison", "Arrows", "Zap", "Void"):
        assert card_probes.spell_max_targets(_card(name)) is None, name
    assert card_probes.spell_max_targets(KNIGHT) is None


def test_a_tower_is_reached_from_further_than_a_troop():
    """The tower's body is wider than a Knight's, and the disc hits what it
    overlaps.
    """
    reach = card_probes.spell_tower_reach(LIGHTNING, 1)
    radius, _ = card_probes.spell_effect(LIGHTNING)
    assert reach > radius
    assert card_probes.spell_tower_reach(KNIGHT) == 0.0


def _board(knights):
    """A fresh match with enemy Knights held at `knights`, one per cell centre."""
    env = CE(DECK, DECK, 3600)
    env.reset()
    for x, y in knights:
        env.inject(KNIGHT, x, y, 1, -1.0, HOLD)
    env.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    # Control: every Knight sits in its own cell, with more HP than one strike,
    # so each strike removes exactly the spell's damage.
    count = tactics.spatial(obs)[tactics.CH_ENEMY_COUNT] * tactics.MAX_CELL_UNITS
    _, damage = card_probes.spell_effect(LIGHTNING)
    for x, y in knights:
        assert round(float(count[int(y), int(x)])) == 1, (x, y)
        assert tactics.enemy_hp_map(obs)[int(y), int(x)] > damage, (x, y)
    return env, obs


def _engine_value(env, ax, ay):
    """Troop HP plus tower HP one Lightning cast at cell (ax, ay) removes, and
    the HP each tower lost.
    """
    towers = lambda: [max(0, env.get_tower_hp(1, s)) for s in range(3)]  # noqa: E731
    before_troops, before_towers = env.get_troop_damage_dealt(0), towers()
    env.inject(LIGHTNING, float(ax), float(ay), 0, -1.0, 0)
    read = lambda e: (e.get_troop_damage_dealt(0), tuple(towers()))  # noqa: E731
    troops, after = card_probes._settled_damage(env, read)
    lost = [b - a for b, a in zip(before_towers, after)]
    return troops - before_troops + sum(lost), lost


def _top_map(obs):
    radius, damage = card_probes.spell_effect(LIGHTNING)
    return tactics.spell_top_targets_map(
        obs, radius, damage, card_probes.spell_max_targets(LIGHTNING),
        card_probes.spell_tower_damage(LIGHTNING),
        [card_probes.spell_tower_reach(LIGHTNING, s) for s in range(3)])


# Five Knights on open ground, out of every tower's reach from (9, 21).
CLUMP = [(8.5, 21.5), (9.5, 21.5), (10.5, 21.5), (8.5, 20.5), (9.5, 20.5)]
# Four Knights in front of the left Princess Tower; (3, 24) reaches the tower.
AT_TOWER = [(1.5, 23.5), (2.5, 23.5), (3.5, 23.5), (4.5, 23.5)]
# Three Knights in front of the King Tower; (8, 29) reaches the King only.
AT_KING = [(7.5, 27.5), (8.5, 27.5), (9.5, 27.5)]


def test_a_clump_of_five_scores_three_strikes():
    env, obs = _board(CLUMP)
    radius, damage = card_probes.spell_effect(LIGHTNING)
    score = float(_top_map(obs)[21, 9])
    dealt, lost = _engine_value(env, 9, 21)
    assert lost == [0, 0, 0]
    assert dealt == pytest.approx(3 * damage)
    assert score == pytest.approx(dealt, abs=1.0)
    # The Fireball model counts all five.
    fireball_model = float(tactics.spell_catch_map(obs, radius, 0, damage)[21, 9])
    assert fireball_model == pytest.approx(5 * damage, abs=1.0)


def test_a_princess_tower_takes_a_slot_and_adds_its_chip():
    env, obs = _board(AT_TOWER)
    _, damage = card_probes.spell_effect(LIGHTNING)
    chip = card_probes.spell_tower_damage(LIGHTNING)
    score = float(_top_map(obs)[24, 3])
    dealt, lost = _engine_value(env, 3, 24)
    assert lost == [0, chip, 0]                       # the left Princess Tower
    assert dealt == pytest.approx(2 * damage + chip)
    assert score == pytest.approx(dealt, abs=1.0)


def test_a_sleeping_king_takes_a_slot_but_scores_nothing():
    """Hitting a sleeping King wakes it, which the chip does not pay for."""
    env, obs = _board(AT_KING)
    _, damage = card_probes.spell_effect(LIGHTNING)
    score = float(_top_map(obs)[29, 8])
    dealt, lost = _engine_value(env, 8, 29)
    assert lost[0] > 0 and lost[1:] == [0, 0]         # the King was struck
    assert dealt - lost[0] == pytest.approx(2 * damage)
    assert score == pytest.approx(2 * damage, abs=1.0)


@pytest.fixture
def lightning_rule(monkeypatch):
    """Lightning as an advisor card, the way the Minion Giant deck derives it."""
    kind = AT.advisor_cards_for(DECK)[LIGHTNING]
    assert kind == "spell"
    monkeypatch.setitem(AT.ADVISOR_CARDS, LIGHTNING, kind)


def test_the_target_prefers_three_in_the_open_to_four_at_a_tower(net, lightning_rule):
    """Under the Fireball model the four Knights at the tower win (4 strikes vs
    3); under the engine's rule the tower takes a slot and the open three win.
    """
    _, obs = _board(CLUMP[:3] + AT_TOWER)
    legal = net._placement_legal[LIGHTNING].numpy().astype(bool)
    t = AT.target_logits_for(obs, LIGHTNING, legal)
    assert t is not None
    assert legal[21 * tactics.BOARD_W + 9] and legal[24 * tactics.BOARD_W + 3]
    assert t[21 * tactics.BOARD_W + 9] > t[24 * tactics.BOARD_W + 3]


def test_the_tower_chip_alone_does_not_make_the_rule_speak(net, fresh_obs, lightning_rule):
    """With no enemy troop or building, the only score is a tower's chip, the
    same in every such state: a constant, which the gate exists to refuse.
    """
    _env, obs = fresh_obs
    obs = np.asarray(obs, dtype=np.float32)
    legal = net._placement_legal[LIGHTNING].numpy().astype(bool)
    assert float(_top_map(obs).max()) > 0.0          # the chip is there
    assert AT._advisor_logits_for(obs, LIGHTNING, legal) is None


def test_the_king_is_reached_from_further_than_a_princess_tower():
    """The King's body is wider; each tower's reach is measured on that tower."""
    king, left, right = (card_probes.spell_tower_reach(LIGHTNING, s) for s in range(3))
    assert left == right
    assert king > left


def test_the_king_takes_a_slot_from_beyond_a_princess_towers_reach():
    """Aimed 5.15 tiles from the King's centre: inside the King's reach, outside
    a Princess Tower's. The left Princess Tower is down, so the King is awake and
    its chip counts.
    """
    env = CE(DECK, DECK, 3600)
    env.reset()
    env.destroy_tower(1, 1)
    knights = [(4.5, 25.5), (5.5, 25.5), (6.5, 25.5)]
    for x, y in knights:
        env.inject(KNIGHT, x, y, 1, -1.0, HOLD)
    env.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    _, damage = card_probes.spell_effect(LIGHTNING)
    chip = card_probes.spell_tower_damage(LIGHTNING)
    score = float(_top_map(obs)[26, 6])
    dealt, lost = _engine_value(env, 6, 26)
    assert lost[0] == chip and lost[2] == 0           # the King was struck
    assert dealt == pytest.approx(2 * damage + chip)
    assert score == pytest.approx(dealt, abs=1.0)


def test_a_building_on_a_fallen_towers_cell_is_still_an_enemy():
    """Only a live tower's cell is the tower; once it falls, a building placed
    there is a body like any other.
    """
    cannon = next(c for c in E.get_all_card_ids()
                  if E.get_card_info(c)["name"] == "Cannon")
    env = CE(DECK, DECK, 3600)
    env.reset()
    env.destroy_tower(1, 1)
    x, y = 3.0, 27.0                                  # the left Princess Tower's cell
    env.inject(cannon, x, y, 1, -1.0, 0)
    env.step_self_play(HAND, 0.0, 0.0, HAND, 0.0, 0.0, 1)
    obs = np.asarray(env.get_observation_for_team(0), dtype=np.float32)
    ys, xs, hp = tactics.enemy_bodies(obs)
    assert (int(y), int(x)) in set(zip(ys.tolist(), xs.tolist()))
    # A live tower's cell holds the tower, which is scored from its scalar.
    _env, fresh = _board([])
    assert tactics.enemy_bodies(fresh)[0].size == 0
