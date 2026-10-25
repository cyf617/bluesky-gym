import pytest

from competitor import config
from competitor.config import parse_overrides, resolve_mdp_config


def test_defaults_match_module_constants():
    cfg = resolve_mdp_config()
    assert set(cfg) == set(config.TUNABLE_KEYS)
    assert cfg["PROGRESS_PER_KM"] == config.PROGRESS_PER_KM
    assert cfg["ACTION_MODE"] == "relative"


def test_string_overrides_are_coerced_to_default_type():
    cfg = resolve_mdp_config(parse_overrides(
        ["CONFLICT_WEIGHT=0.3", "ACTION_FREQUENCY=10", "ACTION_MODE=goal_relative"]))
    assert cfg["CONFLICT_WEIGHT"] == pytest.approx(0.3)
    assert cfg["ACTION_FREQUENCY"] == 10 and isinstance(cfg["ACTION_FREQUENCY"], int)
    assert cfg["ACTION_MODE"] == "goal_relative"


def test_overrides_do_not_mutate_module_defaults():
    resolve_mdp_config({"REACH_REWARD": 1.0})
    assert resolve_mdp_config()["REACH_REWARD"] == config.REACH_REWARD


def test_unknown_key_and_bad_mode_raise():
    with pytest.raises(KeyError):
        resolve_mdp_config({"INTRUSION_NM": 3.0})
    with pytest.raises(ValueError):
        resolve_mdp_config({"ACTION_MODE": "absolute"})
    with pytest.raises(ValueError):
        parse_overrides(["CONFLICT_WEIGHT"])


def test_new_safety_and_shield_keys_default_off():
    cfg = resolve_mdp_config()
    for k in ("SECTOR_TIME_WEIGHT", "INTRUSION_EVENT_PENALTY", "SHIELD"):
        assert cfg[k] == 0
    assert resolve_mdp_config({"SHIELD": "1"})["SHIELD"] == 1
    assert resolve_mdp_config({"SHIELD_LOOKAHEAD_S": "60"})["SHIELD_LOOKAHEAD_S"] == 60.0
