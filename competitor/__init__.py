"""Competitor MDP v1 for the AI4REAL-NET BlueSky-Gym competition.

This package does **not** touch the official scoring layer
(``_update_metrics`` / ``_get_info``). It only replaces the three MDP hooks
(observation, action mapping, reward) via subclasses of the shipped
single-agent and multi-agent environments.

See ``competitor.config`` for the v1 constants and ``competitor.envs`` for
the env classes to train and evaluate against.
"""

from competitor.envs import CompetitionEnvV1, CompetitionZooEnvV1

__all__ = ["CompetitionEnvV1", "CompetitionZooEnvV1"]
