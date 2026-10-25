"""Competition env subclasses that install MDP v1.

Import these from training / evaluation code. Do not change
``bluesky_gym.envs.competition_env.CompetitionEnv`` itself.

``mdp_overrides`` replaces any of ``competitor.config.TUNABLE_KEYS`` for this
instance only (reward weights, action scales, ``ACTION_MODE``).
"""

from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_zoo.competition.competition import CompetitionZooEnv

from competitor.config import resolve_mdp_config
from competitor.mdp import CompetitionMdpV1, mdp_v1_init_kwargs


class CompetitionEnvV1(CompetitionMdpV1, CompetitionEnv):
    """Single-agent competition env with observation/action/reward v1."""

    def __init__(self, mdp_overrides=None, **kwargs):
        cfg = resolve_mdp_config(mdp_overrides)
        super().__init__(**mdp_v1_init_kwargs(kwargs, cfg))
        self._setup_mdp_v1(n_neighbors=self.n_intruders, mdp_cfg=cfg)


class CompetitionZooEnvV1(CompetitionMdpV1, CompetitionZooEnv):
    """Multi-agent competition env with the same MDP v1 hooks as the SA env."""

    def __init__(self, mdp_overrides=None, **kwargs):
        cfg = resolve_mdp_config(mdp_overrides)
        super().__init__(**mdp_v1_init_kwargs(kwargs, cfg))
        self._setup_mdp_v1(n_neighbors=self.n_agents - 1, mdp_cfg=cfg)
