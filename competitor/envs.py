"""Competition env subclasses that install MDP v1.

Import these from training / evaluation code. Do not change
``bluesky_gym.envs.competition_env.CompetitionEnv`` itself.
"""

from bluesky_gym.envs.competition_env import CompetitionEnv
from bluesky_zoo.competition.competition import CompetitionZooEnv

from competitor.mdp import CompetitionMdpV1, mdp_v1_init_kwargs


class CompetitionEnvV1(CompetitionMdpV1, CompetitionEnv):
    """Single-agent competition env with observation/action/reward v1."""

    def __init__(self, **kwargs):
        super().__init__(**mdp_v1_init_kwargs(kwargs))
        self._setup_mdp_v1(n_neighbors=self.n_intruders)


class CompetitionZooEnvV1(CompetitionMdpV1, CompetitionZooEnv):
    """Multi-agent competition env with the same MDP v1 hooks as the SA env."""

    def __init__(self, **kwargs):
        super().__init__(**mdp_v1_init_kwargs(kwargs))
        self._setup_mdp_v1(n_neighbors=self.n_agents - 1)
