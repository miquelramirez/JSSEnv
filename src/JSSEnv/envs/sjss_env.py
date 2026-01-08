from pathlib import Path
from typing import Any
from dataclasses import dataclass
import logging

import numpy as np

import gymnasium as gym

from JSSEnv.problem import ProblemData

logger = logging.getLogger(__name__)

# Types
EnvSpecType = dict[str, Any]
ObservationSpaceType = dict[str, Any]
ActionSpaceType = list[tuple[int,int]]
InfoType = dict[str, Any]

@dataclass
class State(object):

    jobs: int
    machines: int
    pending: set[int]
    working: set[tuple[int, int]]
    completed: set[int]
    failed: set[int]


class StochasticEnv(gym.Env):
    """
    Stochastic Scheduling Environment
    """
    metadata = {'render.modes': []}

    def __init__(self, spec: dict[str, Any] | None = None):
        """
        Initialize the environment
        """
        self.spec: EnvSpecType | None = spec
        if self.spec is None:
            self.spec = dict(instance_path=Path(__file__).parent.absolute() / "instances" / "ta80")

        self.instance = ProblemData(self.spec["instance_path"])

        self.trace: list[State] = []
        self.current_time_step: int = 0
        self.release_times = np.zeros_like(self.instance.instance_matrix)

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObservationSpaceType, InfoType]:
        """
        Resets the environment to an initial internal state
        """
        super().reset(seed=seed, options=options)
        self._choose_release_and_deadline_times()


    def _choose_release_and_deadline_times(self) -> None:
        """
        Choose the release times
        """
        pass