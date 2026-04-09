from pathlib import Path
from typing import Any
from dataclasses import dataclass
import logging
import copy

import numpy as np

import gymnasium as gym

from ssenvs.problem import ProblemData
from ssenvs.envs.poss import State, EnvSpecType, ObservationSpaceType, ActionSpaceType, InfoType

from abstract_dbns.full_graph import JobScheduling as JobSchedulingBeliefState
from abstract_dbns.full_graph.job_scheduling_elements import obtain_machine_usage_levels, obtain_job_probability_pending
from abstract_dbns.full_graph import JobParams 
from abstract_dbns.full_graph import MachineParams 
from abstract_dbns.full_graph import Job 
from abstract_dbns.common.propagate import propagate

logger = logging.getLogger(__name__)

class PredictionEnv(gym.Env):
    """
    Belief State Predictive Environment.
    """
    metadata = {'render.modes': []}

    def __init__(self):
        """
        Initialize the environment
        """
        self.trace: list[ObservationSpaceType] = []
        self.current_time_step: int = 0
        self.elapsed: np.ndarray | None = None
        self.t_max: int | None = None
        self.feedback: dict[tuple[int, int], float] = {}

    def reset(self, seed: int | None = None,
              options: dict[str, Any] | None = None) -> tuple[ObservationSpaceType, InfoType]:
        """
        Resets the environment to an initial internal state
        """
        super().reset(seed=seed, options=options)

        if options is None:
            raise ValueError(f"Prediction environment requires initial state to be provided as a "
                             f"key in the options dictionary.")

        self.problem: JobSchedulingBeliefState = options.get('initial')
        self.current_time_step = 0 
        self.t_max = options.get('t_max')

        self.feedback = {}

        # TODO: Check indexing here.
        mu = {
            j: np.eye(200)[self.problem.jobs[j].set_map[self.problem.jobs[j].params.state_prior]] for j in range(len(self.problem.jobs))
        }

        # Machine utilisation
        machine_utilisation = obtain_machine_usage_levels(self.problem, mu)


        self.trace = [(mu, machine_utilisation)]


        return self.trace[-1], self._get_info()
    
    def step(self, action: list[tuple[int, int]]) -> tuple[ObservationSpaceType, InfoType, float, bool, bool]:
        """
        Steps the environment
        action is an assigment from jobs to machines
        """
        self.current_time_step += 1

        if self.current_time_step == self.t_max:
            logging.debug(f"Done with rollout")
            return self.trace[-1], self._get_info(), self._get_reward(), True, False

        mu_t, m_utils_t = self.trace[-1]
        job_idx = [j.params.name for j in self.problem.jobs]
        action_dict = {j_idx: 0 for j_idx in job_idx}

        # Translate to internal representation for DBN. We use 0 as a dummy action. I.e., no action.
        for j, i in action:
            # TODO check indexing.
            action_dict[j] = i + 1

        trans_models = {}
        for j, i in action_dict.items():
            machine_utilisation = m_utils_t[i - 1]
            trans_models[j] = self.problem.jobs[j].get_transition_model(action=i, time_step = self.current_time_step, machine_utilisation=machine_utilisation) 

        # Propagate transitions
        mu_t_plus_1 = {}
        for j in job_idx:
            mu_t_plus_1[j] = np.zeros_like(mu_t[j])
            mu_t_plus_1[j] = propagate(mu_t[j], trans_models[j])

        m_utils_t_plus_1 = obtain_machine_usage_levels(self.problem, mu_t_plus_1)

        self.trace.append((mu_t_plus_1, m_utils_t_plus_1))

        return self.trace[-1], self._get_info(), 0.0, False, False   

    def _get_reward(self) -> float:
        """
        Returns reward
        """
        mu_t, _ = self.trace[-1]
        objective = 0
        for job in self.problem.jobs:
            completed_prob = mu_t[job.param.name][1]
            objective += job.params.value * completed_prob
        return objective

    def _get_info(self) -> InfoType:
        return {}

    def _calc_available_arms(self) -> list[tuple[int, int]]:
        applicable: list[tuple[int, int]] = []
        for j in self.pending:
            for i in self.idle:
                if self.current_time_step + self.instance.process_times[j, i] < self.deadlines[j]:
                    applicable.append((j, i))

        return applicable



