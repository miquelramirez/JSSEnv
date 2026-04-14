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

        js_problem, mu, machine_utilisation, time_step = options.get('initial')
        self.current_time_step = time_step
        self.t_max = options.get('t_max')

        self.feedback = {}

        self.trace = [(js_problem, mu, machine_utilisation, time_step)]


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

        js_problem, mu_t, m_utils_t, _ = self.trace[-1]
        job_idx_map = {num: j.params.name for num, j in enumerate(js_problem.jobs)}
        action_dict = {j_idx: 0 for j_idx in job_idx_map.values()}

        # Translate to internal representation for DBN. We use 0 as a dummy action. I.e., no action.
        for j, i in action:
            # TODO check indexing.
            action_dict[j] = i + 1

        trans_models = {}
        for j in range(len(js_problem.jobs)):
            job_idx = job_idx_map[j]
            act = action_dict[job_idx]
            trans_models[j] = js_problem.jobs[j].get_transition_model(action=act, time_step = self.current_time_step, machine_utilisation=m_utils_t) 

        # Propagate transitions
        mu_t_plus_1 = {}
        for j in range(len(js_problem.jobs)):
            mu_t_plus_1[j] = np.zeros_like(mu_t[j])
            mu_t_plus_1[j] = propagate(mu_t[j], trans_models[j])

        m_utils_t_plus_1 = obtain_machine_usage_levels(js_problem, mu_t_plus_1)

        self.trace.append((js_problem, mu_t_plus_1, m_utils_t_plus_1, self.current_time_step))

        return self.trace[-1], self._get_info(), 0.0, False, False   

    def _get_reward(self) -> float:
        """
        Returns reward
        """
        js_problem, mu_t, _, _ = self.trace[-1]
        objective = 0
        for j_idx, job in enumerate(js_problem.jobs):
            completed_prob = mu_t[j_idx][1]
            objective += job.params.value * completed_prob
        return objective

    def _get_info(self) -> InfoType:
        return dict(t=self.current_time_step, 
                    arms=self._calc_available_arms(), 
                    )

    def _calc_available_arms(self) -> list[tuple[int, int]]:
        js_problem, mu_t, m_utils_t, _ = self.trace[-1]
        applicable: list[tuple[int, int]] = []
        for j_num, j in enumerate(js_problem.jobs):
            for m_num, m in enumerate(js_problem.machines):
                if self.current_time_step + j.params.t_process[m_num] < j.params.deadline and m_utils_t[m_num] < 1.0:
                    applicable.append((j.params.name, m.name))
        return applicable



