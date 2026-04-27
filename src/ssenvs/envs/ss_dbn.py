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

from collections import defaultdict

logger = logging.getLogger(__name__)

PENDING = 0
COMPLETED = 1
FAILED = 2

@dataclass
class BeliefState(object):
    js_problem: JobSchedulingBeliefState
    mu: dict[int, np.ndarray]
    machine_utils: np.ndarray
    time_step: int

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

        state: BeliefState = options.get('initial')
        self.current_time_step = state.time_step
        self.t_max = options.get('t_max')

        self._app_mask = np.zeros((len(state.js_problem.machines) + 1, len(state.js_problem.jobs)), dtype=np.bool)

        self.feedback = {}

        self.trace = [state]

        self.jobs_completed_monitor = np.array([state.mu[j][1] for j in range(len(state.js_problem.jobs))])
        self.log = {}
        
        return self.trace[-1], self._get_info()
    
    def step(self, action: list[tuple[int, int]]) -> tuple[ObservationSpaceType, InfoType, float, bool, bool]:
        """
        Steps the environment
        action is an assigment from jobs to machines
        """

        # Change action to regular format.
        action = [(t[1], t[0]) for t in zip(*np.nonzero(action)) if t[0] < len(self.trace[-1].js_problem.machines)]
        #print(action)

        self.current_time_step += 1
        self.feedback = {}

        if self.current_time_step == self.t_max:
            logging.debug(f"Done with rollout")
            info = self._get_info()
            info["p_F"] = self.p_f_geq_0_dp()
            info["final_return"] = self._evaluate_remaining_return()
            return self.trace[-1], info, self._get_reward(), True, False

        state: BeliefState = self.trace[-1]
        action_dict = {j_idx: 0 for j_idx in range(len(state.js_problem.jobs))}

        # Translate to internal representation for DBN. We use 0 as a dummy action. I.e., no action.
        for j, i in action:
            # TODO check indexing.
            action_dict[j] = i + 1

        trans_models = {}
        for j_idx in range(len(state.js_problem.jobs)):
            act = action_dict[j_idx]
            trans_models[j_idx] = state.js_problem.jobs[j_idx].get_transition_model(action=act, time_step = self.current_time_step, machine_utilisation=state.machine_utils) 

        # Propagate transitions
        mu_t_plus_1 = {}
        for j in range(len(state.js_problem.jobs)):
            mu_t_plus_1[j] = np.zeros_like(state.mu[j])
            mu_t_plus_1[j] = propagate(state.mu[j], trans_models[j])
        
        m_utils_t_plus_1 = obtain_machine_usage_levels(state.js_problem, mu_t_plus_1)

        self.trace.append(BeliefState(
            js_problem=state.js_problem, 
            mu =  mu_t_plus_1,
            machine_utils = m_utils_t_plus_1, 
            time_step = self.current_time_step
        ))

        return self.trace[-1], self._get_info(), self._get_reward(), False, False   

    def _get_reward(self) -> list[tuple[int, float]]: 
        """
        Returns rewards
        """
        rewards: dict[int, list[tuple[int, float]]] = {}
        s_t_minus_1: BeliefState = self.trace[-2]
        s_t: BeliefState = self.trace[-1]
        for j_idx, j in enumerate(s_t.js_problem.jobs):
            for exe in j.execution_to_remove: 
                set_idx = j.current_executions[exe].keywords.get("working_set_index")
                rew = s_t_minus_1.mu[j_idx][set_idx] * j.params.value
                self.jobs_completed_monitor[j_idx] += s_t_minus_1.mu[j_idx][set_idx]
                start_time = j.current_executions[exe].keywords.get("start_time")
                m_idx = int(j.current_executions[exe].keywords["machine"])
                rewards[start_time - 1] = rewards.get(start_time - 1, []) + [(j_idx, m_idx, rew)]
            if self.current_time_step == j.params.deadline + 1: 
                # TODO: Not currently clear how to do this
                pass
                #rew = s_t_minus_1.mu[j_idx][FAILED] * j.params.value
                #rewards.append((j_idx, -rew))
        return rewards

    def _calc_available_arms(self) -> list[tuple[int, int]]:
        js_problem, mu_t, m_utils_t, _ = self.trace[-1]
        applicable: list[tuple[int, int]] = []
        for j_idx, j in enumerate(js_problem.jobs):
            for m_idx, m in enumerate(js_problem.machines):
                if self.current_time_step + j.params.t_process[m_idx] < j.params.deadline and m_utils_t[m_idx] < 1.0:
                    applicable.append((j_idx, m.name))
        return applicable
    
    def _get_feedback(self):
        state = self.trace[-1]
        feedback = {}
        for j_idx, j in enumerate(state.js_problem.jobs):
            for exe in j.current_executions.values():
                if state.time_step > exe.keywords["finish_time"]:
                    m_idx = None
                    for exe_tuple, value in j.set_map.items():
                        if value == exe.keywords["working_set_index"]:
                            m_idx = exe_tuple[0]
                            break
                    if m_idx is None:
                        raise ValueError(f"Value {value} is not in j.set_map")
                    key = (j_idx, state.js_problem.machines[m_idx].name)
                    success_prob = state.mu[j_num][1] # 1 == Complete 
                    feedback[key] = float(success_prob)
        return feedback

    def _get_info(self, terminal=False) -> InfoType:
        self._update_available_arms()
        return dict(t=self.current_time_step,
                    feedback=self.feedback, #TODO: is this used?
                    weights=np.array([j.params.value for j in self.trace[-1].js_problem.jobs]),
                    arms=self._app_mask.copy(),
                    elapsed=self.elapsed,
                    p_F=0.0)

    def _update_available_arms(self) -> None:
        self._app_mask = np.zeros_like(self._app_mask, dtype=bool)
        m: int = self._app_mask.shape[0]
        state: BeliefState = self.trace[-1]
        for j_idx, j in enumerate(state.js_problem.jobs):
            for m_idx, m in enumerate(state.js_problem.machines):
                if (
                    self.current_time_step + j.params.t_process[m_idx] < j.params.deadline
                    and state.machine_utils[m_idx] < 1
                    and state.mu[j_idx][PENDING] > 0
                ):
                    self._app_mask[m_idx, j_idx] = True
            self._app_mask[-1, j_idx] = True # Always allow do nothing

    def p_f_geq_0_dp(self):
        state = self.trace[-1]
        thetas = []
        weights = []
        for j_idx, j in enumerate(state.js_problem.jobs):
            thetas.append(state.mu[j_idx][1])
            weights.append(j.params.value)

        dist = defaultdict(float)
        dist[0.0] = 1.0

        for w, theta in zip(weights, thetas):
            new_dist = defaultdict(float)
            for val, prob in dist.items():
                new_dist[val + w] += theta * prob        # X_j = 1
                new_dist[val - w] += (1 - theta) * prob  # X_j = 0
            dist = new_dist

        return sum(p for v, p in dist.items() if v < 0)#, dict(dist)

    def _evaluate_remaining_return(self) -> float:
        """
        Returns reward
        """
        state: BeliefState = self.trace[-1]
        objective = 0
        for j_idx, job in enumerate(state.js_problem.jobs):
            #completed_prob = state.mu[j_idx][1] - self.jobs_completed_monitor[j_idx]
            #objective += job.params.value * completed_prob
            failed_prob = state.mu[j_idx][2]
            objective += - (job.params.value * failed_prob)
        return objective
