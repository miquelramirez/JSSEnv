from pathlib import Path
from typing import Any
from dataclasses import dataclass
import logging
import copy

import numpy as np

import gymnasium as gym

from ssenvs.problem import ProblemData
from ssenvs.envs.poss import State, EnvSpecType, ObservationSpaceType, ActionSpaceType, InfoType

logger = logging.getLogger(__name__)

class PredictionEnv(gym.Env):
    """
    Stochastic Scheduling Environment
    """
    metadata = {'render.modes': []}

    def __init__(self, spec: dict[str, Any]):
        """
        Initialize the environment
        """
        self.spec: EnvSpecType = spec
        self.instance = ProblemData(Path(__file__).parent.absolute()
                                            / "instances"
                                            / "stochastic"
                                            / f"{self.spec["instance_name"]}.json")

        self.trace: list[ObservationSpaceType] = []
        self.current_time_step: int = 0
        self.deadlines: np.ndarray | None = None
        self.elapsed: np.ndarray | None = None
        self.t_max: int | None = None

        self.completed: set[int] = set()
        self.completion_times: dict[int, int] = {}
        self.failed: set[int] = set()
        self.idle: set[int] = set()
        self.working: set[tuple[int, int]] = set()
        self.pending: set[int] = set()

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

        s0: State = options.get('initial')

        self.completed = copy.copy(s0.completed)
        self.failed = copy.copy(s0.failed)
        self.working = copy.copy(s0.working)
        self.pending = copy.copy(s0.pending)
        self.idle = copy.copy(s0.idle)
        self.elapsed = copy.copy(options['elapsed'])
        self.current_time_step = options['current_time_step']
        self.t_max = options['t_max']
        self.deadlines = options['deadlines']

        # put all released jobs out

        self.trace = [self._get_obs()]
        self.feedback = {}

        return self.trace[-1], self._get_info()

    def step(self, action: list[tuple[int, int]]) -> tuple[ObservationSpaceType, InfoType, float, bool, bool]:
        """
        Steps the environment
        """

        for j, i in action:
            if j not in self.pending:
                raise ValueError(f"Invalid assignment: ({j}, {i}): Job {j} not in pending.")
            if i not in self.idle:
                raise ValueError(f"Invalid assignment: ({j}, {i}): Machine {i} not in idle.")

        self.current_time_step += 1

        if self.current_time_step == self.t_max:
            logging.debug(f"Done with rollout")
            self.trace.append(self._get_obs())
            return self.trace[-1], self._get_info(), self._get_reward(), True, False

        # Updated working set
        logging.debug(f"Updating working set: {self.working}")
        next_working: set[tuple[int, int]] = set()
        self.feedback = {}

        for j, i in self.working:
            logging.debug(f"Checking status of job {j} being processed by machine {i}")
            if self.current_time_step > self.deadlines[j]:
                self.failed.add(j)
                self.idle.add(i)
                self.elapsed[j] = 0
                logging.debug(f"Job {j} FAILS: current time: {self.current_time_step}, deadline: {self.deadlines[j]}")
            else:
                if self.elapsed[j] > self.instance.process_times[j, i]:
                    logging.debug(f"Job {j} COMPLETED at machine {i}: current time: {self.current_time_step}, deadline: {self.elapsed[j]}")
                    self.completed.add(j)
                    self.idle.add(i)
                    self.elapsed[j] = 0
                    self.feedback[(j, i)] = 1.0 # note we are assuming unit weights
                else:
                    v_ji = self.np_random.random()
                    q_ji = self.instance.process_probs[j, i]
                    logging.debug(f"Test for job {j} processing abortion at machine {i}: disturbance: {v_ji}, probability: {q_ji}")
                    if v_ji > q_ji:
                        logging.debug(f"Job {j} ABORTED at machine {i}: current time: {self.current_time_step}, deadline: {self.deadlines[j]}")
                        self.pending.add(j)
                        self.idle.add(i)
                        self.elapsed[j] = 0
                    else:
                        logging.debug(f"Job {j} ON TRACK at machine {i}: current time: {self.current_time_step}, deadline: {self.deadlines[j]}")
                        self.elapsed[j] += 1
                        next_working.add((j, i))
        for j, i in action:
            v_ji = self.np_random.random()
            q_ji = self.instance.process_probs[j, i]
            logging.debug(f"Test for job {j} successfully starts processing at machine {i}: disturbance: {v_ji}, probability: {q_ji}")
            if v_ji > q_ji:
                logging.debug(f"Job {j} ABORTED upon starting at machine {i}: current time: {self.current_time_step}, deadline: {self.deadlines[j]}")
            else:
                next_working.add((j, i))
                self.elapsed[j] = 1
                self.idle.remove(i)
                self.pending.remove(j)
                logging.debug(f"Adding new entry to working set: job={j}, machine={i}")
        logging.debug(f"Next working set: {next_working}")
        logging.debug(f"Elapsed time: {self.elapsed}")
        self.working = next_working

        next_pending: set[int] = set()
        # Check expired jobs
        for j in self.pending:
            if self.current_time_step > self.deadlines[j]:
                self.failed.add(j)
                self.elapsed[j] = 0
            else:
                next_pending.add(j)
        self.pending = next_pending

        self.trace.append(self._get_obs())
        return self.trace[-1], self._get_info(), 0.0, False, False

    def _get_reward(self) -> float:
        """
        Returns reward
        """
        completed_weights: int = 0
        for j in self.completed:
            completed_weights += self.instance.job_weights[j]
        return completed_weights

    def _get_obs(self) -> ObservationSpaceType:
        """
        Returns current state observation
        """
        return dict(state=State(machines=self.instance.machines,
                                pending=self.pending,
                                completed=self.completed,
                                working=self.working,
                                failed=self.failed,
                                idle=self.idle),
                    jobs_proc_times={j: self.instance.process_times[j, :] for j in self.pending},
                    jobs_probs={j: self.instance.process_probs[j, :] for j in self.pending},
                    jobs_deadlines={j: self.deadlines[j] for j in self.pending},)

    def _get_info(self) -> InfoType:
        return dict(t=self.current_time_step,
                    feedback=self.feedback,
                    arms=self._calc_available_arms(),
                    elapsed=self.elapsed,)

    def _calc_available_arms(self) -> list[tuple[int, int]]:
        applicable: list[tuple[int, int]] = []
        for j in self.pending:
            for i in self.idle:
                if self.current_time_step + self.instance.process_times[j, i] < self.deadlines[j]:
                    applicable.append((j, i))

        return applicable