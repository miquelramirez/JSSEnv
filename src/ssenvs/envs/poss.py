from pathlib import Path
from typing import Any
from dataclasses import dataclass
import logging

import numpy as np

import gymnasium as gym

from ssenvs.problem import ProblemData

logger = logging.getLogger(__name__)

# Types
EnvSpecType = dict[str, Any]
ObservationSpaceType = dict[str, Any]
ActionSpaceType = list[tuple[int,int]]
InfoType = dict[str, Any]

@dataclass
class State(object):

    machines: int
    pending: set[int]
    working: set[tuple[int, int]]
    completed: set[int]
    failed: set[int]
    idle: set[int]


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
            self.spec = dict(instance_name="scen001.json")

        if self.spec.get('instance_name') is not None:
            self.instance = ProblemData(Path(__file__).parent.absolute()
                                                / "instances"
                                                / "stochastic"
                                                / f"{self.spec["instance_name"]}.json")
        elif self.spec.get('instance_path') is not None:
            self.instance = ProblemData(self.spec['instance_path'])
        else:
            raise ValueError("Specification needs to either provide an instance name or a path")

        self.trace: list[ObservationSpaceType] = []
        self.current_time_step: int = 0
        self.deadlines: np.ndarray = np.ones(self.instance.jobs, dtype=int)
        self.releases: np.ndarray = np.zeros(self.instance.jobs, dtype=int)
        self.elapsed: np.ndarray = np.zeros(self.instance.jobs, dtype=int)
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
        self._choose_release_and_deadline_times()

        self.completed = set()
        self.failed = set()
        self.working = set()
        self.idle = set([i for i in range(0, self.instance.machines)])

        # put all released jobs out
        self.current_time_step = 0
        self.pending = set([j for j in range(0, self.instance.jobs) if self.releases[j] == self.current_time_step])
        
        # reset elapsed time
        self.elapsed: np.ndarray = np.zeros(self.instance.jobs, dtype=int)

        self.trace = [self._get_obs()]
        self.feedback = dict()

        return self.trace[-1], self._get_info()

    def step(self, action: list[tuple[int, int]]) -> tuple[ObservationSpaceType, InfoType, float, bool, bool]:
        """
        Steps the environment
        """
        self.just_completed = set()
        self.just_failed = set()

        for j, i in action:
            if j not in self.pending:
                raise ValueError(f"Invalid assignment: ({j}, {i}): Job {j} not in pending.")
            if i not in self.idle:
                raise ValueError(f"Invalid assignment: ({j}, {i}): Job {j} not in idle.")

        self.current_time_step += 1

        if self.current_time_step == self.t_max:
            logging.debug(f"End of service interval reached.")
            self.trace.append(self._get_obs())
            return self.trace[-1], self._get_info(), self._get_reward(), True, False

        # Updated working set
        self.feedback = {}
        logging.debug(f"Updating working set: {self.working}")
        next_working: set[tuple[int, int]] = set()
        for j, i in self.working:
            logging.debug(f"Checking status of job {j} being processed by machine {i}")
            if self.current_time_step > self.deadlines[j]:
                self.failed.add(j)
                self.just_failed.add(j)
                self.idle.add(i)
                self.elapsed[j] = 0
                logging.debug(f"Job {j} FAILS: current time: {self.current_time_step}, deadline: {self.deadlines[j]}")
            else:
                if self.elapsed[j] > self.instance.process_times[j, i]:
                    logging.debug(f"Job {j} COMPLETED at machine {i}: current time: {self.current_time_step}, deadline: {self.elapsed[j]}")
                    self.completed.add(j)
                    self.just_completed.add(j)
                    self.idle.add(i)
                    self.feedback[(j,i)] = 1.0
                    self.elapsed[j] = 0
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
            assert j in self.pending
            assert i in self.idle
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
                self.just_failed.add(j)
                self.elapsed[j] = 0
            else:
                next_pending.add(j)
        # Check releases
        for j in range(self.instance.jobs):
            if self.current_time_step == self.releases[j]:
                next_pending.add(j)
        self.pending = next_pending

        self.trace.append(self._get_obs())
        return self.trace[-1], self._get_info(), 0.0, False, False

    def _get_reward(self) -> float:
        """
        Returns reward
        """
        completed_weights: int = 0
        failed_weights: int = 0
        for j in self.just_completed:
            completed_weights += self.instance.job_weights[j]
        for k in self.just_failed:
            failed_weights -= self.instance.job_weights[k]
        return completed_weights + failed_weights

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
        return dict(current_time_step=self.current_time_step,
                    elapsed=self.elapsed,
                    feedback=self.feedback,
                    arms=self._calc_available_arms(),
                    t_max=self.t_max, )

    def _calc_available_arms(self) -> list[tuple[int, int]]:
        applicable: list[tuple[int, int]] = []
        for j in self.pending:
            for i in self.idle:
                if self.current_time_step + self.instance.process_times[j, i] < self.deadlines[j]:
                    applicable.append((j, i))

        return applicable

    def _choose_release_and_deadline_times(self) -> None:
        """
        Choose the release times
        """

        for j in range(self.instance.jobs):
            z = self.np_random.integers(self.instance.job_releases_lo[j], self.instance.job_releases_hi[j] + 1)
            self.releases[j] = z
            self.deadlines[j] = self.releases[j] + self.instance.job_deadlines[j]
        self.t_max = max(self.deadlines)