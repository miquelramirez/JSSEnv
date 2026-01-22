from pathlib import Path
import numpy as np
import json

class ProblemData(object):
    job_failure_rate: float = 0.1
    job_deadline_multiplier: int = 2

    def __init__(self, data: Path):
        self.instance_path = Path(data)
        # Scheduling problem data
        self.jobs: int = 0
        self.machines: int = 0
        self.process_times: np.ndarray | None = None
        self.process_probs: np.ndarray | None = None
        self.jobs_min_length: np.ndarray | None = None
        self.jobs_max_length: np.ndarray | None = None
        self.job_weights: np.ndarray | None = None
        self.job_deadlines: np.ndarray | None = None
        self.job_releases_lo: np.ndarray | None = None
        self.job_releases_hi: np.ndarray | None = None

        with open(self.instance_path) as instance_file:
            data: dict = json.load(instance_file)

            self.jobs = len(data.get('jobs'))
            self.machines = len(data.get('machines'))
            self.process_times = np.zeros((self.jobs, self.machines), dtype=int)
            self.process_probs = np.ones((self.jobs, self.machines))
            self.jobs_min_length = 1e20 * np.ones(self.jobs)
            self.jobs_max_length = np.zeros(self.jobs)
            self.job_weights = np.ones(self.jobs)
            self.job_deadlines = np.ones(self.jobs)
            self.job_releases_lo = np.zeros(self.jobs)
            self.job_releases_hi = np.zeros(self.jobs)

            for j, job in enumerate(data.get('jobs')):
                self.job_weights[j] = float(job.get('weight'))
                self.job_deadlines[j] = float(job.get('deadline'))
                self.job_releases_lo[j] = float(job.get('release')[0])
                self.job_releases_hi[j] = float(job.get('release')[1])

            for i, machine in enumerate(data.get('machines')):
                if len(machine.get('times')) != self.jobs:
                    raise RuntimeError(f"Machine lists {len(machine.get('times'))} processing times, "
                                       f"but we have {self.jobs} jobs")
                for j, p_ij in enumerate(machine.get('times')):
                    self.process_times[j, i] = p_ij
                # Note that we model probability of job being aborted by the machine
                if len(machine.get('probs')) != self.jobs:
                    raise RuntimeError(f"Machine lists {len(machine.get('probs'))} abort probabilities, "
                                       f"but we have {self.jobs} jobs")
                for j, q_ij in enumerate(machine.get('probs')):
                    self.process_probs[j, i] = 1 - q_ij

        self.max_time_jobs = max(self.jobs_max_length)
        # Check Problem data correctness
        if self.jobs <= 0:
            raise ValueError(f"Job count must be positive: value {self.jobs}")
        if self.machines <= 1:
            raise ValueError(f"At least two machines are required {self.machines}")

