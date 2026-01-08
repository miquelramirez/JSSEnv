from pathlib import Path
import numpy as np

class ProblemData(object):

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

        with open(self.instance_path) as instance_file:
            for line_cnt, line_str in enumerate(instance_file, start=1):
                split_data = list(map(int, line_str.split()))

                if line_cnt == 1:
                    self.jobs, self.machines = split_data
                    self.process_times = np.zeros((self.jobs, self.machines), dtype=int)
                    self.process_probs = 0.1*np.ones((self.jobs, self.machines))
                    self.jobs_min_length = 1e20*np.ones(self.jobs)
                    self.jobs_max_length = np.zeros(self.jobs)
                    self.job_weights = np.ones(self.jobs)
                else:
                    assert len(split_data) % 2 == 0 and len(split_data) // 2 == self.machines
                    job_nb = line_cnt - 2
                    for i in range(0, len(split_data), 2):
                        machine, time = split_data[i], split_data[i + 1]
                        self.process_times[job_nb, machine] = int(time)
                        if i // 2 == machine:
                            self.process_probs[job_nb, machine] = 1.0
                        self.jobs_min_length[job_nb] = min(self.jobs_min_length[job_nb], int(time))
                        self.jobs_max_length[job_nb] = max(self.jobs_max_length[job_nb], int(time))

        self.max_time_jobs = max(self.jobs_max_length)
        # Check Problem data correctness
        if self.jobs <= 0:
            raise ValueError(f"Job count must be positive: value {self.jobs}")
        if self.machines <= 1:
            raise ValueError(f"At least two machines are required {self.machines}")

