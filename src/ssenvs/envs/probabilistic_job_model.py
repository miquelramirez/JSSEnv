from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.inference import DBNInference
import numpy as np
import copy

class Job:

    def __init__(self, 
                 deadline: int = 10, 
                 current_processes: list[dict] = [],
                 prior: dict[str, float] = {"C": 0.0, "P": 1.0, "F": 0.0}, 
                 ):
        self.deadline = deadline
        self.current_processes = current_processes
        self.prior = prior
        self.evidence = {("K", 0): 0, ("K", 1): 0}

    def new_execution(self, machine_info: dict) -> None:
        required_keys = {"id", "elapsed_time", "q_ij", "p_ij"}
        missing = required_keys - machine_info.keys()
        if missing:
            raise KeyError(f"Missing keys in machine_info: {missing}")

        if not isinstance(machine_info["id"], str) or not machine_info["id"]:
            raise ValueError("'id' must be a non-empty string")

        if not isinstance(machine_info["elapsed_time"], int) or machine_info["elapsed_time"] != 0:
            raise ValueError("'elapsed_time' must be an int equal to 0")

        if not isinstance(machine_info["q_ij"], float) or not (0 < machine_info["q_ij"] <= 1):
            raise ValueError("'q_ij' must be a float in (0, 1]")

        if not isinstance(machine_info["p_ij"], int) or machine_info["p_ij"] < 0:
            raise ValueError("'p_ij' must be a non-negative int")
        
        self.current_processes.append(machine_info)

    def update_job(self) -> None:
        #self.deadline = max(0, self.deadline - 1)
        proc_num = len(self.current_processes)
        for idx in range(proc_num - 1, -1, -1): # count backwards
            if self.current_processes[idx]["elapsed_time"] >= self.current_processes[idx]["p_ij"]:
                # Remove job if it has completed 
                self.current_processes.pop(idx)
            else:
                # Count processing steps
                self.current_processes[idx]["elapsed_time"] += 1
        return self.deadline == 0 # Returns true if deadline is 0
    
    def update(self, posterior: dict[str, float]):
        self.prior = posterior
        self.prior = {
            k: v for k, v in self.prior.items() if v > 0 or k in ["C", "P", "F"]
        }
        self.update_job()
    
    def update_prior(self) -> dict[str, float]:
        for proc in self.current_processes:
            key = "W_%s,%i"%(proc["id"], proc["elapsed_time"])
            self.prior[key] = self.prior.get(key, 0.0)
        return self.prior
    
    def get_new_execution(self) -> str:
        for proc in self.current_processes:
            if proc["elapsed_time"] == 0:
                return True, proc
        return False, None
    
    def get_midway(self) -> list:
        """
        Return all proc that are mid-way through
        """
        return [proc for proc in self.current_processes if proc["elapsed_time"] > 0 and proc["elapsed_time"] <  proc["p_ij"]]
    
    def get_final(self) -> list:
        """
        Return all proc that are mid-way through
        """
        return [proc for proc in self.current_processes if proc["elapsed_time"] == proc["p_ij"]]

def job_to_transition_matrix(job: Job):

    # Index 1 is the X1, index 2 is the X0
    prior = job.prior

    state = list(prior.keys())
    transition_matrix = np.zeros((len(state), len(state) * 2))
    # C 
    # No Failure
    transition_matrix[state.index("C"), (state.index("C") * 2)] = 1.0
    # Failure
    transition_matrix[state.index("C"), (state.index("C") * 2) + 1] = 1.0
    # P
    new_exe, proc = job.get_new_execution()
    if new_exe:
        # No failure
        working_set = "W_%s,%i"%(proc["id"], proc["elapsed_time"])
        transition_matrix[state.index("P"), state.index("P")*2] = 1 - proc["q_ij"]
        transition_matrix[state.index(working_set), state.index("P")*2] = proc["q_ij"]
        #transition_matrix[state.index(working_set), state.index(working_set)*2] = 1.0
        # Failure
        transition_matrix[state.index("F"), (state.index("P")*2)+1] = 1
    else:
        # No Failure
        transition_matrix[state.index("P"), state.index("P") * 2] = 1
        # Failure
        transition_matrix[state.index("F"), (state.index("P") * 2) + 1] = 1
    # F
    # No failure
    transition_matrix[state.index("F"), state.index("F") * 2] = 1.0
    # Failure
    transition_matrix[state.index("F"), (state.index("F") * 2) + 1] = 1.0
    # W
    for proc in job.get_midway():
        working_set_old = "W_%s,%i"%(proc["id"], proc["elapsed_time"] - 1)
        working_set_new = "W_%s,%i"%(proc["id"], proc["elapsed_time"])
        # No failure
        transition_matrix[state.index(working_set_new), state.index(working_set_old) * 2] = proc["q_ij"]
        transition_matrix[state.index("P"), state.index(working_set_old) * 2] = 1 - proc["q_ij"]
        # Failure
        transition_matrix[state.index(working_set_new), (state.index(working_set_old) * 2) + 1] = proc["q_ij"]
        transition_matrix[state.index("F"), (state.index(working_set_old) * 2) + 1] = 1 - proc["q_ij"]
    for proc in job.get_final():
        working_set = "W_%s,%i"%(proc["id"], proc["elapsed_time"] - 1)
        # No failure
        transition_matrix[state.index("C"), state.index(working_set) * 2] = proc["q_ij"]
        transition_matrix[state.index("P"), state.index(working_set) * 2] = 1 - proc["q_ij"]
        # Failure
        transition_matrix[state.index("C"), (state.index(working_set) * 2)+1] = proc["q_ij"]
        transition_matrix[state.index("F"), (state.index(working_set) * 2)+1] = 1 - proc["q_ij"]

    for i in range(transition_matrix.shape[-1]):
        if transition_matrix[:, i].sum() != 1:
            transition_matrix[i // 2, i] = 1
    return transition_matrix

def build_dbn(job: Job) -> DBN:
    """
    A complex job model that allows for multiple executions to run simultaneously. 
    The set of states is inferred dynamically. 
    There isn't a notion of action, only failure.
    """
    # Returns a 
    prior: dict[str, float] = job.prior

    # DBN init
    dbnet = DBN()

    # Define edges
    dbnet.add_edges_from([
    (("X", 0), ("X", 1)),   # state persistence
    (("K", 0), ("X", 1)),   # action affects next state
    (("K", 0), ("K", 1)),
    ])

    # Initial distribution 
    # TODO: check this works
    x0_cpd = TabularCPD(
        variable=("X", 0),
        variable_card=len(prior),
        values=np.array([[float(k)] for k in prior.values()])
    )

    # Define whether job has failed or not
    k0_indicator = TabularCPD(
        variable=("K", 0),
        variable_card=2,
        values=np.ones((2,1), dtype=float)/2,
    )

    # Policy template at t=1: P(A_1 | X_1). Necessary for 2TBN
    k1_indicator = TabularCPD(
        variable=("K", 1),
        variable_card=2,
        values=np.ones((2, 2), dtype=float)/2,
        evidence=[("K", 0)],
        evidence_card=[2],
    )
    x_trans = TabularCPD(
        variable=("X", 1),
        variable_card=len(job.prior),
        values=job_to_transition_matrix(job),
        evidence=[("X", 0), ("K", 0)],
        evidence_card=[len(prior), 2],
    )

    dbnet.add_cpds(x0_cpd, k0_indicator, k1_indicator, x_trans)
    dbnet.initialize_initial_state()
    dbnet.check_model()
    return dbnet

def build_and_execute_dbn(job: Job):
    """
    Builds and executes a DBN, obtaining the next posterior and updating the job
    
    Args:
        job (Job): A job object containing current processes
    """
    # Update prior
    job.update_prior()
    # Build DBN
    dbn = build_dbn(job)
    # Build inference object
    infer = DBNInference(dbn)
    # Conduct inference
    q5 = infer.query(variables=[("X", 1)], evidence=job.evidence)
    posterior = q5[("X", 1)].values
    posterior = {k: float(v) for k, v in zip(job.prior.keys(), posterior)}
    # Update job, including process states
    job.update(posterior)
    return job

if __name__ == "__main__":
    job = Job(deadline=5, 
          current_processes=[], 
          )

    job.new_execution({
        "id": str(1), 
        "elapsed_time": 0, 
        "q_ij": 0.64, 
        "p_ij": 2,
    })

    for i in range(10):
        print(i)
        job.update_prior()
        job = build_and_execute_dbn(job) 
        if i >= job.deadline:
            print("Deadline exceeded")
            job.evidence = {
                ("K", 0): 1, ("K", 1): 1
            }

        print(job.prior)
