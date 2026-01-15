import ssenvs.envs.instances as jsslib
import ssenvs.problem as io

def test_load_ta_01() -> None:

    instance = io.ProblemData(jsslib.get_path() / "stochastic" / "scen001.json")
    print(f"Jobs in instance: {instance.jobs}")
    print(f"Machines in instance: {instance.machines}")
    for j in range(instance.jobs):
        print(f"Job {j} times: {instance.process_times[j,:]}")
        print(f"Job {j} completion probs: {instance.process_probs[j, :]}")
        print(f"Min time: {instance.jobs_min_length[j]} max time: {instance.jobs_max_length[j]}")

