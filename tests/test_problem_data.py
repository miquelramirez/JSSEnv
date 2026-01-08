import JSSEnv.envs.instances as jsslib
import JSSEnv.problem as io

def test_load_ta_01() -> None:

    instance = io.ProblemData(jsslib.get_path() / "ta13")
    print(f"Jobs in instance: {instance.jobs}")
    print(f"Machines in instance: {instance.machines}")
    for j in range(instance.jobs):
        print(f"Job {j} times: {instance.process_times[j,:]}")
        print(f"Job {j} completion probs: {instance.process_probs[j, :]}")
        print(f"Min time: {instance.jobs_min_length[j]} max time: {instance.jobs_max_length[j]}")

