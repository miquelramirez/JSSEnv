import logging
import copy
import heapq
import numpy as np
from numpy.random import Generator
from argparse import ArgumentParser, Namespace

from sympy import prime

import ssenvs.envs.instances as jsslib
from ssenvs.envs.poss import StochasticEnv
from ssenvs.envs.ss import PredictionEnv


logger = logging.getLogger(__name__)
logger.propagate = True

def process_command_line() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-N", "--trials", type=int, help="Number of trials", default=10)
    parser.add_argument("-i", "--instance", type=str, default="ta01", help="Instance name")

    return parser.parse_args()


def select_action(pending: list[int],
                  idle: set[int],
                  rng: np.random.Generator,
                  gamma: float = 0.5) -> list[tuple[int, int]]:
    """
    Samples the arm
    """
    action: list[tuple[int, int]] = []

    if len(pending) > 0 and len(idle) > 0:
        for j in pending:
            # allow for a job to be forgone
            if rng.random() < gamma:
                action.append((j, idle.pop()))
            else:
                continue
            if len(idle) == 0:
                break

    return action

def rollout(pred_env: PredictionEnv, a0: list[tuple[int, int]], rng: Generator) -> float:
    """
    Random policy rollout
    """
    G: float = 0
    obs, info, r, done, _ = pred_env.step(a0)
    G += r
    while not done:
        pending: list = [pj for pj in obs.get('state').pending]
        rng.shuffle(pending)
        idle = copy.copy(obs.get('state').idle)
        a = select_action(pending, idle, rng)
        obs, info, r, done, _ = pred_env.step(a)
        G += r


    return G

def eval_mc_rollout(instance: str, seed: int) -> float:
    """

    """

    env = StochasticEnv(spec=dict(instance_path=jsslib.get_path() / instance))

    G: int = 0
    obs, info = env.reset(seed=prime(seed))
    done: bool = False
    t: int = 0
    policy_rng = np.random.default_rng(seed)
    cseed: int = seed + 1

    while not done:
        # Sample super-arms
        pred_env = PredictionEnv(spec=dict(instance=jsslib.get_path() / instance))

        # Note that this information is the same for all rollouts
        pending: list = [pj for pj in obs.get('state').pending]
        policy_rng.shuffle(pending)
        idle = copy.copy(obs.get('state').idle)

        Q_t: dict[tuple, float] = {}

        for k in range(5):
            a_0 = select_action(pending, idle, policy_rng)

            #logging.info(f"First action in rollout, time {t}: {a_0}")
            # Start rollout
            pred_env.reset(seed=cseed,
                           options=dict(initial=obs.get('state'),
                                        elapsed=env.elapsed,
                                        deadlines=env.deadlines,
                                        current_time_step=env.current_time_step,
                                        t_max=env.t_max,))
            Q_for_a_0 = rollout(pred_env, a_0, policy_rng)
            a_0_as_tuple = tuple(a_0)
            if a_0_as_tuple in Q_t:
                Q_t[a_0_as_tuple] = max(Q_t[a_0_as_tuple], Q_for_a_0)
            else:
                Q_t[tuple(a_0)] = Q_for_a_0
            cseed += 1


        arms: list = []
        for a, Qa in Q_t.items():
            heapq.heappush(arms, (-Qa, a))
        #print(arms)
        best_Q, best_action = heapq.heappop(arms)
        #print(f"Best action in rollout, time {t}: {best_action}, value: {-best_Q}")

        # Change back from tuple of tuples to list of tuples
        obs, info, r, done, _ = env.step(list(best_action))
        G += r
        t += 1


    return G

def main(opt: Namespace) -> None:
    logging.basicConfig(level=logging.DEBUG,
                        filename='./logs/stochastic_scheduling/random_policy.log',
                        filemode='w')
    for i in range(1, opt.trials+1):
        logging.info(f"Starting trial #{i}")
        G_0: int = eval_mc_rollout(opt.instance, prime(i))
        print(f"Trial {i} cost: {G_0}")
        logging.info(f"Trial ended, mc policy value: {G_0}")

if __name__ == "__main__":
    main(process_command_line())