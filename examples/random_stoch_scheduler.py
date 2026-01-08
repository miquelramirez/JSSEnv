import logging
import copy
from argparse import ArgumentParser, Namespace

from sympy import prime

import ssenvs.envs.instances as jsslib
from ssenvs.envs.poss import StochasticEnv


logger = logging.getLogger(__name__)
logger.propagate = True

def eval_random_policy(instance: str, n: int) -> int:
    logging.info(f"Starting trial #{n}")
    env = StochasticEnv(spec=dict(instance_path=jsslib.get_path() / instance))

    G: int = 0
    obs, info = env.reset(seed=n)
    done: bool = False
    t: int = 0

    while not done:
        idle = copy.copy(obs.get('state').idle)
        pending = copy.copy(obs.get('state').pending)
        action: list[tuple[int, int]] = []
        if len(pending) > 0 and len(idle) > 0:
            for j in pending:
                action.append((j, idle.pop()))
                if len(idle) == 0:
                    break
        logging.info(f"Action at time {t}: {action}")
        obs, info, r, done, _ = env.step(action)
        G += r


    logging.info(f"Trial ended, random policy value: {G}")
    return G

def process_command_line() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("-N", "--trials", type=int, help="Number of trials", default=10)
    parser.add_argument("-i", "--instance", type=str, default="ta01", help="Instance name")

    return parser.parse_args()

def main(opt: Namespace) -> None:
    logging.basicConfig(level=logging.DEBUG, filename='./logs/stochastic_scheduling/random_policy.log')
    for i in range(opt.trials):
        G_0: int = eval_random_policy(opt.instance, prime(i+1))
        print(f"Trial {i+1} cost: {G_0}")

if __name__ == "__main__":
    main(process_command_line())