import ssenvs.envs.instances as jsslib
from ssenvs.envs.poss import StochasticEnv
import logging

logger = logging.getLogger(__name__)
logger.propagate = True

def test_sim_ta01(caplog) -> None:
    #logging.basicConfig(level=logging.DEBUG, filename='./logs/test_sim_ta01.log')
    caplog.set_level(logging.DEBUG)
    logger.info("Started simulation")

    env = StochasticEnv(spec=dict(instance_path=jsslib.get_path() / "ta01"))

    obs0, info0 = env.reset(seed=42)

    logger.info(f"Pending jobs at t=0: {obs0.get('state').pending}")
    assert len(obs0.get('state').pending)==0
    print(env.releases)
    print(env.t_max)

    for _ in range(10):
        obs_t, info_t, r_t, done_t, trunc_t = env.step([])
        print(obs_t.get('state'))

    logger.info("Finished simulation")