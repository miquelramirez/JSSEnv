import JSSEnv.envs.instances as jsslib
from JSSEnv.envs.sjss_env import StochasticEnv
import logging

logger = logging.getLogger(__name__)

def test_sim_ta01() -> None:
    logging.basicConfig(level=logging.DEBUG, filename='logs/test_sim_ta01.log')
    logger.info("Started simulation")

    env = StochasticEnv(spec=dict(instance_path=jsslib.get_path() / "ta01"))

    logger.info("Finished simulation")