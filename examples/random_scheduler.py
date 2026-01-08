from pathlib import Path
import numpy as np

import gymnasium as gym
import ssenvs
import ssenvs.envs.instances as jsslib

def main() -> None:
    """
    Main function
    """

    env = gym.make('jss-v1',
                   env_config=dict(instance_path=jsslib.get_path() / "ta80",
                                   always_allow_noop=True))

    obs_0, info_0 = env.reset(seed=42)
    done: bool = False
    policy_cost_obs: float = 0

    while not done:
        # Get legal actions
        legal_actions = [a for a, applicable in enumerate(obs_0.get('action_mask')) if applicable]
        if len(legal_actions) == 0:
            raise RuntimeError('No op should always be a legal action')
        # Get a random legal action
        action = np.random.choice(legal_actions, 1)[0]

        obs_t, r_t, done_t, trunc_t, info_t = env.step(action)
        policy_cost_obs += r_t

if __name__ == '__main__':
    main()