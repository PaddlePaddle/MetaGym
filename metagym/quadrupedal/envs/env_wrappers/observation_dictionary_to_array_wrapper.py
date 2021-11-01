# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation
"""An env wrapper that flattens the observation dictionary to an array."""
import gym
from metagym.quadrupedal.envs.utilities import env_utils


class ObservationDictionaryToArrayWrapper(gym.Env):
  """An env wrapper that flattens the observation dictionary to an array."""
  def __init__(self, gym_env, observation_excluded=()):
    """Initializes the wrapper."""
    self.observation_excluded = observation_excluded
    self._gym_env = gym_env
    self.observation_space = self._flatten_observation_spaces(
        self._gym_env.observation_space)
    self.action_space = self._gym_env.action_space

  def __getattr__(self, attr):
    return getattr(self._gym_env, attr)

  def _flatten_observation_spaces(self, observation_spaces):
    flat_observation_space = env_utils.flatten_observation_spaces(
        observation_spaces=observation_spaces,
        observation_excluded=self.observation_excluded)
    return flat_observation_space

  def _flatten_observation(self, input_observation):
    """Flatten the dictionary to an array."""
    return env_utils.flatten_observations(
        observation_dict=input_observation,
        observation_excluded=self.observation_excluded)

  def reset(self,**kwargs):
    observation,info = self._gym_env.reset(**kwargs)
    return self._flatten_observation(observation),info

  def step(self, action):
    """Steps the wrapped environment.
    Args:
      action: Numpy array. The input action from an NN agent.
    Returns:
      The tuple containing the flattened observation, the reward, the epsiode
        end indicator.
    """
    observation_dict, reward, done, _ = self._gym_env.step(action)
    return self._flatten_observation(observation_dict), reward, done, _

  def render(self, mode='human'):
    return self._gym_env.render(mode)