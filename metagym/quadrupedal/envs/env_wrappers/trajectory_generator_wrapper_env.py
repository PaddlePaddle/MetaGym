# Third party code
#
# The following code are copied or modified from:
# https://github.com/google-research/motion_imitation/blob/master/motion_imitation/envs/env_wrappers/trajectory_generator_wrapper_env.py


"""A wrapped MinitaurGymEnv with a built-in controller."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

class TrajectoryGeneratorWrapperEnv(object):
  """A wrapped LocomotionGymEnv with a built-in trajectory generator."""

  def __init__(self, gym_env, trajectory_generator):
    """Initialzes the wrapped env.
    Args:
      gym_env: An instance of LocomotionGymEnv.
      trajectory_generator: A trajectory_generator that can potentially modify
        the action and observation. Typticall generators includes the PMTG and
        openloop signals. Expected to have get_action and get_observation
        interfaces.
    Raises:
      ValueError if the controller does not implement get_action and
      get_observation.
    """
    self._gym_env = gym_env
    if not hasattr(trajectory_generator, 'get_action') or not hasattr(
        trajectory_generator, 'get_observation'):
      raise ValueError(
          'The controller does not have the necessary interface(s) implemented.'
      )

    self._trajectory_generator = trajectory_generator

    # The trajectory generator can subsume the action/observation space.
    if hasattr(trajectory_generator, 'observation_space'):
      self.observation_space = self._trajectory_generator.observation_space

    if hasattr(trajectory_generator, 'action_space'):
      # print('yes!')
      self.action_space = self._trajectory_generator.action_space
      # print('action_space:',self.action_space)

  def __getattr__(self, attr):
    return getattr(self._gym_env, attr)

  def _modify_observation(self, observation):
    return self._trajectory_generator.get_observation(observation)

  def reset(self, **kwargs):
    if getattr(self._trajectory_generator, 'reset'):
      self._trajectory_generator.reset()
    observation,info = self._gym_env.reset(**kwargs)
    if "info" in kwargs.keys() and kwargs["info"]:
      return self._modify_observation(observation)[0],info
    else:
      return self._modify_observation(observation)[0]

  def step(self, action):
    """Steps the wrapped environment.
    Args:
      action: Numpy array. The input action from an NN agent.
    Returns:
      The tuple containing the modified observation, the reward, the epsiode end
      indicator.
    Raises:
      ValueError if input action is None.
    """

    if action is None:
      raise ValueError('Action cannot be None')

    new_action = self._trajectory_generator.get_action(
        self._gym_env.robot.GetTimeSinceReset(), action)

    original_observation, reward, done, info = self._gym_env.step(new_action)

    info['real_action'] = new_action

    return self._modify_observation(original_observation)[0], reward, done, info