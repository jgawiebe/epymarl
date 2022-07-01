import numpy as np
from gym import spaces, Env

import sys
sys.path.append("..")

from CybORG.CybORG import CybORG
from CybORG.Agents.Wrappers import BaseWrapper, OpenAIGymWrapper, BlueTableWrapper,RedTableWrapper,EnumActionWrapper

from gym import Env, spaces

class FlattenObservation():
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self._env.observation_space, observation)
            ]
        )

class PyMARLWrapper(Env, BaseWrapper):

    def __init__(self, agent_name: str, env, agent=None,
            reward_threshold=None, max_steps = None, n_agents=1):
        super().__init__(env, agent)
        self.agent_name = agent_name
        if agent_name.lower() == 'red':
            table_wrapper = RedTableWrapper
        elif agent_name.lower() == 'blue':
            table_wrapper = BlueTableWrapper
        else:
            raise ValueError('Invalid Agent Name')

        env = table_wrapper(env, output_mode='vector')
        env = EnumActionWrapper(env)
        env = OpenAIGymWrapper(agent_name=agent_name, env=env)

        self._env = env
        self._obs = None

        #self.action_space = self._env.action_space
        #self.observation_space = self._env.observation_space
        self.reward_threshold = reward_threshold
        self.max_steps = max_steps
        self.step_counter = None

        #pyMARL compatibility
        self.n_agents = n_agents
        self.observation_space = spaces.Tuple(
            tuple(self.n_agents * [self._env.observation_space])
        )
        self.action_space = spaces.Tuple(tuple(self.n_agents * [self._env.action_space]))

        # necessary?
        self.longest_action_space = max(self.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self.observation_space, key=lambda x: x.shape
        )

    # step takes array of actions passes into a single agent step function and
    # converts output into iterables for parsing by gymma
    def step(self,actions):
        # single agent action
        action = actions[0]
        obs, reward, done, info = self._env.step(action=action)
        # single agent obs
        self._obs = tuple([obs])
        rewards = tuple([reward])

        self.step_counter += 1
        if self.max_steps is not None and self.step_counter >= self.max_steps:
            done = True

        dones = tuple([done])
        return self._obs, rewards, dones, {}

    def seed(self, seed: int):
        self._env.set_seed(seed)

    # gyma methods (TEMPORARY FIX)
    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

#    def get_obs_size(self):
#        """ Returns the shape of the observation """
#        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

#    def get_state_size(self):
#        """ Returns the shape of the state"""
#        return self.n_agents * flatdim(self.longest_observation_space)

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

#    def get_obs_size(self):
#        """ Returns the shape of the observation """
#        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

#    def get_state_size(self):
#        """ Returns the shape of the state"""
#        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

#    def get_avail_agent_actions(self, agent_id):
#        """ Returns the available actions for agent_id """
#        valid = flatdim(self._env.action_space[agent_id]) * [1]
#        invalid = [0] * (self.longest_action_space.n - len(valid))
#        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self.step_counter = 0
        self._obs = self._env.reset()

        return tuple([self._obs])

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

#    def seed(self):
#       return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}

#    def reset(self):
#        self.step_counter = 0
#        return self._env.reset()

    def get_attr(self,attribute:str):
        return self._env.get_attr(attribute)

    def get_observation(self, agent: str):
        return self._env.get_observation(agent)

    def get_agent_state(self,agent:str):
        return self._env.get_agent_state(agent)

    def get_action_space(self, agent=None) -> dict:
        return self._env.get_action_space(self.agent_name)

    def get_last_action(self,agent):
        return self.get_attr('get_last_action')(agent)

    def get_ip_map(self):
        return self.get_attr('get_ip_map')()

    def get_rewards(self):
        return self.get_attr('get_rewards')()