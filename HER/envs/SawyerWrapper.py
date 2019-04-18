import multiworld.envs.mujoco
from multiworld.envs.mujoco.sawyer_xyz.sawyer_reach import SawyerReachXYEnv
from multiworld.envs.mujoco.sawyer_xyz.sawyer_push_and_reach_env import SawyerPushAndReachXYEnv
from gym.spaces import Box
import numpy as np

#  TODO: seed() is being called. It calls just the base gym's seed, not sure if any other rngs need to be seeded
#  TODO: arpit's code checks for a get_state method to save the env. Not implementing this as it might mess up the Serializable things
class WrapSawyerReachXYEnv(SawyerReachXYEnv):
    def __init__(self, *args, **kwargs):

        # defined in serializable
        self.quick_init(locals())

        # Sawyer env never seems to terminate and just returns false for done at each step
        # Adding step limit so that episodes end
        if 'nb_step_limit' in kwargs:
            self.nb_step_limit = kwargs['nb_step_limit']
            del kwargs['nb_step_limit']
        else:
            self.nb_step_limit = 1000
        self.nb_steps = 0
        self.episode_done = False

        SawyerReachXYEnv.__init__(self, *args, **kwargs)

        # self.observation_space is of type gym.spaces.dict.Dict
        # But this repo needs it to be of type gym.spaces.box.Box
        self.original_observation_space = self.observation_space

        obs_space = self.observation_space.spaces['observation']
        goal_space = self.observation_space.spaces['desired_goal']
        agoal_space = self.observation_space.spaces['achieved_goal']

        self.space_dim = obs_space.shape[0]

        self.observation_space = Box(np.concatenate([obs_space.low, goal_space.low, agoal_space.low]),
                                     np.concatenate([obs_space.high, goal_space.high, agoal_space.high]),
                                     dtype=np.float32)

    def reset(self):
        # saved current observation dict returned by the parent
        self.curr_obs_dict = super().reset()
        self.nb_steps = 0
        self.episode_done = False
        return np.concatenate([self.curr_obs_dict['observation'], self.curr_obs_dict['achieved_goal'], self.curr_obs_dict['desired_goal']])

    def step(self, action):
        assert not self.episode_done, "Episode is done, call reset"
        self.curr_obs_dict, reward, done, info = super().step(action)
        next_obs = np.concatenate([self.curr_obs_dict['observation'], self.curr_obs_dict['achieved_goal'], self.curr_obs_dict['desired_goal']])
        self.nb_steps += 1
        if self.nb_steps == self.nb_step_limit or info['hand_success']:
            self.episode_done = True

        if info['hand_success']:
           info['done'] = "goal reached"
        else:
           info['done'] = "goal not reached"
        return next_obs, reward, self.episode_done, info

    def calc_reward(self, state):
        """
        This is based on SawyerReachXYEnv.compute_rewards
        It takes in action as the first argument but does not use it(as demonstrated by the code not crashing on passing None)
        """
        obs = {}
        obs['state_achieved_goal'] = np.expand_dims(state[:self.space_dim], axis=0)
        obs['state_desired_goal'] = np.expand_dims(state[-self.space_dim:], axis=0)
        r = super().compute_rewards(None, obs).item()
        return r

    def apply_hindsight(self, states, actions, goal_state):
        goal = goal_state[self.space_dim:2*self.space_dim]

        states.append(goal_state)
        num_tuples = len(actions)

        her_states, her_rewards = [], []

        states[0][-self.space_dim:] = goal.copy()
        her_states.append(states[0])
        for i in range(1, num_tuples + 1):
            state = states[i]
            state[-self.space_dim:] = goal.copy()    # copy the new goal into state
            reward = self.calc_reward(state)
            her_states.append(state)
            her_rewards.append(reward)

        return her_states, her_rewards


class WrapSawyerPushAndReachXYEnv(SawyerPushAndReachXYEnv):
    def __init__(self, *args, **kwargs):
        # defined in serializable
        self.quick_init(locals())
        if 'nb_step_limit' in kwargs:
            self.nb_step_limit = kwargs['nb_step_limit']
            del kwargs['nb_step_limit']
        else:
            self.nb_step_limit = 1000
        self.nb_steps = 0
        self.episode_done = False

        SawyerPushAndReachXYEnv.__init__(self, *args, **kwargs)

        # self.observation_space is of type gym.spaces.dict.Dict
        # But this repo needs it to be of type gym.spaces.box.Box
        self.original_observation_space = self.observation_space

        obs_space = self.observation_space.spaces['observation']
        goal_space = self.observation_space.spaces['desired_goal']
        agoal_space = self.observation_space.spaces['achieved_goal']

        self.space_dim = obs_space.shape[0]

        self.observation_space = Box(np.concatenate([obs_space.low, goal_space.low, agoal_space.low]),
                                     np.concatenate([obs_space.high, goal_space.high, agoal_space.high]),
                                     dtype=np.float32)

    def reset(self):
        # saved current observation dict returned by the parent
        self.curr_obs_dict = super().reset()
        self.nb_steps = 0
        self.episode_done = False
        return np.concatenate([self.curr_obs_dict['observation'], self.curr_obs_dict['achieved_goal'], self.curr_obs_dict['desired_goal']])

    def step(self, action):
        assert not self.episode_done, "Episode is done, call reset"
        self.curr_obs_dict, reward, done, info = super().step(action)
        next_obs = np.concatenate([self.curr_obs_dict['observation'], self.curr_obs_dict['achieved_goal'], self.curr_obs_dict['desired_goal']])
        self.nb_steps += 1
        if self.nb_steps == self.nb_step_limit or info['hand_success']:
            self.episode_done = True

        if info['puck_success']:
           info['done'] = "goal reached"
        else:
           info['done'] = "goal not reached"
        return next_obs, reward, self.episode_done, info

    def calc_reward(self, state):
        """
        This is based on SawyerPushAndReachXYZEnv.compute_rewards
        It takes in action as the first argument but does not use it(as demonstrated by the code not crashing on passing None)
        """
        obs = {}
        obs['state_achieved_goal'] = np.expand_dims(state[:self.space_dim], axis=0)
        obs['state_desired_goal'] = np.expand_dims(state[-self.space_dim:], axis=0)
        r = super().compute_rewards(None, obs).item()
        return r

    def apply_hindsight(self, states, actions, goal_state):
        goal = goal_state[self.space_dim:2*self.space_dim]

        states.append(goal_state)
        num_tuples = len(actions)

        her_states, her_rewards = [], []

        states[0][-self.space_dim:] = goal.copy()
        her_states.append(states[0])
        for i in range(1, num_tuples + 1):
            state = states[i]
            state[-self.space_dim:] = goal.copy()    # copy the new goal into state
            reward = self.calc_reward(state)
            her_states.append(state)
            her_rewards.append(reward)

        return her_states, her_rewards
