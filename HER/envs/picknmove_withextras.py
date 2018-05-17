from HER.envs import grasping_withgap
import numpy as np 
from gym.envs.robotics.utils import mocap_set_action

class BaxterEnv(grasping_withgap.BaxterEnv):
    
    def __init__(self, max_len=50, test = False):
        super(BaxterEnv, self).__init__(max_len=max_len, test=test)

    
    def reset_model(self):
        # randomize gripper loc

        # gripper_pos = np.array([0.6 , 0.3 , 0.15])
        gripper_pos = self.np_random.uniform(self.target_range_min[:self.space_dim] + 0.05, self.target_range_max[:self.space_dim] - 0.05, size=self.space_dim)        
        self.apply_action(pos=gripper_pos)


        # 0: random, 1: grasped
        sample = self.np_random.choice(2)
        # print("sample", sample)
        if sample == 1 and (not self.test):
            
            # define one pose in hand 
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)
            object_qpos[:self.space_dim] = gripper_pos[:self.space_dim] - np.array([0., 0., 0.1])
            add_noise = False
            self.data.ctrl[:] = np.array([0.,0.])
        else:
            
            # spawn near gripper
            object_qpos = self.sim.data.get_joint_qpos('box') 
            assert object_qpos.shape == (7,)

            dim = 2
            object_qpos[:dim] = self.np_random.uniform(self.target_range_min[:dim] + 0.05, self.target_range_max[:dim] - 0.05 , size=dim) 
            object_qpos[dim] = 0.1

            
            while(np.linalg.norm(gripper_pos[:dim] - object_qpos[:dim]) < 0.05):
                object_qpos[:dim] = self.np_random.uniform(self.target_range_min[:dim] + 0.05, self.target_range_max[:dim] - 0.05, size=dim)


        # spawning obj on ground
        if sample == 0: object_qpos[2] = 0.1
        object_qpos[3:] = np.array([1., 0.,0.,0.])
        self.sim.data.set_joint_qpos('box', object_qpos)  

        # print("obj2grip", np.linalg.norm(gripper_pos[:2] - object_qpos[:2]))
        if sample == 1 and (not self.test):
            # the object drops a bit as the gripper is still adjusting its gap
            self.sim.step()
            object_qpos = self.data.get_joint_qpos('box')

        target_qpos = self.data.get_joint_qpos('target')
        target_qpos[:2] = self.np_random.uniform(self.target_range_min[:2] + 0.05, self.target_range_max[:2] - 0.05, size=2) 
        if(self.space_dim==3):
            target_qpos[2] = self.np_random.rand()*0.1
        while(np.linalg.norm(target_qpos[:self.space_dim] - object_qpos[:self.space_dim]) < 0.05):
            target_qpos[:2] = self.np_random.uniform(self.target_range_min[:2] + 0.05, self.target_range_max[:2] - 0.05, size=2) 
            if(self.space_dim==3):
                target_qpos[2] = self.np_random.rand()*0.1
        
        object_qpos = self.sim.data.get_joint_qpos('box') 
        
        self.data.set_joint_qpos('target', target_qpos)
        # print("target pos", target_qpos[:3])
        object_qpos = self.sim.data.get_joint_qpos('box') 
        self.sim.forward()
        self.num_step = 1
        return self._get_obs()

    def apply_hierarchical_hindsight(self, states, actions, goal_state, sub_states_list):
        '''generates hierarchical hindsight rollout based on the goal
        '''
        goal = goal_state[self.space_dim:2*self.space_dim]    ## this is the absolute goal location = obj last loc
        # enter the last state in the list
        states.append(goal_state)
        num_tuples = len(actions)

        her_states, her_rewards = [], []
        # make corrections in the first state 
        states[0][-self.space_dim:] = goal.copy()
        her_states.append(states[0])
        for i in range(1, num_tuples + 1):
            state = states[i]
            state[-self.space_dim:] = goal.copy()    # copy the new goal into state
            # use all the sub states to calculate total reward
            reward = 0.
            for sub_state in sub_states_list[i-1]:
                reward += self.calc_reward(sub_state)
            her_states.append(state)
            her_rewards.append(reward)

        return her_states, her_rewards

