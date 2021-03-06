import numpy as np
from gym.envs.mujoco import mujoco_env
from gym import utils
from gym import spaces
import os 
import os.path as osp
import signal 
import gym

import mujoco_py

from trac_ik_python.trac_ik_wrap import TRAC_IK
from trac_ik_python import trac_ik_wrap as tracik

import HER.envs
from time import sleep

# from ipdb import set_trace

class BaxterEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """cts env, 6dim
    state space: relative state space position of gripper and (target-gripper)
    random restarts for target on the table
    reward function: - 1(not reaching)
    actions: (delta_x, delta_y, delta_z) 5cm push
    starting state: (0.63, 0.2, 0.55, 0.3)
    """
    def __init__(self, max_len=10):
        dirname = os.path.dirname(os.path.abspath(__file__)) 
        mujoco_env.MujocoEnv.__init__(self, os.path.join(dirname, "mjc/baxter_orient_left_3dreacher_modified.xml") , 1)
        utils.EzPickle.__init__(self)

        ## mujoco things
        # task space action space

        low = np.array([-1., -1., -1])
        high = np.array([1., 1., 1])

        self.action_space = spaces.Box(low, high)

        self.tuck_pose = {
                            'left':  np.array([-0.08, -1.0, -1.19, 1.94,  0.67, 1.03, -0.50])
                       }

        self.start_pose = {
                            'left' : np.array([-0.21, -0.75, -1.4, 1.61, 0.60, 0.81, -0.52])
                            }
        

        ## starting pose
        self.init_qpos = self.data.qpos.copy().flatten()
        self.init_qpos[1:8] = np.array(self.start_pose["left"]).T
        
        
        ## ik setup
        urdf_filename = osp.join(dirname, "urdf", "baxter_modified.urdf")
                
        with open(urdf_filename) as f:
            urdf = f.read()
        
        # mode; Speed, Distance, Manipulation1, Manipulation2
        self.ik_solver = TRAC_IK("base",
                        "left_gripper",
                        urdf,
                        0.005,  # default seconds
                        1e-5,  # default epsilon
                        "Speed")

        self.old_state = np.zeros((6,))
        self.max_num_steps = max_len
        print("INIT DONE!")
      

    ## gym methods

    def reset_model(self):
        print("last state:",self.old_state)
        print("New Episode!")
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        ## random target location
        qpos[-3:-1] = qpos[-3:-1] + self.np_random.uniform(low=-0.15, high=0.15, size=2)
        qpos[-1] = qpos[-1] + self.np_random.uniform(low=-0.1, high=0.1, size=1)
        
        self.set_state(qpos, qvel)

        target_pos = np.array([0.6 , 0.3 , 0.2])
        target_quat = np.array([1.0, 0.0 , 0.0, 0])
        target = np.concatenate((target_pos, target_quat))
        action_jt_space = self.do_ik(ee_target= target, jt_pos = self.data.qpos[1:8].flat)
        if action_jt_space is not None:
            self.apply_action(action_jt_space)
        ## for calculating velocities
        # self.old_state = np.zeros((6,))
        self.contacted = False
        self.out_of_bound = 0
        self.num_step = 0
        ob = self._get_obs()
        gripper_pose = ob[:3]
        target_pose = ob[-3:]

        relative_ob = np.concatenate([gripper_pose, target_pose - gripper_pose])
        return relative_ob

    def viewer_setup(self):
        # cam_pos = np.array([0.1, 0.0, 0.7, 0.01, -45., 0.])
        cam_pos = np.array([1.0, 0.0, 0.7, 0.5, -45, 180])
        self.set_cam_position(self.viewer, cam_pos)

    def _get_obs(self):
        ee_x, ee_y, ee_z = self.data.site_xpos[0][:3]
        target_x, target_y, target_z = self.data.site_xpos[1][:3]

        state = np.array([ee_x, ee_y, ee_z, target_x, target_y, target_z])
        vel = (state - self.old_state)/self.dt

        self.old_state = state.copy()
        return state
        

    def set_state(self, qpos, qvel):
        assert qpos.shape == (self.model.nq,) and qvel.shape == (self.model.nv,)
        self.model.data.qpos = qpos
        self.model.data.qvel = qvel
        # self.model._compute_subtree() #pylint: disable=W0212
        self.model.forward()

    ## my methods
    def apply_action(self, action):  
        ctrl = self.data.ctrl.copy()
        # print(ctrl.shape)
        ctrl[:7,0] = np.array(action)
        self.data.ctrl = ctrl
        self.do_simulation(ctrl, 1000)
    
    def do_ik(self, ee_target, jt_pos):
        
        # print("starting to do ik")
        # print(ee_target[:3])
       
        # Populate seed with current angles if not provided
        init_pose_trac = tracik.DoubleVector()
        for i in range(7):
            init_pose_trac.push_back(jt_pos[i])

        x,y,z,qx,qy,qz,qw = ee_target
        qout = list(self.ik_solver.CartToJnt(init_pose_trac, x,y,z,qx,qy,qz,qw ))

        if(len(qout)>0):
            # print("ik sol:",qout)
            return qout
        else:
            print("!no result found")
            return None

    def set_cam_position(self, viewer, cam_pos):
        for i in range(3):
            viewer.cam.lookat[i] = cam_pos[i]
        viewer.cam.distance = cam_pos[3]
        viewer.cam.elevation = cam_pos[4]
        viewer.cam.azimuth = cam_pos[5]
        viewer.cam.trackbodyid = -1



    def _step(self, action):


        ## hack for the init of mujoco.env
        if(action.shape[0]>3):
            return np.zeros((6,1)), 0, False, {}
        
        self.num_step += 1
        old_action_jt_space = self.data.qpos[1:8].T.copy()

        ## parsing of primitive actions
        delta_x, delta_y, delta_z = action
        # print("delta x:%.4f, y:%.4f"%(delta_x, delta_y))
        x, y,z = self.old_state[:3].copy()
        # print("old x:%.4f, y:%.4f, z:%.4f"%(x,y,z))
        # print("delta x:%.4f, y:%.4f, z:%.4f"%(delta_x, delta_y,delta_z))
        x += delta_x*0.05
        y += delta_y*0.05
        z += delta_z*0.05
        # print("x:%.4f, y:%.4f, z:%.4f"%(x,y,z))
        
        ## after checking out of bound, we should push the actions close to the boundary. 
        ## this will correct if the old action are of bound and some motion to reach a new state
        out_of_bound = (x<0.4 or x>0.8) or (y<0.0 or y>0.6) or (z<0.0 or z>0.5)


        if np.abs(delta_x*0.05)>0.0001 or np.abs(delta_y*0.05)>0.0001 or np.abs(delta_z*0.05)>0.0001:
            target_pos = np.array([x , y , z])
            target_quat = np.array([1.0, 0.0 , 0.0, 0])
            target = np.concatenate((target_pos, target_quat))
            action_jt_space = self.do_ik(ee_target= target, jt_pos = self.data.qpos[1:8].flat)
            if (action_jt_space is not None) and (not out_of_bound):
                # print("ik:", action_jt_space)
                self.apply_action(action_jt_space)
            else:
                action_jt_space = old_action_jt_space.copy()

        else:
            action_jt_space = old_action_jt_space.copy()

        # print("controller:",self.data.qpos[1:8].T)
        ## getting state
        ob = self._get_obs()
        gripper_pose = ob[:3]
        target_pose = ob[3:6]
        
        ## reward function definition
        reward_reaching_goal = np.linalg.norm(gripper_pose - target_pose) < 0.05           
        total_reward = -1*(not reward_reaching_goal)

        info = {}

        if reward_reaching_goal == 1:
            done = True
            info["done"] = "goal reached"
        elif (self.num_step > self.max_num_steps):
            done = True
            info["done"] = "max_steps_reached"
        else: 
            done = False

        info['absolute_ob'] = ob.copy()
        
        relative_ob = np.concatenate([gripper_pose, target_pose - gripper_pose ])
        return relative_ob, total_reward, done, info
                                        
    def apply_hindsight(self, states, actions, goal_state):
        '''generates hindsight rollout based on the goal
        '''
        goal = states[-1][:3]    ## this is the absolute goal location
        her_states, her_rewards = [], []
        for i in range(len(actions)):
            state = states[i]
            state[-3:] = goal.copy() - state[:3]
            reward = self.calc_reward(state, goal, actions[i])
            her_states.append(state)
            her_rewards.append(reward)

        goal_state[-3:] = np.array([0., 0., 0.])
        her_states.append(goal_state)

        return her_states, her_rewards
    
    def calc_reward(self, state, goal, action):
        
        gripper_pose = state[:3]
        target_pose = state[-3:] + gripper_pose
        
        ## reward function definition
        reward_reaching_goal = np.linalg.norm(gripper_pose- target_pose) < 0.03             #assume: my robot has 2cm error
        total_reward = -1*(not reward_reaching_goal)
        return total_reward



if __name__ == "__main__":
    
    from ipdb import set_trace
    np.set_printoptions(precision=4)
    env = BaxterEnv(max_len=50)
    # env = gym.make("Baxter3dReacher-v1")
    EVAL_EPISODE = 10
    reward_mat = []

    try:

        for l in range(EVAL_EPISODE):
            print("Evaluating:%d"%(l+1))
            done = False
            i =0
            random_r = 0
            ob = env.reset()
            print(ob)

            for _ in range(10):
                env.render()
            while((not done) and (i<1000)):
                
                ee_x, ee_y, ee_z = env.data.site_xpos[0][:3]
                box_x, box_y, box_z = env.data.site_xpos[1][:3]
                action = np.array([(box_x - ee_x), (box_y - ee_y), (box_z - ee_z)])
                action /= np.linalg.norm(action)
                # action = env.action_space.sample()
                ob, reward, done, info = env.step(action)
                # print(i, action, ob, reward)
                # print(i, ob, reward, info)
                print( i, reward,ob, action)    
                # set_trace()
                i+=1
                sleep(.01)
                env.render()
                random_r += reward

            print("num steps:%d, total_reward:%.4f"%(i+1, random_r))

            for _ in range(100):
                env.render()
            reward_mat += [random_r]
        print("reward - mean:%f, var:%f"%(np.mean(reward_mat), np.var(reward_mat)))
    except KeyboardInterrupt:
        print("Exiting!")
