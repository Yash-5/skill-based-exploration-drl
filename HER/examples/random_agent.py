import HER.envs
import sys
import gym
from time import sleep
import numpy as np 

from ipdb import set_trace
    
if __name__ == "__main__":
    
    print("Loading %s"%sys.argv[1])
    np.set_printoptions(precision=3)
    env = gym.make(sys.argv[1])
    EVAL_EPISODE = 100
    reward_mat = []

    try:

        ob = env.reset()
        print(ob)
        # set_trace()
        # while True:
        #     env.render(mode = 'human')

        # print("Crossed over")
        
        action3 = [0., 0., 0.1, -1]
        action1 = [0, 1., 0., -1]
        action2 = [1.,0.,0., 1]
        for l in range(EVAL_EPISODE):
            print("Evaluating:%d"%(l+1))
            done = False
            i =0
            random_r = 0
            ob = env.reset()
            print(ob[:6], ob[-3:])

            # print("l joint", env.data.get_joint_qpos("l_gripper_l_finger_joint"))
            # print("r joint", env.data.get_joint_qpos("l_gripper_r_finger_joint"))
            # print("obj initial pos", env.data.get_joint_qpos("box"))

            # for _ in range(50):
            #     sleep(0.1)
            #     env.render(mode='human')
            # set_trace()
            # grip_pos = env.env.sim.data.get_site_xpos('robot0:grip')
            # grip_velp = env.env.sim.data.get_site_xvelp('robot0:grip')
            # print(grip_pos, grip_velp)
            # exit(0)
            while((not done) and (i<1000)):
                

                # print("l joint", env.data.get_joint_qpos("l_gripper_l_finger_joint"))
                # print("r joint", env.data.get_joint_qpos("l_gripper_r_finger_joint"))
            
                # ee_x, ee_y, obj_x, obj_y, t_x, t_y = ob
                
                # action = np.array([(obj_x - ee_x)/0.05, (obj_y -ee_y)/0.05])
                # action = [0.,0., 0.1,-1.]
                action = env.action_space.sample()
                # action = [0.,0.,1.]
                # for checking grasping
                # action = [0., 0., 0.0, -1]
                # action = [0]*7
                # action[0] = 0.1
                # if(i<5):
                #     action = action3
                # elif(i>=5 and i<10):
                #     action = action1
                # else:
                #     action = action2
                
                sleep(0.1)
                ob, reward, done, info = env.step(action)
                # print(ob[9]-ob[10])
                # print(i, action, ob, reward)
                # print(i, ob, reward, info)
                print( i, ob,action) 
                # set_trace()
                # print("target:x:%.4f,y:%.4f,z:%.4f"%(ob[-3], ob[-2], ob[-1]))
                # print("gripper", env.data.get_site_xpos("grip_r"))   
                i+=1
                sleep(0.01)
                env.render(mode='human')
                random_r += reward


                # set_trace()

            print("num steps:%d, total_reward:%.4f"%(i, random_r))
            
            # k = 0
            # while k<10:
            #     k += 1
            #     sleep(0.1)
            #     env.render(mode='human')
            reward_mat += [random_r]
        print("reward - mean:%f, var:%f"%(np.mean(reward_mat), np.var(reward_mat)))
    except KeyboardInterrupt:
        action = [0.,0.]
        ob, reward, done, info = env.step(action)
        env.render(mode='human')
        sleep(0.5)

        print("Exiting!")