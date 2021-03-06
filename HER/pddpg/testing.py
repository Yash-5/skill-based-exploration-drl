import os
import time
from collections import deque
import pickle
import math

from HER.ddpg.ddpg import DDPG
from HER.ddpg.util import mpi_mean, mpi_std, mpi_max, mpi_sum
import HER.common.tf_util as U
from HER.ddpg.util import read_checkpoint_local

from HER import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

import os.path as osp
from time import sleep

from ipdb import set_trace

def test(env, render_eval, reward_scale, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_eval_steps, batch_size, memory,
    tau=0.01, eval_env=None, param_noise_adaption_interval=50, **kwargs):
    
    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    
    if kwargs['skillset']:
        action_shape = (kwargs['my_skill_set'].len + kwargs['my_skill_set'].params,)
    else:
        action_shape = env.action_space.shape

    agent = DDPG(actor, critic, memory, env.observation_space.shape, action_shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale)
    
    
    var_list_restore = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="actor") + \
                            tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="obs_rms")

    print(var_list_restore)
    saver = tf.train.Saver(var_list = var_list_restore)

    with U.single_threaded_session() as sess:
        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        ## restore
        if kwargs['skillset']:
            ## restore skills
            my_skill_set = kwargs['my_skill_set']
            my_skill_set.restore_skillset(sess=sess)

        ## restore meta controller weights
        restore_dir = osp.join(kwargs["restore_dir"], "model")
        if (restore_dir is not None):
            print('Restore path : ',restore_dir)
            # checkpoint = tf.train.get_checkpoint_state(restore_dir)
            # if checkpoint and checkpoint.model_checkpoint_path:
            model_checkpoint_path = read_checkpoint_local(restore_dir)
            if model_checkpoint_path:
                
                saver.restore(U.get_session(), model_checkpoint_path)
                print( "checkpoint loaded:" , model_checkpoint_path)
                tokens = model_checkpoint_path.split("-")[-1]
                # set global step
                global_t = int(tokens)
                print( ">>> global step set:", global_t)
            else:
                print(">>>no checkpoint file found")
        
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
                
        # Evaluate.
        eval_episode_rewards = []
        eval_episode_rewards_history = []
        eval_episode_success = []
        for i in range(100):
            print("Evaluating:%d"%(i+1))
            eval_episode_reward = 0.
            eval_obs = eval_env.reset()
            print("start obs", eval_obs[:6], eval_obs[-3:])

            # for _ in range(1000):
            #     eval_env.render()
            #     print(eval_env.sim.data.get_site_xpos('box'))
            #     sleep(0.1)

            eval_done = False
            
            while(not eval_done):
                eval_paction, eval_pq = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                if(kwargs['skillset']):
                    ## break actions into primitives and their params    
                    eval_primitives_prob = eval_paction[:my_skill_set.len]
                    eval_primitive_id = np.argmax(eval_primitives_prob)

                    print("skill chosen%d"%eval_primitive_id)
                    eval_r = 0.
                    eval_skill_obs = eval_obs.copy()
                    for _ in range(kwargs['commit_for']):
                        eval_action = my_skill_set.pi(primitive_id=eval_primitive_id, obs = eval_skill_obs.copy(), primitive_params=eval_paction[kwargs['my_skill_set'].len:])
                        eval_skill_new_obs, eval_skill_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    
                        eval_skill_obs = eval_skill_new_obs
                        eval_r += eval_skill_r
                        if render_eval:
                            eval_env.render()
                            sleep(0.1)

                        if eval_done or my_skill_set.termination(eval_skill_new_obs, eval_primitive_id, primitive_params = eval_paction[my_skill_set.len:]):
                            break

                    eval_new_obs = eval_skill_new_obs

                else:
                    eval_action, q = eval_paction, eval_pq
                    eval_new_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])

                    
                eval_episode_reward += eval_r
                eval_obs = eval_new_obs

            print("ended",eval_info["done"])
                
            print("episode reward::%f"%eval_episode_reward)
            
            eval_episode_rewards.append(eval_episode_reward)
            eval_episode_rewards_history.append(eval_episode_reward)
            eval_episode_success.append(eval_info["done"]=="goal reached")
            eval_episode_reward = 0.
            
        print("episode reward - mean:%.4f, var:%.4f, success:%.4f"%(np.mean(eval_episode_rewards), np.var(eval_episode_rewards), np.mean(eval_episode_success)))

            


