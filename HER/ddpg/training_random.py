import os
import time
from collections import deque
import pickle
import math 
from time import sleep

from HER.ddpg.ddpg import DDPG
from HER.ddpg.util import normal_mean, normal_std, mpi_max, mpi_sum
import HER.common.tf_util as U
from HER.ddpg.util import read_checkpoint_local
from HER.planning.MC_planning_random import Planning_with_memories
from HER.common.schedules import LinearSchedule

from HER import logger
import numpy as np
import tensorflow as tf
from mpi4py import MPI

import os.path as osp
from ipdb import set_trace

def train(env, nb_epochs, nb_epoch_cycles, render_eval, reward_scale, render, param_noise, actor, critic,
    normalize_returns, normalize_observations, critic_l2_reg, actor_lr, critic_lr, action_noise,
    popart, gamma, clip_norm, nb_train_steps, nb_rollout_steps, batch_size, memory,
    tau=0.05, eval_env=None, param_noise_adaption_interval=50, nb_eval_episodes=20, **kwargs):
    rank = MPI.COMM_WORLD.Get_rank()

    np.set_printoptions(precision=4)

    assert (np.abs(env.action_space.low) == env.action_space.high).all()  # we assume symmetric actions.
    max_action = env.action_space.high
    
    if "dologging" in kwargs: 
        dologging = kwargs["dologging"]
    else:
        dologging = True

    if "tf_sum_logging" in kwargs: 
        tf_sum_logging = kwargs["tf_sum_logging"]
    else:
        tf_sum_logging = False
        
    if "invert_grad" in kwargs: 
        invert_grad = kwargs["invert_grad"]
    else:
        invert_grad = False

    if "actor_reg" in kwargs:
        actor_reg = kwargs["actor_reg"]
    else:
        actor_reg = False

    if dologging: logger.info('scaling actions by {} before executing in env'.format(max_action))

    if kwargs['look_ahead']:
        look_ahead = True
        look_ahead_planner = Planning_with_memories(skillset=kwargs['my_skill_set'], 
                                                        env=env, 
                                                        num_samples= 1)
        exploration = LinearSchedule(schedule_timesteps=int(nb_epochs * nb_epoch_cycles),
                                     initial_p=1.0,
                                     final_p=kwargs['exploration_final_eps'])
    else:
        look_ahead = False

    agent = DDPG(actor, critic, memory, env.observation_space.shape, env.action_space.shape,
        gamma=gamma, tau=tau, normalize_returns=normalize_returns, normalize_observations=normalize_observations,
        batch_size=batch_size, action_noise=action_noise, param_noise=param_noise, critic_l2_reg=critic_l2_reg,
        actor_lr=actor_lr, critic_lr=critic_lr, enable_popart=popart, clip_norm=clip_norm,
        reward_scale=reward_scale,
        inverting_grad = invert_grad,
        actor_reg = actor_reg
        )
    if dologging: logger.debug('Using agent with the following configuration:')
    if dologging: logger.debug(str(agent.__dict__.items()))

    # Set up logging stuff only for a single worker.
    if rank != -1:
        saver = tf.train.Saver(keep_checkpoint_every_n_hours=2, max_to_keep=5, save_relative_paths=True)
        save_freq = kwargs["save_freq"]
    else:
        saver = None
    

    # step = 0
    global_t = 0
    episode = 0
    eval_episode_rewards_history = deque(maxlen=100)
    episode_rewards_history = deque(maxlen=100)

    
   
    with U.single_threaded_session() as sess:
        
        # Set summary saver
        if dologging and tf_sum_logging and rank==0: 
            tf.summary.histogram("actor_grads", agent.actor_grads)
            tf.summary.histogram("critic_grads", agent.critic_grads)
            actor_trainable_vars = actor.trainable_vars
            for var in actor_trainable_vars:
                tf.summary.histogram(var.name, var)
            critic_trainable_vars = critic.trainable_vars
            for var in critic_trainable_vars:
                tf.summary.histogram(var.name, var)

            tf.summary.histogram("actions_out", agent.actor_tf)
            tf.summary.histogram("critic_out", agent.critic_tf)
            tf.summary.histogram("target_Q", agent.target_Q)

            summary_var = tf.summary.merge_all()
            writer_t = tf.summary.FileWriter(osp.join(logger.get_dir(), 'train'), sess.graph)
        else:
            summary_var = tf.no_op()

        # Prepare everything.
        agent.initialize(sess)
        sess.graph.finalize()

        ## restore
        if kwargs['skillset']:
            ## restore skills
            my_skill_set = kwargs['my_skill_set']
            my_skill_set.restore_skillset(sess=sess)

        if kwargs["restore_dir"] is not None:
            restore_dir = osp.join(kwargs["restore_dir"], "model")
            if (restore_dir is not None):
                print('Restore path : ',restore_dir)
                # checkpoint = tf.train.get_checkpoint_state(restore_dir)
                # if checkpoint and checkpoint.model_checkpoint_path:
                model_checkpoint_path = read_checkpoint_local(restore_dir)
                if model_checkpoint_path:
                    saver.restore(U.get_session(), model_checkpoint_path)
                    print( "checkpoint loaded:" , model_checkpoint_path)
                    logger.info("checkpoint loaded:" + str(model_checkpoint_path))
                    tokens = model_checkpoint_path.split("-")[-1]
                    # set global step
                    global_t = int(tokens)
                    print( ">>> global step set:", global_t)
        
        agent.reset()
        obs = env.reset()
        
        # for _ in range(10):
        #     env.render()
        #     sleep(0.1)

        done = False
        episode_reward = 0.
        episode_step = 0
        episodes = 0
        t = 0

        # skill 
        skill_done = True
        num_skill_steps = 0
        paction = None

        epoch = 0
        start_time = time.time()

        epoch_episode_rewards = []
        epoch_episode_steps = []
        epoch_episode_eval_rewards = []
        epoch_episode_eval_steps = []
        epoch_start_time = time.time()
        epoch_actions = []
        epoch_qs = []
        epoch_episodes = 0

        ## containers for hindsight
        if kwargs["her"]: 
            # logger.info("-"*50 +'\nWill create HER\n' + "-"*50)
            states, actions = [], []

        print("Ready to go!")
        for epoch in range(global_t, nb_epochs):
            
            # stat containers
            epoch_actor_losses = []
            epoch_critic_losses = []
            epoch_adaptive_distances = []

            eval_episode_rewards = []
            eval_qs = []
            eval_episode_success = []

            for cycle in range(nb_epoch_cycles):
                # print("cycle:%d"%cycle)
                # Perform rollouts.
                for t_rollout in range(int(nb_rollout_steps/MPI.COMM_WORLD.Get_size())):
                    # print(rank, t_rollout)
                    
                    # Predict next action.
                    if kwargs['look_ahead'] and (np.random.rand() < exploration.value(epoch*nb_epoch_cycles + cycle)):
                        if skill_done:
                            paction, planner_info = look_ahead_planner.create_plan(env, obs)
                            skill_done = False
                            num_skill_steps = 0
                            # print("skill:%d"%np.argmax(paction[:my_skill_set.len]))
                            # print(planner_info['sequence'])
                            # print("received meta action",paction)
                            # set_trace()
                        
                        primitives_prob = paction[:my_skill_set.len]
                        primitive_id = np.argmax(primitives_prob)
                        action = my_skill_set.pi(primitive_id=primitive_id, obs = obs.copy(), primitive_params=paction[my_skill_set.len:].copy())
                        num_skill_steps += 1
                    else:
                        action, q = agent.pi(obs, apply_noise=True, compute_Q=True)
                    
                    # Execute next action.
                    if rank == 0 and render:
                        env.render()
                    
                    new_obs, r, done, info = env.step(max_action * action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                    
                    # print("len of traj:%d, skill step:%d"%(len(planner_info["trajectories"]), num_skill_steps))
                    # traj_idx = min(len(planner_info["trajectories"])-1, num_skill_steps-1)

                    # print("start")
                    # print(obs[[0,1,2,3,4,5,-3,-2,-1]])
                    # print(planner_info['trajectories'][traj_idx][0][[0,1,2,3,4,5,-3,-2,-1]])

                    # print("action")
                    # print(action)
                    # print(planner_info['trajectories'][traj_idx][1])

                    # print("end")
                    # print(new_obs[[0,1,2,3,4,5,-3,-2,-1]])
                    # print(planner_info['trajectories'][traj_idx][2][[0,1,2,3,4,5,-3,-2,-1]])

                    if kwargs['look_ahead'] and (num_skill_steps == kwargs['commit_for'] or 
                                        my_skill_set.termination(new_obs, primitive_id, primitive_params = paction[my_skill_set.len:].copy())):
                        skill_done = True

                        # print("succ model pred", planner_info['next_state'][:6])
                        # print("actual end state",new_obs[:6], new_obs[-3:])

                        # print("diff succ model",np.linalg.norm(planner_info['next_state'][:6] - new_obs[:6]))
                        # print("diff nn model",np.linalg.norm(planner_info['next_state_nn'][:6] - new_obs[:6]))
                        # set_trace()

                    t += 1
                    if rank == 0 and render:
                        env.render()
                        sleep(0.1)
                    episode_reward += r
                    episode_step += 1

                    # Book-keeping.
                    epoch_actions.append(action)
                    # epoch_qs.append(q)
                    agent.store_transition(obs, action, r, new_obs, done)

                    ## storing info for hindsight
                    states.append(obs.copy())
                    actions.append(action.copy())

                    obs = new_obs

                    if done:
                        # print("reset")
                        # Episode done.
                        epoch_episode_rewards.append(episode_reward)
                        episode_rewards_history.append(episode_reward)
                        epoch_episode_steps.append(episode_step)
                        episode_reward = 0.
                        episode_step = 0
                        epoch_episodes += 1
                        episodes += 1

                        if kwargs["her"]:
                            # logger.info("-"*50 +'\nCreating HER\n' + "-"*50)

                            ## create hindsight experience replay
                            her_states, her_rewards = env.apply_hindsight(states, actions, new_obs.copy())
                            
                            ## store her transitions: her_states: n+1, her_rewards: n
                            for her_i in range(len(her_states)-2):
                                agent.store_transition(her_states[her_i], actions[her_i], her_rewards[her_i], her_states[her_i+1],False)
                            #store last transition
                            agent.store_transition(her_states[-2],actions[-1], her_rewards[-1], her_states[-1], True)


                            ## refresh the storage containers
                            del states, actions
                            states, actions = [], []

                        agent.reset()
                        obs = env.reset()
                        #print(obs)

                # print(rank, "Training!")
                # Train.
                
                for t_train in range(nb_train_steps):
                    # print(rank, t_train)
                    # Adapt param noise, if necessary.
                    if memory.nb_entries >= batch_size and t % param_noise_adaption_interval == 0:
                        distance = agent.adapt_param_noise()
                        epoch_adaptive_distances.append(distance)

                    cl, al, current_summary = agent.train(summary_var)
                    epoch_critic_losses.append(cl)
                    epoch_actor_losses.append(al)
                    agent.update_target_net()

                    if dologging and tf_sum_logging and rank==0:
                        
                        writer_t.add_summary(current_summary, epoch*nb_epoch_cycles*nb_train_steps + cycle*nb_train_steps + t_train)

                # print("Evaluating!")
                # Evaluate.
                
                
            if (eval_env is not None) and rank==0:
                for _ in range(nb_eval_episodes):
                    eval_episode_reward = 0.
                    eval_obs = eval_env.reset()
                    eval_obs_start = eval_obs.copy()
                    eval_done = False
                    while(not eval_done):
                        eval_action, eval_q = agent.pi(eval_obs, apply_noise=False, compute_Q=True)
                        eval_obs, eval_r, eval_done, eval_info = eval_env.step(max_action * eval_action)  # scale for execution in env (as far as DDPG is concerned, every action is in [-1, 1])
                        if render_eval:
                            sleep(0.1)
                            print("Render!")
                            
                            eval_env.render()
                            print("rendered!")
                        eval_episode_reward += eval_r

                        eval_qs.append(eval_q)
                        
                    eval_episode_rewards.append(eval_episode_reward)
                    eval_episode_rewards_history.append(eval_episode_reward)
                    eval_episode_success.append(eval_info["done"]=="goal reached")
                    if(eval_info["done"]=="goal reached"):
                        logger.info("success, training epoch:%d,starting config:"%epoch, eval_obs_start, 'final state', eval_obs)
                    
            if dologging and rank==0: 
                print("Logging!")
                # Log stats.
                epoch_train_duration = time.time() - epoch_start_time
                duration = time.time() - start_time
                stats = agent.get_stats()
                combined_stats = {}
                for key in sorted(stats.keys()):
                    combined_stats[key] = normal_mean(stats[key])

                # Rollout statistics.
                combined_stats['rollout/return'] = normal_mean(epoch_episode_rewards)
                if len(episode_rewards_history)>0:
                    combined_stats['rollout/return_history'] = normal_mean( np.mean(episode_rewards_history))
                else:
                    combined_stats['rollout/return_history'] = 0.
                combined_stats['rollout/episode_steps'] = normal_mean(epoch_episode_steps)
                combined_stats['rollout/episodes'] = np.sum(epoch_episodes)
                combined_stats['rollout/actions_mean'] = normal_mean(epoch_actions)
                combined_stats['rollout/actions_std'] = normal_std(epoch_actions)
                combined_stats['rollout/Q_mean'] = normal_mean(epoch_qs)
        
                # Train statistics.
                combined_stats['train/loss_actor'] = normal_mean(epoch_actor_losses)
                combined_stats['train/loss_critic'] = normal_mean(epoch_critic_losses)
                combined_stats['train/param_noise_distance'] = normal_mean(epoch_adaptive_distances)
                if kwargs['look_ahead']: combined_stats['train/exploration'] = exploration.value(epoch*nb_epoch_cycles + cycle)
                
                # Evaluation statistics.
                if eval_env is not None:
                    combined_stats['eval/return'] = normal_mean(eval_episode_rewards)
                    combined_stats['eval/success'] = normal_mean(eval_episode_success)
                    if len(eval_episode_rewards_history) > 0:
                        combined_stats['eval/return_history'] = normal_mean( np.mean(eval_episode_rewards_history) )
                    else:
                        combined_stats['eval/return_history'] = 0.
                    combined_stats['eval/Q'] = normal_mean(eval_qs)
                    combined_stats['eval/episodes'] = normal_mean(len(eval_episode_rewards))


                # Total statistics.
                combined_stats['total/duration'] = normal_mean(duration)
                combined_stats['total/steps_per_second'] = normal_mean(float(t) / float(duration))
                combined_stats['total/episodes'] = normal_mean(episodes)
                combined_stats['total/epochs'] = epoch + 1
                combined_stats['total/steps'] = t
                
                for key in sorted(combined_stats.keys()):
                    logger.record_tabular(key, combined_stats[key])
                logger.dump_tabular()
                logger.info('')
                logdir = logger.get_dir()
                if rank == 0 and logdir:
                    print("Dumping progress!")
                    if hasattr(env, 'get_state'):
                        with open(os.path.join(logdir, 'env_state.pkl'), 'wb') as f:
                            pickle.dump(env.get_state(), f)
                    if eval_env and hasattr(eval_env, 'get_state'):
                        with open(os.path.join(logdir, 'eval_env_state.pkl'), 'wb') as f:
                            pickle.dump(eval_env.get_state(), f)


                ## save tf model
                if rank==0  and (epoch+1)%save_freq == 0 :
                    print("Saving the model!")
                    os.makedirs(osp.join(logdir, "model"), exist_ok=True)
                    saver.save(U.get_session(), logdir+"/model/ddpg", global_step = epoch)





