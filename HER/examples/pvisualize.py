import argparse
import time
import os
import logging
from HER import logger, bench
from HER.common.misc_util import (
    set_global_seeds,
    boolean_flag,
)

from HER.ddpg.skills import SkillSet
import HER.pddpg.testing as testing
from HER.pddpg.models import Actor, Critic
from HER.pddpg.memory import Memory
from HER.pddpg.noise import *

import gym
import tensorflow as tf

## my imports
import HER.envs

def run(env_id, seed, noise_type, layer_norm, evaluation, **kwargs):
    # Configure things.
    rank = 0
    if rank != 0:
        logger.set_level(logger.DISABLED)

    dologging = kwargs["dologging"]
    
    # Create envs.
    env = gym.make(env_id)
    gym.logger.setLevel(logging.WARN)

    if evaluation and rank==0:
        eval_env = gym.make(env_id)
    else:
        eval_env = None

    tf.reset_default_graph()
    
    if kwargs['skillset']:
        skillset_file = __import__("HER.skills.%s"%kwargs['skillset'], fromlist=[''])
        my_skill_set = SkillSet(skillset_file.skillset)
        nb_actions = my_skill_set.params +  my_skill_set.len

    else:
        nb_actions = env.action_space.shape[-1]

    # Configure components.
    memory = Memory(limit=int(1e6), action_shape=env.action_space.shape, observation_shape=env.observation_space.shape)
    critic = Critic(layer_norm=layer_norm)
    
    if kwargs['skillset'] is None:
        actor = Actor(discrete_action_size = env.env.discrete_action_size, cts_action_size = nb_actions - env.env.discrete_action_size, layer_norm=layer_norm)
        my_skill_set = None
    else:
        # pass
        # get the skillset and make actor accordingly
        actor = Actor(discrete_action_size = my_skill_set.len , cts_action_size = nb_actions - my_skill_set.len, layer_norm=layer_norm)

    # Seed everything to make things reproducible.
    seed = seed + 1000000 * rank
    logger.info('rank {}: seed={}, logdir={}'.format(rank, seed, logger.get_dir()))
    
    set_global_seeds(seed)
    env.seed(seed)
    if eval_env is not None:
        eval_env.seed(seed)

    # Disable logging for rank != 0 to avoid noise.
    if rank == 0:
        start_time = time.time()

    testing.test(env=env, eval_env=eval_env, param_noise=None, action_noise=None,
        actor=actor, critic=critic, memory=memory, my_skill_set=my_skill_set, **kwargs)
    env.close()
    if eval_env is not None:
        eval_env.close()
    if rank == 0:
        logger.info('total runtime: {}s'.format(time.time() - start_time))


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--env-id', type=str, default='Baxter-v1')
    boolean_flag(parser, 'render-eval', default=True)
    boolean_flag(parser, 'layer-norm', default=True)
    boolean_flag(parser, 'normalize-returns', default=False)
    boolean_flag(parser, 'normalize-observations', default=True)
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--critic-l2-reg', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)  # per MPI worker
    parser.add_argument('--actor-lr', type=float, default=1e-4)
    parser.add_argument('--critic-lr', type=float, default=1e-3)
    boolean_flag(parser, 'popart', default=False)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--reward-scale', type=float, default=1.)
    parser.add_argument('--clip-norm', type=float, default=None)
    parser.add_argument('--nb-eval-steps', type=int, default=10000)  # per epoch cycle and MPI worker
    parser.add_argument('--noise-type', type=str, default='ou_0.02')  # choices are adaptive-param_xx, ou_xx, normal_xx, none
    boolean_flag(parser, 'evaluation', default=True)

    ## saving and restoring param parser
    parser.add_argument('--log-dir', type=str, default='/tmp/run1')
    parser.add_argument('--save-freq', type=int, default=100000)
    parser.add_argument('--restore-dir', type=str, default=None)
    boolean_flag(parser, 'dologging', default=False)    

    parser.add_argument('--skillset', type=str, default='set8')
    parser.add_argument('--commit-for', type=int, default=1)
    

    args = parser.parse_args()
    dict_args = vars(args)
    return dict_args


if __name__ == '__main__':
    args = parse_args()

    # Run actual script.
    try:
        run(**args)
    except KeyboardInterrupt:
        print("Exiting!")
