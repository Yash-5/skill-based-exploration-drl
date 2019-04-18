from gym.envs.registration import register
import HER

# Sawyer wrappers
register(
    id='WrapSawyerReachXYEnv-v1',
    entry_point='HER.envs.SawyerWrapper:WrapSawyerReachXYEnv',
    kwargs={
    'hide_goal_markers': True,
    'norm_order': 2,
    'nb_step_limit': 200,
    }
)

register(
    id='WrapSawyerPushAndReachEnvEasy-v0',
    entry_point='HER.envs.SawyerWrapper:WrapSawyerPushAndReachXYEnv',
    kwargs=dict(
        goal_low=(-0.15, 0.4, 0.02, -.1, .45),
        goal_high=(0.15, 0.7, 0.02, .1, .65),
        puck_low=(-.1, .45),
        puck_high=(.1, .65),
        hand_low=(-0.15, 0.4, 0.02),
        hand_high=(0.15, .7, 0.02),
        norm_order=2,
        xml_path='sawyer_xyz/sawyer_push_puck.xml',
        reward_type='state_distance',
        reset_free=False,
        clamp_puck_on_step=True,
        nb_step_limit=200,
    )
)

register(
    id='WrapSawyerPushAndReachEnvMedium-v0',
    entry_point='HER.envs.SawyerWrapper:WrapSawyerPushAndReachXYEnv',
    kwargs=dict(
        goal_low=(-0.2, 0.35, 0.02, -.15, .4),
        goal_high=(0.2, 0.75, 0.02, .15, .7),
        puck_low=(-.15, .4),
        puck_high=(.15, .7),
        hand_low=(-0.2, 0.35, 0.05),
        hand_high=(0.2, .75, 0.3),
        norm_order=2,
        xml_path='sawyer_xyz/sawyer_push_puck.xml',
        reward_type='state_distance',
        reset_free=False,
        clamp_puck_on_step=True,
        nb_step_limit=200,
    )
)

register(
    id='WrapSawyerPushAndReachEnvHard-v0',
    entry_point='HER.envs.SawyerWrapper:WrapSawyerPushAndReachXYEnv',
    kwargs=dict(
        goal_low=(-0.25, 0.3, 0.02, -.2, .35),
        goal_high=(0.25, 0.8, 0.02, .2, .75),
        puck_low=(-.2, .35),
        puck_high=(.2, .75),
        hand_low=(-0.25, 0.3, 0.02),
        hand_high=(0.25, .8, 0.02),
        norm_order=2,
        xml_path='sawyer_xyz/sawyer_push_puck.xml',
        reward_type='state_distance',
        reset_free=False,
        clamp_puck_on_step=True,
        nb_step_limit=200,
    )
)

# joint space envs
register(
    id='Reacher3dj-v0',
    entry_point='HER.envs.joint_space_reacher3d:BaxterEnv',
    kwargs={'max_len':20}
)

# Fetch envs
register(
    id='transfer-v0',
    entry_point='HER.envs.transfer_withextras:BaxterEnv',
    kwargs={'max_len':10}
)

register(
    id='fetchpnp-v0',
    entry_point='HER.envs.fetchpnp:FetchPnp',
    kwargs={'reward_type':''},
    max_episode_steps=50,
)

# Baxter envs
register(
    id='putainb-v3',
    entry_point='HER.envs.putainb_withextras:BaxterEnv',
    kwargs={'max_len':100, 
            'filename':"mjc/putainb4.xml"
            }
)

register(
    id='putainbt-v3',
    entry_point='HER.envs.putainb_withextras:BaxterEnv',
    kwargs={'max_len':100,
            'test':True,
            'filename':"mjc/putainb4.xml"
            }
)

register(
    id='putainb-v2',
    entry_point='HER.envs.putainb_withextras:BaxterEnv',
    kwargs={'max_len':100, 
            'filename':"mjc/putainb3.xml"
            }
)

register(
    id='putainbt-v2',
    entry_point='HER.envs.putainb_withextras:BaxterEnv',
    kwargs={'max_len':100,
            'test':True,
            'filename':"mjc/putainb3.xml"
            }
)

register(
    id='putainb-v1',
    entry_point='HER.envs.putainb_withextras:BaxterEnv',
    kwargs={'max_len':100, 
            'filename':"mjc/putainb2.xml"
            }
)

register(
    id='putainbt-v1',
    entry_point='HER.envs.putainb_withextras:BaxterEnv',
    kwargs={'max_len':100,
            'test':True,
            'filename':"mjc/putainb2.xml"
            }
)

register(
    id='putaoutb-v2',
    entry_point='HER.envs.putaoutbtable_withextras:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='putaoutbt-v2',
    entry_point='HER.envs.putaoutbtable_withextras:BaxterEnv',
    kwargs={'max_len':50, 'test':True}
)

register(
    id='putaoutb-v1',
    entry_point='HER.envs.putacompleteoutb_withextras:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='putaoutbt-v1',
    entry_point='HER.envs.putacompleteoutb_withextras:BaxterEnv',
    kwargs={'max_len':50, 'test':True}
)

register(
    id='putaoutb-v0',
    entry_point='HER.envs.putaoutb_withextras:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='putaoutbt-v0',
    entry_point='HER.envs.putaoutb_withextras:BaxterEnv',
    kwargs={'max_len':50, 'test':True}
)

register(
    id='putaonb-v0',
    entry_point='HER.envs.putaonb_withextras:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='putaonbt-v0',
    entry_point='HER.envs.putaonb_withextras:BaxterEnv',
    kwargs={'max_len':50, 'test':True}
)

register(
    id='picknmove-v5',
    entry_point='HER.envs.picknmove_withextras_singlestart:BaxterEnv',
    kwargs={'max_len':10}
)

register(
    id='picknmove-v4',
    entry_point='HER.envs.picknmove_withgap:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='picknmovet-v4',
    entry_point='HER.envs.picknmove_withgap:BaxterEnv',
    kwargs={'max_len':50, 'test':True}
)


register(
    id='putainb-v0',
    entry_point='HER.envs.putainb_withextras:BaxterEnv',
    kwargs={'max_len':100}
)

register(
    id='putainbt-v0',
    entry_point='HER.envs.putainb_withextras:BaxterEnv',
    kwargs={'max_len':100,
            'test':True}
)

register(
    id='Reacher2d-v0',
    entry_point='HER.envs.reacher2d:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='Reacher2d-v1',
    entry_point='HER.envs.reacher2d_rel:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='Reacher3d-v0',
    entry_point='HER.envs.reacher3d:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='pusher-v0',
    entry_point='HER.envs.pusher:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='pusher-v1',
    entry_point='HER.envs.close_pusher:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='pusher-v2',
    entry_point='HER.envs.close_pusher_rel:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='grasping-v0',
    entry_point='HER.envs.grasping:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='grasping-v1',
    entry_point='HER.envs.grasping_rel:BaxterEnv',
    kwargs={'max_len':20}
)

register(
    id='grasping-v2',
    entry_point='HER.envs.grasping_withgap:BaxterEnv',
    kwargs={'max_len':20}
)
register(
    id='graspingt-v2',
    entry_point='HER.envs.grasping_withgap:BaxterEnv',
    kwargs={'max_len':20, 'test':True}
)

register(
    id='picknmove-v0',
    entry_point='HER.envs.picknmove:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='picknmove-v1',
    entry_point='HER.envs.picknmove_rel:BaxterEnv',
    kwargs={'max_len':50}
)


register(
    id='picknmove-v2',
    entry_point='HER.envs.picknmove_withextras:BaxterEnv',
    kwargs={'max_len':50}
)

register(
    id='picknmovet-v2',
    entry_point='HER.envs.picknmove_withextras:BaxterEnv',
    kwargs={'max_len':50, 'test':True}
)

register(
    id='picknmoved-v2',
    entry_point='HER.envs.picknmovedense_withextras:BaxterEnv',
    kwargs={'max_len':100}
)

register(
    id='picknmovedt-v2',
    entry_point='HER.envs.picknmovedense_withextras:BaxterEnv',
    kwargs={'max_len':50, 'test':True}
)

register(
    id='picknmove-v3',
    entry_point='HER.envs.picknmove_withextras:BaxterEnv',
    kwargs={'max_len':100}
)

register(
    id='picknmovet-v3',
    entry_point='HER.envs.picknmove_withextras:BaxterEnv',
    kwargs={'max_len':100, 'test':True}
)
