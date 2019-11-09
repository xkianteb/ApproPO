import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='FrozenMarsRoverEnvDynamic-v0',
    entry_point='ApproPO.envs.gym_frozenmarsrover.envs:FrozenMarsRoverEnv',
    kwargs={'map_name' : '8x8', 'is_slippery': False, 'type':'rcpo'},
    max_episode_steps=300,
)
