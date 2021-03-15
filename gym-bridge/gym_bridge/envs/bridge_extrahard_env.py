import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np

class BridgeExtraHardEnv(gym.Env):
    '''
    This file solely exists because
    https://github.com/openai/gym/blob/master/docs/creating-environments.md
    told me to do so. Not used further.
    '''
    metadata = {'render.modes': ['human', 'terminal']}
