import numpy as np
import itertools
import gym_bridge.analyze as analyze

class Agent(object):
    """Parent class to oversee general functionality."""
    def __init__(self, action_space, observation_space, RESHAPE_OBS=False,
                 RESHAPE_ENV=False):
        self.action_space = action_space
        self.observation_space = observation_space
        self.reshape_obs = RESHAPE_OBS
        self.reshape_env = RESHAPE_ENV

    def print_info(self,observation, valid_actions):
        print("obs:{}\nrew:{}\ndon:{}\nact:{}".format(observation, reward,
                                                      done, valid_actions))

class RandomAgent(Agent):
    """The world's simplest agent!"""
    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

    def __repr__(self):
        return "random"

    def act(self, observation, valid_actions):
        return np.random.choice(valid_actions)
        """Uniform distribution over actions, except pass with p=0.5."""

class PassingAgent(Agent):
    """An arguably even simpler agent..."""
    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

    def __repr__(self):
        return "passer"

    def act(self, observation, valid_actions):
        return 35 # pass

class DoublingRedoublingAgent(Agent):
    """Doubles or Redoubles whenever possible. Passes otherwise."""
    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

    def __repr__(self):
        return "doubler"

    def act(self, observation, valid_actions):
        if np.any(valid_actions == 36):
            return 36 # double
        elif np.any(valid_actions == 37):
            return 37 # redouble
        else:
            return 35 # pass

class AceBiddingAgent(Agent):
    "If it has an Ace in a suit, it bids 7suit if that is possible."
    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

    def act(self, observation, valid_actions):
        nonzero = np.nonzero(observation[:52])[0]
        for i in range(4):
            if np.any(nonzero == 51-i) and np.any(valid_actions == 33-i):
                return 33-i
        return 35

class UserAgent(Agent):
    """ User input. """
    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

    def act(self, observation, valid_actions):
        BIDMAP = { 'C':0, 'c':0, 'D':1, 'd':1, 'H':2, 'h':2,
                   'S':3, 's':3, 'N':4, 'n':4, 'NT':4, 'nt':4 }
        valid_action = None
        while valid_action is None:
            action = input("Choose your Action: ([1-7][c/d/h/s/n])|(p|d|r): ")
            a = None
            if len(action)==2:
                try:
                    a = (int(action[0])-1) * 5 + BIDMAP[action[1]]
                except:
                    print("Invalid action selected.")
                    continue
            elif len(action)==1:
                if action in 'pP':
                    a = 35
                elif action in 'dD':
                    a = 36
                elif action in 'rR':
                    a = 37
            else:
                continue
            if a in valid_actions:
                valid_action = a
        return valid_action
