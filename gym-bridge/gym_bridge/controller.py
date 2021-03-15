import numpy as np
import random

class Controller():
    '''
    Master agent which delegates action selection and observations to correct
    simple agent depending on whose turn it is. Each timestep, the Controller
    'takes an action', meaning it lets the correct sub-agent perform an action.
    Controller receives joint observations, i.e. observation.hands contains all
    four hands of the players. It passes the local information on to the correct
    sub-agent.
    '''

    def __init__(self, agents, NUM_AGENTS=4):
        self.NUM_AGENTS = NUM_AGENTS # NESW
        self.agents = agents # 4 agents it oversees

    def act(self, acting_player, initial_player, observation,
            valid_actions, duplicate_idx=0):
        '''
        The controller acts by calling the correct subagent's act() method.
        First, it filters the subagent's observation from the joint observation,
        then it lets this agent act based on this local information.

        Additionally, it saves this observation and action to the
        agent's memory in case it implements this functionality.

        args:
            - acting_player(int): The agent to act. N:0, E:1, S:2, W:3
            - initial_player(int): The initial player
            - observation(State): joint observation (e.g. observation.hands
                                                     contains all hands).
            - valid_actions(list(int)): list of valid actions in {0,1,...37}
            - duplicate_idx(int): if 1, this means that we are playing the
                                  second table in duplicate bridge format. Thus,
                                  we rotate the acting agent 1 pos. clockwise.

        returns:
            - action(int): Action {0,1,...37}: 0<=a<=34 is ordered set of bids,
                           i.e. 1C<1D<...<7H<7S<7N, 35=Pass, 36=Double, 37=Redouble
        '''
        agent_obs = self.get_subagent_observation(acting_player,
                                                  initial_player,
                                                  observation,
                                                  valid_actions,
                                                  duplicate_idx)
        actor = (acting_player - duplicate_idx) % self.NUM_AGENTS
        action = self.agents[actor].act(agent_obs, valid_actions)
        if action == -1: # NaN Probs
            self.agents[actor].remove_last_trajectory()
            return action
        try:
            self.agents[actor].save_observation_action(agent_obs, action)
        except: # Agent doesn't save anything to memory, continue
            pass
        return action

    def get_subagent_observation(self, acting_player, initial_player,
                                 observation, valid_actions, duplicate_idx=0):
        '''
        Filters the joint observation to correct local information based on the
        acting_player.

        args:
            - acting_player(int): N:0, E:1, S:2, W:3
            - observation(State): Joint observation

        returns:
            - agent_obs(np.array(len=372)): First 52 are one-hot encoding of hand,
                                          next 2 are encoding of vuln.,
                                          next 318 are encoding of bid history.
                  - If RESHAPE_OBS, array is 229 instead.
        '''
        if acting_player == 1 or acting_player == 3: # Flip vuln for EW: base is NS
            agent_vuln = np.flip(observation.vulnerability)
        else:
            agent_vuln = observation.vulnerability
        if self.agents[(acting_player - duplicate_idx) % 4].reshape_obs:
            history = observation.reshape_history(acting_player, initial_player)
        else:
            history = observation.history
        one_hot_valid_actions = np.zeros(38)
        for a in valid_actions:
            one_hot_valid_actions[a] = 1
        agent_obs = np.concatenate(
            [observation.hands[acting_player],
             agent_vuln,
             history,
             one_hot_valid_actions], axis=None)
        return agent_obs
