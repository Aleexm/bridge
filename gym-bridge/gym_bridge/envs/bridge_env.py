import gym
from gym import error, spaces, utils
import numpy as np
from gym.utils import seeding
from gym_bridge.state import State
import gym_bridge.analyze as analyze
import random
import os
import json
from gym_bridge.consts import *

class BridgeEnv(gym.Env):
    '''
    Bridge environment for Bridge Bidding. Contains a State object which
    contains the hands, vulnerability and the bidding history.
        - A hand is encoded as a 52-bitvector of the ordered set of cards,
          i.e. 2C<2D<...<AH<AS. A '1' indicates this card is held.
        - The vulnerability is encoded as a 2-bitvector. First bit is NS vuln,
          second bit is EW vuln.
        - The bidding history is encoded as a 318-bitvector,
          where a 1 in the i-th entry denotes that the i-th bid in the possible
          maximum bidding sequence is called, i.e.
          p-p-p|1C-p-p-d-p-p-r-p-p|1D-p-p-d-p-p-r-p-p|...|7N-p-p-d-p-p-r-p-p.
          The final pass can be inferred.
          See Competitive Bridge Bidding with Deep Neural Networks (Rong et al.)
          for details.

    Observations in this environment are the State, which contains all global
    information. A Controller (see class) then filters the joint observation
    and passes the correct local information to the acting agent.

    At each timestep, one of the four agents makes a bid and the state is updated.
    After three consecutive passes, the bidding phase concludes (4 passes if no
    contract bid was ever made.).

    Actions are integers in {0,1,...37}, and are explained in ACTION_MEANING

    '''

    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.action_space = spaces.Discrete(38)
        self.seed()
        self.observation_space = spaces.MultiBinary(410)
        self.state = State()
        self.initial_player = random.choice(np.arange(4))
        self.acting_player = self.initial_player
        self.reset()
        self.local = False

    def step(self, action):
        """Updates the environment based on the action that was taken."""
        assert self.action_space.contains(action)
        last_bid_idx = np.nonzero(self.state.history)[0][-1] if \
                       len(np.nonzero(self.state.history)[0]) > 0 else -1
        self._is_valid_action(action, last_bid_idx) # Raises assertion if action is invalid
        done = self._check_done(action, last_bid_idx)
        if not done:
            self.state.update_state(self.acting_player, action, last_bid_idx)
            self._update_valid_actions()
        if done:
            contract, declarer = self.state.get_contract_declarer(self.initial_player)
            self.state.contract = contract
            self.state.declarer = declarer
        self.acting_player = (self.acting_player + 1) % 4
        return self.state, 0, done, {}

    def _check_done(self, action, last_bid_idx):
        '''
        Returns whether the game has finished (4 or 3 consecutive passes):
        4 initial passes or 3 passes after a contract bid.

        args:
            - action(int): the action (see ACTION_MEANING)
            - last_bid_idx(int): index in self.state.history of last bid
        '''
        curr_idx = last_bid_idx + 1
        # Contract bids, doubles and redoubles are always % 3 == 0. Thus,
        # if last_bid_idx % 3 == 2, we know that there have been two passes
        # Since the last non-passing bid. Action == 35 is the third pass.
        # if last_bid_idx == 2 and action == 35:
        #     return True
        if last_bid_idx > 0 and last_bid_idx % 3 == 2 and action == 35:
            return True
        return False

    def _update_valid_actions(self):
        '''Updates the 38-bit vector state.valid_actions.'''
        self.valid_actions = np.zeros(38)
        valid_actions = self.get_valid_actions()
        for a in valid_actions:
            self.valid_actions[a] = 1

    def _is_valid_action(self, action, last_bid_idx):
        '''
        Checks whether the action is valid. This should always pass as we
        check for action validity in get_valid_actions(). Might remove later...

        args:
            - action(int): the action (see ACTION_MEANING below)
            - last_bid_idx(int): index in self.state.history of last bid
        '''
        if action < 35: # contract bid
            assert last_bid_idx < action * 9 + 3, \
            "contract bid was non-increasing! {}, {}".format(action,
                                                             self.state.history)
        elif action == 36: # double
            last_bid_idx = last_bid_idx - 3
            # Contract-pass-pass-double:
            assert last_bid_idx % 9 < 3, \
            "double not applicable! {} {}".format(action, self.state.history)
            # contract bid made:
            assert self.state.history[(last_bid_idx // 9) * 9 + 3] == 1, \
            "No contract to be doubled! {} {}".format(action, self.state.history)
        elif action == 37: # redouble
            last_bid_idx = last_bid_idx - 3
            # Contract-p-p-d-p-p-r at most:
            assert last_bid_idx % 9 < 6, \
            "redouble not applicable! {} {}".format(action, self.state.history)
            # Contract bid made
            assert self.state.history[(last_bid_idx) // 9 * 9 + 3] == 1, \
            "No contract bid to be redoubled! {} {}".format(action,
                                                            self.state.history)
            # Contract bid was doubled made:
            assert self.state.history[(last_bid_idx) // 9 * 9 + 6] == 1, \
            "No double to redouble! {} {}".format(action, self.state.history)

    def get_valid_actions(self):
        """Returns a numpy array of valid actions in [0,1,...37]."""
        nonzero = np.nonzero(self.state.history)[0] # gets the array only
        actions = np.arange(0,36) # Assume all contract bids and 'p' are legal
        if len (nonzero) == 0: #no bids made
            return actions # i.e. everything but double/redouble valid
        else:
            tmp = nonzero[-1] - 3 # Remove invalid contract bids hereafter
            actions = actions[tmp // 9 + 1:] if tmp >= 0 else actions
            if tmp % 9 == 0 or tmp % 9 == 2: # last contract bid 1 or 3 ago
                actions = np.append(actions, 36) # So doubling is valid
            if tmp % 9 == 3 or tmp % 9 == 5: # last double 1 or 3 ago
                actions = np.append(actions, 37) # So redoubling is valid
        return actions

    def handle_scoring(self, duplicate_idx, dds=None):
        '''
        On termination of the bidding phase, compares the number of tricks
        taken to the contract, and scores accordingly.

        args:
            - dds(list(list(int))): DDS table containing tricks taken (out of 13)
                                    If we load a deal and DDS from data, it is
                                    None. Otherwise, we pass the DDS for this
                                    deal.
        returns:
            - score(int): Raw score for the contract, from the perspective of
                          NS partnership
        '''
        self.state.set_contract_declarer(self.initial_player)
        if self.state.contract is None:
            return 0
        if dds is None:
            tricks_taken = self.state.dds[self.state.declarer][self.state.contract.bid%5]
        else:
            tricks_taken = dds[self.state.declarer][self.state.contract.bid%5]
        score = analyze.score(self.state.contract, self.state.declarer,
                              self.state.vulnerability, tricks_taken)

        if self.state.declarer % 2 != 0: # Always score from our perspective
            score *= -1
        return score

    def compute_pass_value(self, duplicate_idx, scores, dds_table=None):
        '''
        Computes the value of the final pass of our agents. Is positive if we
        passed correctly, negative if we could have scored better.

        args:
            - duplicate_idx(int): 0 or 1 indicating whether we sit NS or EW resp
            - scores(list(int)): Raw scores from NS perspective always.
            - dds_table(list(list(int))): Is passed if we did not load deal, dds
                                          from data.

        returns:
            - pass_value(int): the value of our agents' final pass.
        '''
        declarers = self.state.get_all_our_declarers(self.initial_player,
                                                     duplicate_idx)
        dds_table = self.state.dds if dds_table is None else dds_table
        max_score, max_entry = analyze.compute_max_score(
            dds_table, declarers, self.state.vulnerability)
        if self.state.contract is None:
            assert scores[duplicate_idx] == 0
            return analyze.IMP_difference(0, max_score) / 24

        max_contract = (dds_table[max_entry[0]][max_entry[1]] - 7) * 5 \
                       + max_entry[1]
        achievable = True if self.state.contract.bid < max_contract else False
        score = scores[duplicate_idx]
        if duplicate_idx == 1: # Score should be from EW perspeective, not NS
            score*=-1

        if self.state.declarer % 2 == duplicate_idx: # We made this contract
            if int(score) == int(max_score):
                pass_value = analyze.IMP_difference(np.abs(score), 0) / 24
            elif score < 0 and not achievable: # We simply bid too high
                pass_value = 0.3
            else: # We could have done better
                pass_value = analyze.IMP_difference(score, max_score) / 24
        else: # Opp made the contract
            if achievable: # We could have done better
                pass_value = analyze.IMP_difference(score, max_score)/24
            else:
                pass_value = 0.3

        return pass_value

    def reset(self, from_deal=False, soft=False):
        '''
        If not soft, resets the environment and returns an initial observation.
        Soft is used for duplicate bridge, to reset only part of the environment.
        Hands etc are kept, as per definition.

        args:
            - from_deal(bool): whether to load in deal and DDS table from data.
            - soft(bool): whether to create a new environment (True) or
                          only reset the bidding history, declarer, contract
                          and acting player (False). Used for Duplicate Bridge.

        returns:
            - self.state(State): the state of the game (hands, vuln, etc)
        '''
        if soft:
            self.state.reset_history()
            self.state.declarer = None
            self.state.contract = None
            self.acting_player = self.initial_player
        else:
            self.state = State()
            self.initial_player = random.choice(np.arange(4))
            self.acting_player = self.initial_player

            if from_deal: # Load deal and DDS table from random sample
                not_found=True
                while not_found:
                    if self.local:
                        path = os.path.join(os.getcwd(), 'gym_bridge', 'data')
                    else:
                        path = os.path.join('/scratch', 'avmandersloot',
                                            'gym_bridge', 'data')
                    dir = np.random.choice(os.listdir(path))
                    while len(os.listdir(os.path.join(path, dir))) == 0:
                        dir = np.random.choice(os.listdir(path))
                    file = np.random.choice(os.listdir(os.path.join(path, dir)))
                    try:
                        with open(os.path.join(path, dir, file), 'r') as f:
                            data = json.loads(f.read())
                            idx = np.random.randint(0, len(data['dds'])-1)
                            self.state.hands = analyze.serialize_hands(data['hands'][idx])
                            self.state.dds = np.array(data['dds'][idx], dtype=np.int32)
                            not_found = False
                    except:
                        pass
        return self.state

    def render(self, mode='human'):
        ...

    def close(self):
        ...
