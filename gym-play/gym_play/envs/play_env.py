import gym
import sys
from gym_bridge.consts import *
import random
from gym import error, spaces, utils
from gym.utils import seeding
from gym_play.state import State
from gym_bridge.analyze import deserialize_hands
import numpy as np

class PlayEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, declarer, contract, hands=None):
      self.declarer = declarer
      self.initial_player = (declarer+1)%4
      self.active_player = self.initial_player
      self.contract = contract
      self.trump = self.contract.bid%5
      self.state = State(self.active_player, self.trump)
      self.action_space = spaces.Discrete(52)
      self.observation_space = spaces.MultiBinary(410)  # TODO: other
      self.played_cards = np.zeros([4,52], dtype=np.int32)
      self.tricks_taken = np.zeros(4, dtype=np.int32)
      if hands is None:
          self.hands = self.deal_and_serialize_hands()
      else:
          self.hands=hands

    def step(self, action):
        self.handle_action(action) # Append to trick and check if trick done
        done = self.check_done() # Check if no cards remaining
        return -1, 0, done, {} # o, r, done, LOG

    def handle_action(self, action):
        self.append_card_to_trick(action)
        self._append_card_to_played_cards(action)
        if self.state.cards_played == 4:
            self.handle_trick_done()
        else:
            self.next_player()

    def append_card_to_trick(self, action):
        self.state.cards_played += 1
        assert self.hands[self.active_player][action] == 1
        self.hands[self.active_player][action] = 0
        self.state.cards_this_trick[self.active_player] = action

    def _append_card_to_played_cards(self, action):
        self.played_cards[self.active_player, action] = 1

    def handle_trick_done(self):
        winner = self.state.trick_winner()
        self.tricks_taken[winner] +=1
        self.tricks_taken[(winner+2)%4] += 1
        self.active_player = winner
        self.state.reset(self.active_player)

    def check_done(self):
        return len(np.nonzero(self.hands[0])[0]) == 0

    def reset(self):
        ...
    def render(self, mode='human'):
        ...
    def close(self):
        ...

    def deal_and_serialize_hands(self):
        """Deals NESW 13 cards and one-hot encodes each hand into 52-bits."""
        cards = np.arange(52)
        np.random.shuffle(cards)
        hands = np.split(cards, 4)
        serialized_hands = []
        for hand in hands:
            serialized_hand = np.zeros((52,))
            for card in hand:
                serialized_hand[card] = 1
            serialized_hands.append(serialized_hand)
        return np.array(serialized_hands)

    def next_player(self):
        self.active_player = (self.active_player + 1) % 4

    def get_valid_actions(self):
        "Returns numpy array containing valid_actions [4, 14, 29, ...]"
        valid_actions = np.nonzero(self.hands[self.active_player])[0]
        if self.active_player == self.state.first_player:
            return valid_actions
        else:
            suit_lead = self.state.cards_this_trick[self.state.first_player] % 4
            suits_in_hand = [c % 4 for c in valid_actions]
            if suit_lead in suits_in_hand:
                return [valid_actions[c_idx] for c_idx, c in \
                        enumerate(suits_in_hand) if c == suit_lead]
            else:
                return valid_actions

    def check_done(self):
        return np.sum(self.tricks_taken) == 26
