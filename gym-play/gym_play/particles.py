import numpy as np
from gym_bridge.analyze import deserialize_hands, serialize_hands
import collections
import random

class Particles:
    '''
    Contains belief functionality. A belief is a collection of particles, used
    in POMCP. Each particle represents a possible underlying deal, and is
    assigned a probability based on the observations during the game.
    '''
    def __init__(self):
        self.particles = {}

    def normalize_particles(self):
        "Normalizes the belief to sum to one."
        total = 0
        for key,val in self.particles.items():
            total+=val
        for key in self.particles.keys():
            self.particles[key] /= total

    def sample_deal(self):
        "Sample a deal (4x52 numpy array) from the belief (of particles)."
        cum_prob = 0
        target = random.uniform(0,1)
        for key,val in self.particles.items():
            cum_prob += val
            if target <= cum_prob:
                return serialize_hands(key)

    def construct_deal(self, declarer, our_hands, opp_cards, opp_played_cards, last_opp):
        "Randomly deals the opponents their possible remaining cards"
        deal = np.zeros([4,52])
        deal[declarer] = our_hands[0] # Declarer
        deal[(declarer+2)%4] = our_hands[1]
        np.random.shuffle(opp_cards)
        if (declarer+1)%4 != last_opp and len(opp_cards)%2 != 0:
            # There is a card imbalance.
            extra_card = 1
        else:
            extra_card = 0
        num_opp_cards = len(opp_cards)//2
        hands = [list(opp_cards[:num_opp_cards+extra_card]),
                 list(opp_cards[num_opp_cards+extra_card:])]
        for hand_idx, hand in enumerate(hands):
            serialized_hand = self._serialize_hand(hand)
            if hand_idx == 0:
                deal[(declarer+1)%4] = serialized_hand
            else:
                deal[(declarer+3)%4] = serialized_hand
        return deal

    def _serialize_hand(self, hand):
        serialized_hand = np.zeros((52,), dtype=np.int32)
        for c in hand:
            serialized_hand[c] = 1
        return serialized_hand

    def filter_known_cards(self, our_hands, all_played_cards):
        "Returns remaining unknown cards in numpy array"
        cards = list(np.arange(52))
        for our_hand in our_hands:
            for c in np.nonzero(our_hand)[0]:
                cards.remove(c)
        for played_cards in all_played_cards:
            for c in np.nonzero(played_cards)[0]:
                cards.remove(c)
        return np.array(cards)

    def add_belief(self, deal):
        if not deal in self.particles:
            self.particles[deal] = 1
        else:
            self.particles[deal] += 1
