import numpy as np
from gym_bridge.consts import *
from gym_bridge.bid import Bid

class Contract():
    def __init__(self, bid, doubled, redoubled):
        self.bid = bid
        self.doubled = doubled
        self.redoubled = redoubled

    def __repr__(self):
        return "bid: {}, d: {} r: {}".format(self.bid,
                                             self.doubled, self.redoubled)

class State():
    def __init__(self):
        self.hands = self.deal_and_serialize_hands() #NESW
        self.vulnerability = np.random.choice([0,1], size=(2,)) #2 bits, NS, EW resp.
        self.history = np.zeros(318) # Increasing bidding order
        self.human_history = list()
        self.declarer = None # Will remain none if 4 consecutive, initial passes
        self.contract = None
        self.dds = None # Will be set if deal and DDS was loaded from data.

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

    def reset_history(self):
        self.history = np.zeros(318)
        self.human_history = list()

    def reshape_history(self, acting_player, initial_player):
        '''
        Reshapes the bidding history to a 175-bitvector. First 35 are the
        player's bids, next 35 are his partner's bids, next 35x2 are left,
        right opponent resp., and last 35 indicate bids that have been
        doubled/redoubled.

        returns:
            - bids(list(int)): 175 Bitvector representing bidding history.
        '''

        bids = np.zeros((5,35))
        for player, bid in enumerate(np.nonzero(self.history)[0]):
            if (player+initial_player) % 4 == acting_player:
                position = 0
            elif (player+initial_player) % 4 == (acting_player + 2) % 4:
                position = 1
            elif (player+initial_player) % 4 == (acting_player + 1) % 4:
                position = 2
            else:
                position = 3
            if (bid - 3) % 9 == 0:
                bids[position, (bid-3) // 9] = 1
            elif (bid - 6) % 9 == 0:
                bids[4, (bid - 6) // 9] = 1
        return bids.flatten()

    def set_contract_declarer(self, initial_player):
        '''
        Sets self.declarer for this game. This method does so by first finding
        the player who made the last contract bid, and then finding which player
        of that partnership first made the bid in that suit, i.e. the declarer.

        args:
            - initial_player(int): 0:N, 1:E, 2:S, 3:W
        '''
        nonzero = np.nonzero(self.history)[0]
        initial_player = initial_player
        if nonzero[-1] < 3: # Everyone passed initially
            return # Keep self.declarer, self.contract as None

        doubled = False; redoubled = False # Get contract hereafter
        for i in range(len(nonzero)): # get last contract bidder (loop backwards)
            if (nonzero[len(nonzero)-i-1] - 3) % 9 == 0: # Contract bid!
                contract_bid = nonzero[len(nonzero)-i-1] // 9 # Set contract
                contract_player = (initial_player + len(nonzero)-i-1) % 4
                break # Found contract_player so exit loop
            if (nonzero[len(nonzero)-i-1] - 3) % 9 == 3: # Double
                doubled = True
            if (nonzero[len(nonzero)-i-1] - 3) % 9 == 6: # Redouble
                redoubled = True
        self.contract = Contract(contract_bid, doubled, redoubled)

        contract_suit = ((nonzero[-1] - 3) // 9) % 5

        for player, b in enumerate(nonzero): # Get the declarer
            if ((b - 3) % 45) == contract_suit * 9: # Bid in contract suit
                if (initial_player + player) % 4 == contract_player: # He started this suit himself
                    self.declarer = contract_player
                    break # Found declarer
                elif (initial_player + player) % 4 == (contract_player + 2) % 4: # Partner started
                    self.declarer = (contract_player + 2) % 4
                    break # Found declarer
        assert self.contract is not None, "No contract set"
        assert self.declarer is not None, "No declarer set"

    def get_all_our_declarers(self, initial_player, duplicate_idx):
        declarers = [None for i in range(5)] # Will contain the declarer for each suit
        for player, b in enumerate(np.nonzero(self.history)[0]):
            if (b-3)%9 == 0 and ((player+initial_player) % 4) % 2 == duplicate_idx: # Contract bid
                suit = ((b-3)//9)%5
                if declarers[suit] is None: # No declarer set yet
                    declarers[suit] = (player+initial_player) % 4
        for i in range(len(declarers)):
            if declarers[i] is None:
                if (player + initial_player + 1) % 2 != duplicate_idx:
                    declarers[i] = (player + initial_player) % 4
                else:
                    declarers[i] = (player + initial_player + 1) % 4
            assert declarers[i] % 2 == duplicate_idx
        return declarers

    def update_state(self, active_player, action, last_bid_idx):
        '''
        Updates the state, by adding to it the recent action.
        Additionally, updates the human-readable state.

        args:
            - action(int): the action (see ACTION_MEANING)
            - last_bid_idx(int): index in self.state.history of last bid
        '''
        if action <= 34: #contract bid
            self.history[action * 9 + 3] = 1
        elif action == 35: # pass
            curr_idx = last_bid_idx + 1
            if curr_idx > 0 and curr_idx % 3 == 0: #  Done
                return
            else:
                self.history[curr_idx] = 1
        elif action == 36: # double
            curr_idx = (last_bid_idx // 3 + 1) * 3 # Equals last contract bid+3
            self.history[curr_idx] = 1
        else: # redouble
            curr_idx = (last_bid_idx // 9 + 1) * 9 # Equals last contract bid+6
            self.history[curr_idx] = 1

        self._update_human_state(active_player, action)

    def _update_human_state(self, active_player, action):
        '''
        Updates the human_state, which is a list of Bids

        args:
            - active_player(Player): the active player
            - action(int): the action (see ACTION_MEANING)
        '''
        if action >= 35:
            suit = None; rank = None
        else:
            suit = Suit(action%5)
            rank = action//5+1
        bid = Bid(active_player, Bid_Enum(action), suit, rank)
        self.human_history.append(bid)

    def get_contract_declarer(self, initial_player):
        contract = self._get_contract(initial_player)
        declarer = self._get_declarer(contract)
        return contract, declarer

    def _get_contract(self, initial_player):
        if all(first_3.bid == Bid_Enum.P for first_3 in self.human_history[:3]):
            return None
        doubled = False; redoubled = False
        for b in reversed(self.human_history):
            if b.bid is Bid_Enum.D:
                doubled = True
            if b.bid is Bid_Enum.R:
                redoubled = True
            if b.bid not in [Bid_Enum.P, Bid_Enum.D, Bid_Enum.R]:
                return Contract(b, doubled, redoubled)

    def _get_declarer(self, contract):
        if contract is None:
            return None
        contract_team_1 = contract.bid.player
        contract_team_2 = (contract.bid.player + 2) % 4
        for b in self.human_history:
            if b.suit == contract.bid.suit:
                if b.player in [contract_team_1, contract_team_2]:
                    return b.player
