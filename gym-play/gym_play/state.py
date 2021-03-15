import numpy as np
import copy

class State():
    "Contains all current trick information."
    def __init__(self, first_player, trump):
        self.first_player = first_player # 0 = N, etc
        self.trump = trump
        self.cards_this_trick = np.zeros(4, dtype=np.int32) - 1 # NESW
        self.cards_played = 0

    def trick_winner(self):
        "Checks the current winner and returns it. Works for partial tricks"
        winner = 0
        best_card = -1
        played_trump = False
        for player, c in enumerate(np.roll(self.cards_this_trick,
                                           -self.first_player)):
            if c == -1: # No further cards played
                break
            if best_card == -1:
                best_card = c
                winner = player
                played_trump = c % 4 == self.trump
            if not played_trump:
                if c % 4 == self.trump: # A trump card was played
                    best_card = c
                    winner = player
                    played_trump = True
                else:
                    if c > best_card and c % 4 == best_card % 4: # Higher card than prev no trump
                        best_card = c
                        winner = player
            else:
                if c % 4 == self.trump and c > best_card:
                    best_card = c
                    winner = player
        return (self.first_player + winner) % 4

    def future_trick_winner(self, player, a):
        "Checks who would win the trick if played played a."
        temp = copy.copy(self.cards_this_trick)
        self.cards_this_trick[player] = a
        future_winner = self.trick_winner()
        self.cards_this_trick = temp
        return future_winner

    def reset(self, first_player):
        self.cards_this_trick = np.zeros(4, dtype=np.int32) - 1
        self.cards_played = 0
        self.first_player = first_player
