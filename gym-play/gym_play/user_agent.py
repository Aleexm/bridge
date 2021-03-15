from gym_bridge.agents import Agent
from gym_bridge.analyze import deserialize_hand, deserialize_hands
import numpy as np

class UserAgent(Agent):
    """ User input. """
    def __init__(self, action_space, observation_space):
        super().__init__(action_space, observation_space)

    def act(self, hands, first_player, active_player, state, declarer):
        PLAYERS = ['N', 'E', 'S', 'W']
        SUITMAP = { 'C':0, 'c':0, 'D':1, 'd':1, 'H':2, 'h':2, 'S':3, 's':3 }
        CARDMAP = { '2':0, '3':1, '4':2, '5':3, '6':4, '7':5, '8':6, '9':7,
                    'T':8, 't':8, '10':8, 'J':9, 'j':9, 'Q':10, 'q':10, 'K': 11,
                    'k': 11, 'A':12, 'a':12}
        valid_action = None
        if len(np.nonzero(hands[(declarer+1)%4])[0]) < 13:
            print('Dummy: {}'.format(deserialize_hand(hands[(declarer+2)%4])))
        while valid_action is None:
            action = input("Player {} choose your Action: ([cdhs][2-A]): ".format(PLAYERS[active_player]))
            a = None
            try:
                a = CARDMAP[action[1:]] * 4 + SUITMAP[action[0]]
            except:
                print("Invalid action selected.")
                continue
            if hands[active_player][a] == 1:
                valid_action = a
            else:
                print("Hand does not contain that card")
        return valid_action
