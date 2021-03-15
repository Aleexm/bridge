from gym_play.particles import Particles
from gym_play.action_tree import ActionTree
from gym_bridge.analyze import deserialize_hands
from copy import deepcopy, copy
import numpy as np
from gym_bridge.analyze import deserialize_hands, serialize_hands
import random

class POMCPTree:
    '''
    A POMCPTree represents a node in which we have to take an action. As such,
    it contains the usual visitiation count, action visitation count and action
    values. Moreover, each POMCPTree maintains a belief over possible gamestates
    (i.e. particles) that are consistent at this point.
    '''
    def __init__(self, root, name=None, parent=None):
        self.belief = Particles()
        self.parent = parent
        self.name = name
        self.children = {}
        self.n = 1
        self.na = {}
        self.va = {}
        self.root = root

    def add_child(self, action_to_child, child):
        if action_to_child not in self.children.keys():
            self.children[action_to_child] = child

    def init_actions(self, valid_actions):
        for a in valid_actions:
            self.na[a] = 1 # 1 to avoid annoying division by 0 errors
            self.va[a] = 0

    def child_after_action(self, a, POMCPTree):
        "We move twice, so POMCPTree as child"
        self.add_child(a, POMCPTree)

    def child_after_observation(self, a):
        "Opp moves after us, so ActionTree as child"
        self.add_child(a, ActionTree(parent=self))

    def set_parent(self, parent):
        self.parent = parent

    def update_belief(self, target, gamestate, prev_tree, agent, actions):
        '''
        Updates the belief particles. It does so by randomly dealing the remaining
        cards amongst the opponents, and checking whether the actions they have
        performed are consistent with this deal. If they are, the particle is
        added to the belief. The process is repeated until target particles are
        added.

        args:
            - target(int): number of particles to be added in total.
            - gamestate(PlayEnv): the environment at the CURRENT beliefstate
                                  (i.e. before updating)
            - prev_tree(POMCPTree): the tree rooted at gamestate
            - agent(Agent): the opponent
            - actions(list(int)): The played actions since gamestate
        '''
        print("======UPDATING BELIEF PARTICLES=======")
        opp_played_cards = [gamestate.played_cards[(gamestate.declarer+1)%4],
                            gamestate.played_cards[(gamestate.declarer+3)%4]]
        our_hands = [gamestate.hands[gamestate.declarer], gamestate.hands[(gamestate.declarer+2)%4]]
        remaining_cards = self.belief.filter_known_cards(our_hands, gamestate.played_cards)
        if len(np.nonzero(gamestate.played_cards[(gamestate.declarer+1)%4])[0]) > \
           len(np.nonzero(gamestate.played_cards[(gamestate.declarer+3)%4])[0]):
            last_opp = (gamestate.declarer+1)%4
        else:
            last_opp = (gamestate.declarer+3)%4

        if len(np.nonzero(gamestate.played_cards[(gamestate.declarer+1)%4])[0]) > 0 \
        and len(actions) == 1 and gamestate.active_player % 2 == gamestate.declarer%2:
            # We have won previous trick, and immediately play again: no additional
            # information.
            self._transfer_belief(prev_tree, gamestate, actions)
            target = 0
        tries = 0 # If above threshold (30k), simply adds random beliefs. This
        # is to avoid getting stuck for niche actions/deals.
        while target > 0:
            tries+=1
            env = deepcopy(gamestate)
            env.hands = self.belief.construct_deal(env.declarer, our_hands,
                                               remaining_cards, opp_played_cards,
                                               last_opp)
            for i in range(len(actions)):
                if env.active_player % 2 == env.declarer % 2:
                    env.step(actions[i])
                    continue
                else:
                    npc_a = agent.act(env.hands, env.state.first_player,
                                      env.active_player, env.state, env.declarer)
                    if not npc_a == actions[i] and tries < 30000:
                        break
                    env.step(npc_a)
                self.belief.add_belief(deserialize_hands(env.hands))
                target -= 1
        self.belief.normalize_particles()
        if tries > 30000:
            print("Action largely inconsistent with internal opp_model/particles")

    def _transfer_belief(self, prev_tree, gamestate, actions):
        "Simply transfer beliefs over, except for removing our last played card."
        assert len(actions)==1
        for k, v in prev_tree.belief.particles.items():
            serialized_hands = serialize_hands(k)
            serialized_hands[gamestate.active_player, actions[0]] = 0
            new_deal = deserialize_hands(serialized_hands)
            self.belief.particles[new_deal] = v
