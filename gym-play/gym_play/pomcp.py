from gym_play.envs.play_env import PlayEnv
import numpy as np
from gym_play.pomcp_tree import POMCPTree
from gym_play.action_tree import ActionTree
from gym_bridge.analyze import serialize_hands, deserialize_hands, score, IMP_difference
from gym_play.particles import Particles
import random
from gym_play.bond_agent import BondAgent
from gym_play.user_agent import UserAgent
from copy import deepcopy
import math
from tqdm import tqdm
import sys
import gym
import time
import collections
from gym_bridge.state import Contract

def env_from_data(hands, declarer, contract):
    contract = Contract(contract, False, False)
    env = PlayEnv(declarer=declarer, contract=contract, hands=hands)
    return env

def random_env():
    contract = Contract(np.random.choice(np.arange(35, dtype=np.int32)), False, False)
    env = PlayEnv(declarer=np.random.choice(np.arange(4, dtype=np.int32)),
                  contract=contract)
    return env

def POMCP(env, agent, num_sims, num_particles, print_values, c):
    '''
    Overarching POMCP method. Until game completion, it alternates between
    our moves (POMCP) and opponent's moves.
    Each time we are to move, we update the belief particles and set the root
    of the constructed search tree at the current gamestate.

    args:
        - env(PlayEnv): The environment which contains the deal + contract.
        - agent(Agent): The opposition player agent.
        - num_sims(int): Number of POMCP simulations before we take an action.
        - num_particles(int): Number of particles maintained in the belief.
        - print_values(Bool): Display the values of each action after POMCP.
        - c(int): Tradeoff parameter between exploration and exploitation in
                  UCTS. Advised: 24.

    returns:
        - tricks_taken(int): Number of tricks taken by the declaring partnership
        - IMP score(int): score converted to IMP based on contract.
        - belief/search times (int): Time spent respectively searching.
    '''
    belief_time = 0
    search_time = 0
    tree = POMCPTree(root=True, name='b')
    cards = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
    suits = ['C','D','H','S']
    players = ['N', 'E', 'S', 'W']
    inner_agent = BondAgent(52, 52) # Agent used for simulation in POMCP
    true_agent = agent # Agent that they actually play against (can be same)
    done = False
    moves = list() # Moves will contain each move that is played since we last
    # updated the POMCP tree. These are used to check for consistency when updating
    # the beliefs.
    env_before_moves = deepcopy(env)
    first_time = True
    while not done:
        # Declaring team plays using POMCP
        if env.active_player % 2 == env.declarer % 2:
            if first_time:
                pruned_tree = tree
            else:
                pruned_tree, env = prune(env_before_moves, tree, moves)
            start = time.time()
            pruned_tree.update_belief(num_particles, env_before_moves,
                                      tree, inner_agent, moves)
            belief_time += time.time()-start
            moves = list() # Clear list, we have updated.
            env_before_moves = deepcopy(env)
            tree = pruned_tree
            start = time.time()
            a = search(env, inner_agent, tree, num_sims, print_values, c)
            search_time += time.time()-start
            moves.append(a)
            first_time = False
        else: # Opposition plays according to the Bridgebond Agent.
            a =  true_agent.act(env.hands, env.state.first_player, env.active_player,
                           env.state, env.declarer)
            moves.append(a)
        print('{} plays {}{}'.format(players[env.active_player], suits[a%4], cards[a//4] ))
        _, _, done, _ = env.step(a)
        # for a, val in tree.va.items():
        #     print(a, val)
        # for a, val in tree.na.items():
        #     print(a, val)
    return env.tricks_taken[env.declarer], _compute_IMP(env), search_time, belief_time

def prune(pass_env, tree, moves):
    '''
    Places the root of the outdated tree at the current gamestate, based on the
    moves that were played since.

    args:
        - pass_env(PlayEnv): Previous gamestate
        - tree(POMCPTree): Previous tree
        - moves(list(int)): Played moves since pass_env

    returns:
        - pruned_tree(POMCPTree): The POMCPTree rooted at the new gamestate
        - env(PlayEnv): The current gamestate
    '''
    env = deepcopy(pass_env)
    pruned_tree = tree
    for a_idx, a in enumerate(moves):
        if env.active_player%2 == env.declarer%2:
            # There is some chance that childnodes were simply not expanded
            # during POMCP, and we thus have to add them in that case.
            # Note that child_after_... does mothing if child is already a child.
            if env.state.cards_played == 3 \
            and env.state.future_trick_winner(env.active_player, a) % 2 == env.declarer % 2:
                new_tree = POMCPTree(root=False, name=tree.name+'a{}'.format(a))
                pruned_tree.child_after_action(a, new_tree) # We move twice
            else:
                pruned_tree.child_after_observation(a) # Opp moves after
        else:
            if env.state.cards_played == 3 \
            and env.state.future_trick_winner(env.active_player, a) % 2 != env.declarer % 2:
                pruned_tree.child_after_observation(a)
            else:
                new_tree = POMCPTree(root=False, name=tree.name+'a{}o{}'.format(moves[a_idx-1], a))
                pruned_tree.add_o(a, new_tree)
        pruned_tree = pruned_tree.children[a]
        env.step(a)
    pruned_tree.root = True
    return pruned_tree, env

def search(pass_env, agent, tree, num_sims, print_values, c):
    "Enter method for MCTS, call with root"
    print("======SIMULATING {} RUNS======".format(num_sims))
    valid_actions = pass_env.get_valid_actions()
    if len(valid_actions) == 1:
        return valid_actions[0]
    for i in tqdm(range(num_sims)):
        env = deepcopy(pass_env)
        env.hands = tree.belief.sample_deal() # Sample deal from particles
        simulate(env, agent, tree, tree, 0, c) # And perform MCTS simulation
    return _get_greedy_action(tree, print_values)

def simulate(env, agent, tree, parent_tree, depth, c):
    '''
    Main simulaton loop: Checks whether this tree is part of the tree.
    Adds it if not, and performs rollout. If this tree is inside the tree,
    it selects the best action according to UCT and recurses.

    Due to the irregularity in turn-taking, we consider POMCPTree and ActionTrees.
    POMCPTrees are 'our' nodes, wherein we must make a move. Subsequently,
    its children could either be another POMCPTree if we are to move again
    (after winning the trick), or an ActionTree if the opponent is to move.
    The same holds for the children of an ActionTree.
    See https://papers.nips.cc/paper/2010/file/edfbe1afcf9246bb0d40eb4d8027d90f-Paper.pdf
    for details.

    args:
        - env(Env): The environment in which to act.
        - agent(Agent): The opponent (must implement .act())
        - tree(POMCPTree): The current tree to consider
        - parent_tree(POMCPTree): The parent
        - depth(int): the number of recursions (not really used)
        - c(int): exploration / exploitation tradeoff parameter in UCT.

    returns:
        - Number of tricks taken by the declaring partnership.
    '''
    assert isinstance(tree, POMCPTree)
    if env.check_done():
        return _compute_IMP(env)
    if tree.parent is None: # New node
        tree.init_actions(env.get_valid_actions())
        tree.set_parent(parent_tree)
        if not tree.root:
            return rollout(env, tree, depth)
    else:
        if not tree.root:
            tree.belief.add_belief(deserialize_hands(env.hands))
        a = _get_uct_action(tree, c)
        if env.active_player == (env.state.first_player-1) % 4 and \
        env.state.future_trick_winner(env.active_player, a) % 2 == env.declarer % 2:
            r = we_move_twice(a, env, agent, tree, depth, c) # We win and open next
        else: # Current trick continues or we lost
            r = opponent_moves_after_us(a, env, agent, tree, depth, c)
        _update_statistics(tree, a, r)
        return r

def we_move_twice(a, env, agent, tree, depth, c):
    'We conclude this trick and open the next'
    env.step(a)
    new_tree = POMCPTree(root=False, name=tree.name+'a{}'.format(a))
    tree.child_after_action(a, new_tree) # So the son is a POMCPTree
    return simulate(env, agent, tree.children[a], tree, depth+1, c)

def opponent_moves_after_us(a, env, agent, tree, depth, c):
    "We do not end the trick OR we end the trick but we lose"
    env.step(a) # Perform our move
    if env.check_done():
        return _compute_IMP(env)
    tree.child_after_observation(a) # So the son is an ActionTree (branch on opp move)
    npc_a = agent.act(env.hands, env.state.first_player, env.active_player,
                      env.state, env.declarer)
    if env.state.cards_played == 3 \
    and env.state.future_trick_winner(env.active_player, npc_a) % 2 != env.declarer % 2:
        opp_moves_again = True # Opp will end AND win the current trick (they move twice)
    else:
        opp_moves_again = False
    env.step(npc_a) # Perform opps move

    if opp_moves_again: # We played the 3rd card, opp played the 4th and we lost
        if env.check_done():
            return _compute_IMP(env)
        tree.children[a].child_after_observation(npc_a) # So son's son is ActionTree
        npc_a_2 = agent.act(env.hands, env.state.first_player, env.active_player,
                            env.state, env.declarer)
        env.step(npc_a_2)
        new_tree = POMCPTree(root=False, name=tree.name+'a{}o{}o{}'.format(a,npc_a,npc_a_2))
        tree.children[a].children[npc_a].add_o(npc_a_2, new_tree)
        return simulate(env, agent, tree.children[a].children[npc_a].children[npc_a_2], tree, depth+1, c)
    else: # We are to move again after our move + opp's move
        new_tree = POMCPTree(root=False, name=tree.name+'a{}o{}'.format(a,npc_a))
        tree.children[a].add_o(npc_a, new_tree)
        return simulate(env, agent, tree.children[a].children[npc_a], tree, depth+1, c)

def _update_statistics(tree, a, r):
    "Increments counts and appends encountered state to the belief."
    tree.n += 1
    tree.na[a] += 1
    tree.va[a] = tree.va[a] + (r-tree.va[a])/tree.na[a]
    tree.ra[a] += r

def _get_uct_action(tree, c):
    "Implements upper confidence bound action selection"
    best_a = -1
    best_value = -sys.maxsize-1
    for a in tree.va.keys():
        a_val = tree.va[a] + c * math.sqrt(math.log(tree.n)/(tree.na[a]))
        if a_val > best_value:
            best_value = a_val
            best_a = a
    return best_a

def rollout(env, tree, depth):
    "Rolls out randomly"
    assert not env.check_done()
    done = False
    while not done:
        _, _, done, _ = env.step(random.choice(env.get_valid_actions()))
    return _compute_IMP(env)

def _compute_IMP(env):
    tricks_taken = env.tricks_taken[env.declarer]
    num_score = score(env.contract, env.declarer, [0,0], tricks_taken)
    return IMP_difference(num_score, 0)

def _get_greedy_action(tree, print_scores):
    "Returns best action upon MCTS completion"
    best_v = -sys.maxsize - 1; best_action = -1
    for k,v in tree.va.items():
        if print_scores:
                cards = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
                suits = ['C','D','H','S']
                print("Estimated tricks for {}{}: {}".format(suits[k%4], cards[k//4], tree.ra[k]/(tree.na[k])))
        if v > best_v:
            best_v = v
            best_action = k
    return best_action
