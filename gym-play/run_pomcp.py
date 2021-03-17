import numpy as np
from gym_play.envs.play_env import PlayEnv
from gym_bridge.analyze import serialize_hands, deserialize_hands
from gym_play.bond_agent import BondAgent
from gym_play.user_agent import UserAgent
import gym
import os
import gym_play.pomcp as pomcp
import argparse
import collections

'''
This file runs POMCP on a single deal. This deal can be randomly generated, or
loaded in from data. See -h for details when executing.
    params: --mode      : interactive or automatic
            --sims      : number of simulations for POMCP
            --particles : number of particles for POMCP
            --file      : name of speciifc .pbn deal to be loaded
    if left empty, will run with default settings:
            automatic 5000 100 random_deal
'''

def parse_pbn(file):
    "Extracts the hands, declarer, contract and tricks taken from single pbn"
    PLAYERS = {0:'N', 1:'E', 2:'S', 3:'W'}
    REV_PLAYERS = {'N':0, 'E':1, 'S':2, 'W':3}
    SUITS = {'C':0, 'D':1, 'H':2, 'S':3, 'N': 4}
    with open(os.path.join('data', file), 'r') as f:
        lines = f.readlines()
        for line in lines:
            if "Deal " in line:
                try:
                    hands = line
                    # Hands do not always start at North, so rotate accordingly.
                    first_player = hands[7]
                    hands = hands[9:-3]
                    hands = hands.split(' ')
                    hands = collections.deque(hands)
                    if first_player == 'E':
                        to_rotate = 1
                    elif first_player == 'S':
                        to_rotate = 2
                    elif first_player == 'W':
                        to_rotate = 3
                    else:
                        to_rotate = 0
                    hands.rotate(to_rotate)
                    hands = ' '.join([str(elem) for elem in hands])
                except:
                    print("Something went wrong reading the deal.")
                    print("Deal found: {}".format(line))
            elif "Declarer" in line:
                try:
                    declarer = REV_PLAYERS[line[11]]
                except: # There were 4 initial passes
                    print("There was no declarer/contract: 4 initial passes.")
                    return None, None, None, None
            elif "Contract" in line:
                contract = line[11:13]
                final_contract = int(contract[0]) * 5 + SUITS[contract[1]]
            elif "Result" in line:
                result = line[9:11]
                try:
                    res = int(result)
                except:
                    res = int(result[0])
        return hands, declarer, final_contract, res

def single_pomcp(agent, num_sims, num_particles, file, c):
    if file is not None:
        hands, declarer, final_contract, res = parse_pbn(file)
        if declarer is None:
            print("4 passes, no contract established.")
            return
        serialized_hands = serialize_hands(hands)
        env = pomcp.env_from_data(serialized_hands, declarer, final_contract)
    else:
        env = pomcp.random_env()
    suits = ['C', 'D', 'H', 'S', 'N']
    players = ['N', 'E', 'S', 'W']
    print('Contract: {}{} declared by {}'.format(env.contract.bid//5,
                                                suits[env.contract.bid%5],
                                                players[env.declarer]))
    return pomcp.POMCP(env, agent, num_sims, num_particles, c)

parser = argparse.ArgumentParser(description='POMCP params')
parser.add_argument("--mode", help='mode: interactive / automatic', type=str)
parser.add_argument("--sims", help="number of simulations for pomcp", type=int)
parser.add_argument("--particles", help="number of particles for pomcp", type=int)
parser.add_argument("--file", help="Specific .pbn deal, i.e. deal_1.pbn. This file must be in /data/... Random deal if left empty", type=str)
args = parser.parse_args()
if args.mode == "automatic" or args.mode is None:
    agent = BondAgent(52,52)
elif args.mode == "interactive":
    agent = UserAgent(52,52)
else:
    print("Incorrect mode submitted: must be interactive or automatic")
if args.sims is not None:
    sims = args.sims
    assert sims > 0, "Number of simulations must be positive"
else:
    sims = 5000
if args.particles is not None:
    particles = args.particles
    assert particles > 0, "Number of particles must be positive"
else:
    particles = 100
tricks, score, search_time, belief_time  = single_pomcp(agent, sims, particles,
                                                        args.file, 24)
print('Score (IMP): {} with {} tricks taken'.format(score, tricks))
