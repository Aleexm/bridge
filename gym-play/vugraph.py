import gym_bridge.DDS as dds
import numpy as np
from gym_play.envs.play_env import PlayEnv
from gym_play.pomcp_tree import POMCPTree
from gym_bridge.analyze import serialize_hands, deserialize_hands
from gym_play.particles import Particles
import random
from gym_play.bond_agent import BondAgent
from copy import deepcopy
import math
import gym
import collections
import os
import gym_play.pomcp as pomcp
import time
import argparse

"Code for comparing VuGraph (expert) games against DDS and POMCP"

def create_files():
    '''
    Make sure you have a directory called vugraph/ogs with .pbn files.
    Vugraph dumps 30+ games inside such pbns. This function creates a separate
    .pbn file for each game inside those, in directory vugraph/data
    '''
    idx = len(os.listdir(os.path.join("vugraph", "data")))
    files = os.listdir(os.path.join("vugraph", "ogs"))
    for file in files:
        if file.endswith('.pbn'):
            try:
                with open(os.path.join("vugraph", "ogs", file), 'r') as f:
                    lines = f.readlines()
                    for line_idx, line in enumerate(lines):
                        if "Event" in line:
                            with open(os.path.join("vugraph", "data", "{}.pbn".format(idx)), 'w') as f2:
                                while True:
                                    r = lines[line_idx+1]
                                    if "Deal" in r or "Contract" in r or "Declarer" in r:
                                        f2.write(r)
                                    if "Result" in r:
                                        f2.write(r)
                                        idx+=1
                                        break
                                    line_idx+=1
            except:
                print('An error occured parsing {}'.format(file))

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

def get_true_results():
    for file in os.listdir(os.path.join("vugraph", 'data')):
        if file.endswith('.pbn'):
            hands, declarer, final_contract, res = parse_pbn(file)
            if declarer is not None:
                write_to_file("true_results.txt", str(res))

def get_dds():
    for file in os.listdir(os.path.join("vugraph", 'data')):
        if file.endswith('.pbn'):
            hands, declarer, final_contract, res = parse_pbn(file)
            if declarer is None:
                continue
            start = time.time()
            dds_table = dds.calc_table_PBN(serialize_hands(hands))
            end = time.time()
            res = dds_table[declarer][final_contract%5]
            write_to_file("dds_results.txt", str(res))
            write_to_file("dds_times.txt", str(end-start))

def get_pomcp(num_sims, num_particles, c):
    np.random.seed(1)
    random.seed(1)
    for file in os.listdir(os.path.join("vugraph", 'data')):
        if file.endswith('.pbn'):
            hands, declarer, final_contract, res = parse_pbn(file)
            if declarer is None:
                continue
            serialized_hands = serialize_hands(hands)
            env = pomcp.env_from_data(serialized_hands, declarer, final_contract)
            start = time.time()
            tricks, score, search_time, belief_time = pomcp.POMCP(env, BondAgent(52,52), num_sims, num_particles, c)
            end = time.time()
            write_to_file("pomcp_results_{}_{}.txt".format(num_sims, num_particles), str(tricks))
            write_to_file("pomcp_times_{}_{}.txt".format(num_sims, num_particles), str(end-start))
            write_to_file("pomcp_search_times_{}_{}.txt".format(num_sims, num_particles), str(search_time))
            write_to_file("pomcp_belief_times_{}_{}.txt".format(num_sims, num_particles), str(belief_time))
            print('finished {}'.format(file))

def write_to_file(filename, res):
    with open(os.path.join("vugraph", filename), 'a') as f:
        f.write('{}\n'.format(res))

parser = argparse.ArgumentParser(description='Mode')
parser.add_argument("mode", help='mode: true,dds or pomcp', type=str)
parser.add_argument("--sims", help="number of simulations for pomcp", type=int)
parser.add_argument("--particles", help="number of particles for pomcp", type=int)
args = parser.parse_args()
if args.mode == "true":
    get_true_results()
elif args.mode == "dds":
    get_dds()
else:
    assert args.sims is not None and args.particles is not None
    get_pomcp(args.sims, args.particles, 24)
