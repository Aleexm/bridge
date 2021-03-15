import gym
import numpy as np
import os
from gym_bridge.controller import Controller
from gym_bridge.a2c_keras import ActorCriticAgent
from gym_bridge.agents import UserAgent
import gym_bridge.analyze as analyze
import gym_bridge.simulation_funcs as sim_fnc
import sys
import time
import gym_bridge.DDS as dds

PLAYERS = {0:'N', 1:'E', 2:'S', 3:'W'}
REV_PLAYERS = {'N':0, 'E':1, 'S':2, 'W':3}
ACTION_MEANING = {
    0: "1C", 1: "1D", 2: "1H", 3: "1S", 4: "1N",
    5: "2C", 6: "2D", 7: "2H", 8: "2S", 9: "2N",
    10: "3C", 11: "3D", 12: "3H", 13: "3S", 14: "3N",
    15: "4C", 16: "4D", 17: "4H", 18: "4S", 19: "4N",
    20: "5C", 21: "5D", 22: "5H", 23: "5S", 24: "5N",
    25: "6C", 26: "6D", 27: "6H", 28: "6S", 29: "6N",
    30: "7C", 31: "7D", 32: "7H", 33: "7S", 34: "7N",
    35: "pass", 36: "double", 37: "redouble"
}

def parse_pbn(file):
    with open(file, 'r') as f:
        dealer = f.readline()[9]
        vuln = f.readline()[13:-3]
        hands = f.readline()[9:-2]
        read = True
        if vuln == "None":
            vuln = np.array([0,0])
        elif vuln == "NS":
            vuln = np.array([1,0])
        if vuln == "EW":
            vuln = np.array([0,1])
        if vuln == "All":
            vuln = np.array([1,1])
        encoded_hands = analyze.serialize_hands(hands)
        print('D ', REV_PLAYERS[dealer], 'V ', vuln, 'H ', analyze.deserialize_hands(encoded_hands))
        env.initial_player, env.acting_player = REV_PLAYERS[dealer], REV_PLAYERS[dealer]
        env.state.vulnerability = vuln
        env.state.hands = encoded_hands


DUPLICATES = 2
PASS_VALUE = True
RESHAPE_OBS = True
OLD_DOUBLE = False
RESHAPE_ENV = True


IMPs = list()
env = gym.make('gym_bridge:bridge-v0')
obs_space = 267 if RESHAPE_OBS else env.observation_space.n

a2c_us = ActorCriticAgent(env.action_space.n, obs_space,
                          RESHAPE_OBS=RESHAPE_OBS, mode='max')
user_agent = UserAgent(env.action_space.n, env.observation_space.n)
filename = sim_fnc.get_filename(a2c_us, DUPLICATES, PASS_VALUE, 200,
                                RESHAPE_OBS, RESHAPE_ENV)
if OLD_DOUBLE:
    filename = 'd_a2c_0.001_200_100_multinomial'

sim_fnc.load(a2c_us.actor, path='models/{}_a'.format(filename), weights_only=True)
sim_fnc.load(a2c_us.critic, path='models/{}_c'.format(filename), weights_only=True)
controller = Controller([a2c_us, user_agent, a2c_us, user_agent])

while True:
    scores = np.zeros(DUPLICATES)
    for duplicate_idx in range(DUPLICATES):
        if duplicate_idx == 0:
            obs = env.reset()
            read = False
            while read == False:
                file = input("Choose .pbn: ")
                try:
                    parse_pbn(os.path.join('wbridge_pbn', '{}.pbn'.format(file)))
                    read = True
                except:
                    print('/wbridge_pbn/{}.pbn does not exist!'.format(file)))
                    continue
        elif duplicate_idx == 1:
            obs = env.reset(soft=True)
        while True: # Repeat until bidding phase concludes (3+ passes)
            valid_actions = env.get_valid_actions()
            a = controller.act(env.acting_player, env.initial_player, obs,
                               valid_actions, duplicate_idx)
            print(PLAYERS[env.acting_player], ' BID ', ACTION_MEANING[a])
            obs, _, done, _ = env.step(a) # Reward is computed later

            if done: # Score and save to agents memory
                dds_table = dds.calc_table_PBN(env.state.hands)
                scores[duplicate_idx] = env.handle_scoring(duplicate_idx,
                                                           dds_table)


                if duplicate_idx == 1:
                    IMP_difference = analyze.IMP_difference(
                                        scores[0], scores[1])
                    IMPs.append(IMP_difference)
                    sim_fnc.write_result(env, scores[duplicate_idx], dds_table,
                                         os.path.join('wbridge_res', '{}.txt'.
                                         format(file)),
                                         IMP_difference)
                else:
                    sim_fnc.write_result(env, scores[duplicate_idx], dds_table,
                                         os.path.join('wbridge_res', '{}.txt'.
                                         format(file)))

                break

        print(dds_table)
        print("IMPDIFF ", np.mean(IMPs), np.std(IMPs))
        print('SCORES ', scores)
        print('DECLARER ', env.state.declarer)
        print(analyze.deserialize_bids(env.state.history))

env.close()
sys.exit()
