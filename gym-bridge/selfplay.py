import gym
import numpy as np
from gym_bridge.controller import Controller
from gym_bridge.a2c_keras import ActorCriticAgent
from gym_bridge.agents import RandomAgent
import gym_bridge.analyze as analyze
import gym_bridge.simulation_funcs as sim_fnc
import sys
import time
import random
from gym_bridge.DDS import calc_table_PBN

random.seed(2)
np.random.seed(2)
TRAIN = True
LOAD = True
SAVE = True
NUM_ITERATIONS = 10000000
FC_PARAMS = 200
DUPLICATES = 2
PASS_VALUE = True
RESHAPE_OBS = True
RESHAPE_ENV = True
OLD_DOUBLE = False

IMPs = list()
env = gym.make('gym_bridge:bridge-v0')
obs_space = 267 if RESHAPE_OBS else env.observation_space.n

a2c_us = ActorCriticAgent(env.action_space.n, obs_space,
                          fc_params=FC_PARAMS, RESHAPE_OBS=RESHAPE_OBS)
filename = sim_fnc.get_filename(a2c_us, DUPLICATES, PASS_VALUE, FC_PARAMS,
RESHAPE_OBS, RESHAPE_ENV)
print(filename)
if OLD_DOUBLE:
    filename = 'd_a2c_0.001_200_100_multinomial'

a2c_opp = ActorCriticAgent(env.action_space.n, obs_space,
                           fc_params=FC_PARAMS, RESHAPE_OBS=RESHAPE_OBS)
if LOAD:
    sim_fnc.load(a2c_us.actor, path='models/{}_a'.format(filename))
    sim_fnc.load(a2c_us.critic, path='models/{}_c'.format(filename))
controller = Controller([a2c_us, a2c_opp, a2c_us, a2c_opp])

for i in range(NUM_ITERATIONS):
    nan = False
    scores, pass_values = np.zeros(DUPLICATES), np.zeros(DUPLICATES)
    last_pass_idx = np.zeros(DUPLICATES, dtype=np.int32)
    for duplicate_idx in range(DUPLICATES):
        if nan:
            continue
        if duplicate_idx == 0:
            obs = env.reset(from_deal=False)
        elif duplicate_idx == 1:
            obs = env.reset(soft=True)
        while True: # Repeat until bidding phase concludes (3+ passes)
            valid_actions = env.get_valid_actions()
            a = controller.act(env.acting_player, env.initial_player, obs,
                               valid_actions, duplicate_idx)
            if a == -1: # Probs contain NaN
                print("NAN")
                nan = True
                break
            obs, _, done, _ = env.step(a) # Reward is computed later

            if done: # Score and save to agents memory
                dds_result = calc_table_PBN(env.state.hands)
                scores[duplicate_idx] = env.handle_scoring(duplicate_idx, dds_result)
                pass_values[duplicate_idx] = env.compute_pass_value(
                                                duplicate_idx, scores, dds_result)
                last_pass_idx[duplicate_idx] = len(a2c_us.m_actions[-1])-1

                if duplicate_idx == 1:
                    IMP_difference = analyze.IMP_difference(
                                        scores[0], scores[1])
                    a2c_us.save_reward(IMP_difference / 24)
                    if PASS_VALUE:
                        a2c_us.save_pass_reward(last_pass_idx, pass_values)
                    IMPs.append(IMP_difference)
                elif duplicate_idx == DUPLICATES-1 == 0:
                    IMP_difference = analyze.IMP_difference(
                                        scores[0], 0)
                    a2c_us.save_reward(IMP_difference / 24)
                    if PASS_VALUE:
                        a2c_us.save_pass_reward(last_pass_idx, pass_values)
                    IMPs.append(IMP_difference)
                break

    if i % 100 == 0:
        print("=======ITERATION {} CONCLUDED======".format(i))
        print('SCORES ', scores)
        print('IMP ', IMP_difference)
        print('DECLARER ', env.state.declarer)
        sim_fnc.print_scores_bidding(IMPs, env.state, env.state.dds, a2c_us)
    if TRAIN:
        a2c_us.train()
        if i % 200 == 0 and SAVE:
            sim_fnc.plot_learning(IMPs, filename='plots/{}.png'.
                                  format(filename), window=100)
            sim_fnc.save(a2c_us.actor, path='models/{}_a'.format(filename))
            sim_fnc.save(a2c_us.critic, path='models/{}_c'.format(filename))
            sim_fnc.load(a2c_opp.actor, path='models/{}_a'.format(filename))
            sim_fnc.load(a2c_opp.critic, path='models/{}_c'.format(filename))

env.close()
sys.exit()
