import gym
import numpy as np
from gym_bridge.controller import Controller
from gym_bridge.a2c_keras import ActorCriticAgent
from gym_bridge.agents import RandomAgent
import gym_bridge.analyze as analyze
import gym_bridge.simulation_funcs as sim_fnc
import sys
import time

NUM_ITERATIONS = 10000
DUPLICATES = 2
RESHAPE_US = True
RESHAPE_OPP = True
RESHAPE_ENV_US = True
RESHAPE_ENV_OPP = True
PV_US = True
PV_OPP = False

IMPs = list()
env = gym.make('gym_bridge:bridge-v0')
env.local = True
obs_space_us = 267 if RESHAPE_US else env.observation_space.n
obs_space_opp = 267 if RESHAPE_OPP else env.observation_space.n

a2c_us = ActorCriticAgent(env.action_space.n,
                          obs_space_us,
                          fc_params=200,
                          RESHAPE_OBS=RESHAPE_US,
                          mode='max')
a2c_opp = ActorCriticAgent(env.action_space.n,
                           obs_space_opp,
                           fc_params=200,
                           RESHAPE_OBS=RESHAPE_OPP,
                           mode='max')
filename_us = sim_fnc.get_filename(a2c_us,
                                   DUPLICATES=DUPLICATES,
                                   PASS_VALUE=PV_US,
                                   FC_PARAMS=200,
                                   RESHAPE_OBS=RESHAPE_US,
                                   RESHAPE_ENV=RESHAPE_ENV_US)
filename_opp = sim_fnc.get_filename(a2c_opp,
                                   DUPLICATES=DUPLICATES,
                                   PASS_VALUE=PV_OPP,
                                   FC_PARAMS=200,
                                   RESHAPE_OBS=RESHAPE_OPP,
                                   RESHAPE_ENV=RESHAPE_ENV_OPP)
sim_fnc.load(a2c_us.actor, path='models/{}_a'.format(filename_us))
sim_fnc.load(a2c_us.critic, path='models/{}_c'.format(filename_us))
sim_fnc.load(a2c_opp.actor, path='models/{}_a'.format(filename_opp))
sim_fnc.load(a2c_opp.critic, path='models/{}_c'.format(filename_opp))
controller = Controller([a2c_us, a2c_opp, a2c_us, a2c_opp])

for i in range(NUM_ITERATIONS):
    scores = np.zeros(DUPLICATES)
    for duplicate_idx in range(DUPLICATES):
        if duplicate_idx == 0:
            obs = env.reset(from_deal=True)
        elif duplicate_idx == 1:
            obs = env.reset(soft=True)
        while True: # Repeat until bidding phase concludes (3+ passes)
            valid_actions = env.get_valid_actions()
            a = controller.act(env.acting_player, env.initial_player, obs,
                               valid_actions, duplicate_idx)
            obs, _, done, _ = env.step(a) # Reward is computed later

            if done: # Score and save to agents memory
                scores[duplicate_idx] = env.handle_scoring(duplicate_idx)

                if duplicate_idx == 1:
                    IMP_difference = analyze.IMP_difference(
                                        scores[0], scores[1])
                    IMPs.append(IMP_difference)
                elif duplicate_idx == DUPLICATES-1 == 0:
                    IMP_difference = analyze.IMP_difference(
                                        scores[0], 0)
                    IMPs.append(IMP_difference)
                break

    if i % 1000 == 0 and i > 0:
        print("=======ITERATION {} CONCLUDED======".format(i))
        print("IMPDIFF ", np.mean(IMPs), np.std(IMPs))
        print('SCORES ', scores)
        print('DECLARER ', env.state.declarer)
        print(analyze.deserialize_bids(env.state.history))
        print(env.state.dds)

env.close()
sys.exit()
