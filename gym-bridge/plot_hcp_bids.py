import keras
from keras import backend as K
import pickle
from gym_bridge.simulation_funcs import load
from gym_bridge.a2c_keras import ActorCriticAgent, actor_loss
from gym_bridge.agents import PassingAgent
import matplotlib.pyplot as plt
import gym
import numpy as np
from gym_bridge.controller import Controller
import seaborn as sns

def hcp_hand(hands):
    hcp = 0
    for b in np.nonzero(hands[0])[0]:
        if b >= 36:
            hcp += b // 4 - 8
    # print(hcp, _deserialize(hands)[:16])
    return hcp

def hcp_per_suit(hands):
    hcps = np.zeros(4)
    for b in np.nonzero(hands[0])[0]:
        if b >= 36:
            hcps[b%4] += b // 4 - 8
    return hcps


def get_hcp_suit_idx(hcps):
    if hcps[0] < 2 and hcps[1] < 2 and hcps[2] < 2 and hcps[3] < 2:
        return 0
    if hcps[0] < 7 and hcps[1] < 2 and hcps[2] < 2 and hcps[3] < 2:
        return 1
    if hcps[0] < 2 and hcps[1] < 7 and hcps[2] < 2 and hcps[3] < 2:
        return 2
    if hcps[0] < 2 and hcps[1] < 2 and hcps[2] < 7 and hcps[3] < 2:
        return 3
    if hcps[0] < 2 and hcps[1] < 2 and hcps[2] < 2 and hcps[3] < 7:
        return 4
    if hcps[0] < 7 and hcps[1] < 7 and hcps[2] < 2 and hcps[3] < 2:
        return 5
    if hcps[0] < 7 and hcps[1] < 2 and hcps[2] < 7 and hcps[3] < 2:
        return 6
    if hcps[0] < 7 and hcps[1] < 2 and hcps[2] < 2 and hcps[3] < 7:
        return 7
    if hcps[0] < 2 and hcps[1] < 7 and hcps[2] < 7 and hcps[3] < 2:
        return 8
    if hcps[0] < 2 and hcps[1] < 7 and hcps[2] < 2 and hcps[3] < 7:
        return 9
    if hcps[0] < 2 and hcps[1] < 2 and hcps[2] < 7 and hcps[3] < 7:
        return 10
    if hcps[0] < 7 and hcps[1] < 7 and hcps[2] < 7 and hcps[3] < 2:
        return 11
    if hcps[0] < 7 and hcps[1] < 7 and hcps[2] < 2 and hcps[3] < 7:
        return 12
    if hcps[0] < 7 and hcps[1] < 2 and hcps[2] < 7 and hcps[3] < 7:
        return 13
    if hcps[0] < 2 and hcps[1] < 7 and hcps[2] < 7 and hcps[3] < 7:
        return 14
    if hcps[0] < 7 and hcps[1] < 7 and hcps[2] < 7 and hcps[3] < 7:
        return 15
    if hcps[0] < 11 and hcps[1] < 7 and hcps[2] < 7 and hcps[3] < 7:
        return 16
    if hcps[0] < 7 and hcps[1] < 11 and hcps[2] < 7 and hcps[3] < 7:
        return 17
    if hcps[0] <7 and hcps[1] < 7 and hcps[2] < 11 and hcps[3] < 7:
        return 18
    if hcps[0] < 7 and hcps[1] < 7 and hcps[2] < 7 and hcps[3] < 11:
        return 19
    if hcps[0] < 11 and hcps[1] < 11 and hcps[2] < 7 and hcps[3] < 7:
        return 20
    if hcps[0] < 11 and hcps[1] < 7 and hcps[2] < 11 and hcps[3] < 7:
        return 21
    if hcps[0] < 11 and hcps[1] < 7 and hcps[2] < 7 and hcps[3] < 11:
        return 22
    if hcps[0] < 7 and hcps[1] < 11 and hcps[2] < 11 and hcps[3] < 7:
        return 23
    if hcps[0] < 7 and hcps[1] < 11 and hcps[2] < 7 and hcps[3] < 11:
        return 24
    if hcps[0] < 7 and hcps[1] < 7 and hcps[2] < 11 and hcps[3] < 11:
        return 25
    if hcps[0] < 11 and hcps[1] < 11 and hcps[2] < 11 and hcps[3] < 7:
        return 26
    if hcps[0] < 11 and hcps[1] < 11 and hcps[2] < 7 and hcps[3] < 11:
        return 27
    if hcps[0] < 11 and hcps[1] < 7 and hcps[2] < 11 and hcps[3] < 11:
        return 28
    if hcps[0] < 7 and hcps[1] < 11 and hcps[2] < 11 and hcps[3] < 11:
        return 29
    if hcps[0] < 11 and hcps[1] < 11 and hcps[2] < 11 and hcps[3] < 11:
        return 30

def plot_suit_hcp_bids(NUM_ITERS, env, a2c, passes=0):
    controller = Controller([a2c, a2c, a2c, a2c])
    hcp_bids = np.zeros((31, 36))
    for i in range(NUM_ITERS):
        if i % 1000 == 0:
            print(i)
        o = env.reset()
        env.state.history[0:passes] = 1
        hcps = hcp_per_suit(o.hands)
        sub_o = controller.get_subagent_observation(0,o)
        a = a2c.act(sub_o, np.arange(36))
        hcp_idx = get_hcp_suit_idx(hcps)
        hcp_bids[hcp_idx,a] += 1
    for i in range(31):
        hcp_bids[i] = hcp_bids[i] / np.sum(hcp_bids[i],axis=None) if np.sum(hcp_bids[i],axis=None) > 0 else hcp_bids[i]

    sns.set()
    ax = sns.heatmap(hcp_bids)
    ax.set(xlabel='Bid')
    ax.set(ylabel='High Card Points')
    ax.set_xticks(np.arange(36))
    ax.set_yticks(np.arange(31))
    ax.set_yticklabels([
        '<2\u2663, <2\u2666, <2\u2665, <2\u2660',
        '2-6\u2663, <2\u2666, <2\u2665, <2\u2660',
        '<2\u2663, 2-6\u2666, <2\u2665, <2\u2660',
        '<2\u2663, <2\u2666, 2-6\u2665, <2\u2660',
        '<2\u2663, <2\u2666, <2\u2665, 2-6\u2660',
        '2-6\u2663, 2-6\u2666, <2\u2665, <2\u2660',
        '2-6\u2663, <2\u2666, 2-6\u2665, <2\u2660',
        '2-6\u2663, <2\u2666, <2\u2665, 2-6\u2660',
        '<2\u2663, 2-6\u2666, 2-6\u2665, <2\u2660',
        '<2\u2663, 2-6\u2666, <2\u2665, 2-6\u2660',
        '<2\u2663, <2\u2666, 2-6\u2665, 2-6\u2660',
        '2-6\u2663, 2-6\u2666, 2-6\u2665, <2\u2660',
        '2-6\u2663, 2-6\u2666, <2\u2665, 2-6\u2660',
        '2-6\u2663, <2\u2666, 2-6\u2665, 2-6\u2660',
        '<2\u2663, 2-6\u2666, 2-6\u2665, 2-6\u2660',
        '2-6\u2663, 2-6\u2666, 2-6\u2665, 2-6\u2660',
        '7-10\u2663, <7\u2666, <7\u2665, <7\u2660',
        '<7\u2663, 7-10\u2666, <7\u2665, <7\u2660',
        '<7\u2663, <7\u2666, 7-10\u2665, <7\u2660',
        '<7\u2663, <7\u2666, <7\u2665, 7-10\u2660',
        '7-10\u2663, 7-10\u2666, <7\u2665, <7\u2660',
        '7-10\u2663, <7\u2666, 7-10\u2665, <7\u2660',
        '7-10\u2663, <7\u2666, <7\u2665, 7-10\u2660',
        '<7\u2663, 7-10\u2666, 7-10\u2665, <7\u2660',
        '<7\u2663, 7-10\u2666, <7\u2665, 7-10\u2660',
        '<7\u2663, <7\u2666, 7-10\u2665, 7-10\u2660',
        '7-10\u2663, 7-10\u2666, 7-10\u2665, <7\u2660',
        '7-10\u2663, 7-10\u2666, <7\u2665, 7-10\u2660',
        '7-10\u2663, <7\u2666, 7-10\u2665, 7-10\u2660',
        '<7\u2663, 7-10\u2666, 7-10\u2665, 7-10\u2660',
        '7-10\u2663, 7-10\u2666, 7-10\u2665, 7-10\u2660'
    ], rotation=0)
    ax.set_xticklabels(['1C','1D','1H','1S','1N',
                        '2C','2D','2H','2S','2N',
                        '3C','3D','3H','3S','3N',
                        '4C','4D','4H','4S','4N',
                        '5C','5D','5H','5S','5N',
                        '6C','6D','6H','6S','6N',
                        '7C','7D','7H','7S','7N', 'p'])
    plt.show()

def plot_hcp_bids(NUM_ITERS, env, a2c, passes=0):
    controller = Controller([a2c, a2c, a2c, a2c])
    hcp_bids = np.zeros((38, 36))
    for i in range(NUM_ITERS):
        if i % 1000 == 0:
            print(i)
        o = env.reset()
        # env.state.history[0:passes] = 1
        hcp = hcp_hand(o.hands)
        # sub_o = controller.get_subagent_observation(0,o)
        # a = a2c.act(sub_o, np.arange(36))
        a = controller.act(env.acting_player, env.initial_player, o, env.get_valid_actions())
        hcp_bids[hcp,a] += 1
    for i in range(38):
        hcp_bids[i] = hcp_bids[i] / np.sum(hcp_bids[i],axis=None) if np.sum(hcp_bids[i],axis=None) > 0 else hcp_bids[i]

    sns.set()
    plt.figure(figsize=[6.4*3,4.8*2])
    ax = sns.heatmap(hcp_bids)
    ax.set(xlabel='Bid')
    ax.set(ylabel='High Card Points')
    ax.set_xticks(np.arange(36))
    ax.set_yticks(np.arange(38))
    ax.set_xticklabels(['1\u2663','1\u2666','1\u2665','1\u2660','1N',
                        '2\u2663','2\u2666','2\u2665','2\u2660','2N',
                        '3\u2663','3\u2666','3\u2665','3\u2660','3N',
                        '4\u2663','4\u2666','4\u2665','4\u2660','4N',
                        '5\u2663','5\u2666','5\u2665','5\u2660','5N',
                        '6\u2663','6\u2666','6\u2665','6\u2660','6N',
                        '7\u2663','7\u2666','7\u2665','7\u2660','7N', 'p'])
    ax.set_yticklabels(np.arange(38))
    plt.savefig('plots/hcp',bbox_inches='tight')

def plot_final_contracts(NUM_ITERS, env, a2c):
    controller = Controller([a2c, a2c, a2c, a2c])
    x = np.arange(5)
    y = np.zeros((7,5))
    passes = 0
    for i in range(NUM_ITERS):
        if i % 1000 == 0:
            print(i)
        obs = env.reset()
        while True:
            valid_actions = env.get_valid_actions()
            action = controller.act(env.acting_player, env.initial_player, obs,
                                     valid_actions, 0)
            if action == -1:
                break
            obs, _, done, _ = env.step(action)
            if done:
                env.state.set_contract_declarer(env.initial_player)
                if env.state.contract is not None:
                    # print(env.state.contract.bid)
                    y[env.state.contract.bid//5][env.state.contract.bid%5]+=1
                else:
                    passes+=1
                break
    print(y)
    print(passes)

    colors = ['darkblue', 'blue', 'aqua', 'chartreuse', 'gold', 'red', 'sienna']
    ax = plt.subplot(111)
    for i in range(7):
        ax.bar(x+(i-3)*0.1, y[i][:], width=0.1,
               color=colors[i], align='center', label=str(i+1))
    plt.xticks(np.arange(5), ['\u2663', '\u2666', '\u2665', '\u2660', 'NT'])
    plt.legend()
    plt.savefig("plots/final_contracts")



    # plt.bar(np.arange(35), contracts)
    # plt.xticks(np.arange(35), labels=['1\u2663', '1\u2666', '1\u2665', '1\u2660', '1N',
    #                                   '2\u2663', '2\u2666', '2\u2665', '2\u2660', '2N',
    #                                   '3\u2663', '3\u2666', '3\u2665', '3\u2660', '3N',
    #                                   '4\u2663', '4\u2666', '4\u2665', '4\u2660', '4N',
    #                                   '5\u2663', '5\u2666', '5\u2665', '5\u2660', '5N',
    #                                   '6\u2663', '6\u2666', '6\u2665', '6\u2660', '6N',
    #                                   '7\u2663', '7\u2666', '7\u2665', '7\u2660', '7N'])
    # plt.show()
    # plt.close()

FC_PARAMS = 200
RESHAPE_OBS = True
RESHAPE_ENV = True
obs_space = 267 if RESHAPE_OBS else env.observation_space.n

env = gym.make('gym_bridge:bridge-v0')
a2c = ActorCriticAgent(env.action_space.n, obs_space,
                          fc_params=FC_PARAMS, RESHAPE_OBS=RESHAPE_OBS, mode='max')
filename = 's_a2c_0.001_200_100_multinomial_d2_p1_r1_e1'
load(a2c.actor, 'models/{}_a'.format(filename))
load(a2c.critic, 'models/{}_c'.format(filename))
env.seed(0)
NUM_ITERS = 100000

# plot_suit_hcp_bids(NUM_ITERS, env, controller, a2c, passes=2)
# plot_hcp_bids(NUM_ITERS, env, a2c, passes=0)
# plot_hcp_bids(NUM_ITERS, env, controller, a2c, passes=2)
plot_final_contracts(NUM_ITERS, env, a2c)

# plt.savefig('hcp_bids_heatmap.png')
