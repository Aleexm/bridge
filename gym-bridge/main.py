mport gym
from gym_bridge.controller import Controller
from gym_bridge.agents import RandomAgent, PassingAgent, DoublingRedoublingAgent, AceBiddingAgent
from gym_bridge.reinforce_agent import REINFORCEAgent, DummyREINFORCEAgent
from gym_bridge.a2c_keras import ActorCriticAgent, actor_loss, DummyActorCriticAgent
import gym_bridge.analyze as analyze
import gym_bridge.DDS as dds
import numpy as np
import time
import keras
from keras import backend as K
import pickle
import copy
import matplotlib.pyplot as plt
import math
import json

def plot_learning(scores, filename, x=None, window=5):
    plt.figure()
    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
	    running_avg[t] = np.mean(scores[max(0, t-window):(t+1)])
    if x is None:
        x = [i for i in range(N)]
    plt.ylabel('Score')
    plt.xlabel('Game')
    plt.plot(x, running_avg, linewidth=1)
    plt.savefig(filename)
    plt.close()


def print_sim_info(i, env, scores, bids, declarers, contracts, tricks_takens):
    assert len(scores) == 2, "Can only print in duplicate bridge format!"
    print("""iteration {} concluded.\nHands: {}\nVuln: {}\nStarter: {}
          \nBids_0: {}\nDeclarer_0: {}\n\Contract_0: {}\ntricks_taken_0:
          {}\nscore_0: {}\n\nBids_1: {}\nDeclarer_1: {}\nContract_1: {}
          \ntricks_taken_1: {}\nscore_1: {}\nIMP: {}""".
          format(i,
                 analyze.deserialize_hands(env.state.hands),
                 env.state.vulnerability,
                 env.initial_player,
                 analyze.deserialize_bids(bids[0]),
                 declarers[0],
                 contracts[0],
                 tricks_takens[0],
                 scores[0],
                 analyze.deserialize_bids(bids[1]),
                 declarers[1],
                 contracts[1],
                 tricks_takens[1],
                 scores[1],
                 IMP
                 ))

def save(model, path):
    model.save_weights('{}_ws.h5'.format(path))
    symbolic_weights = getattr(model.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open('{}_optimizer.pkl'.format(path), 'wb') as f:
        pickle.dump(weight_values, f)

def load(model, path, weights_only=False):
    try:
        model.load_weights('{}_ws.h5'.format(path))
        model._make_train_function()
        if not weights_only:
            with open('{}_optimizer.pkl'.format(path), 'rb') as f:
                weight_values = pickle.load(f)
            model.optimizer.set_weights(weight_values)
    except:
        print("=== NO MODEL LOADED ===")

def save_pass_reward(env, agent, last_pass_idx, last_pass_value,
                     scores, duplicate_idx, dds_table):

    declarers = env.state.get_all_our_declarers(env.initial_player, duplicate_idx)
    max_score, max_entry = analyze.compute_max_score(dds_table, declarers,
                                                     env.state.vulnerability)
    score = scores[duplicate_idx]

    if env.state.contract is None:
        achievable = True
        pass_value = analyze.IMP_difference(score, max_score)/24

    else:
        max_contract = (dds_table[max_entry[0]][max_entry[1]] - 7) * 5 + max_entry[1]
        achievable = True if env.state.contract.bid < max_contract else False

        if env.state.declarer % 2 == duplicate_idx: # We made the contract
            # if max_score > 0 and score > 0 and -1 <= analyze.IMP_difference(score, max_score) <= 0: # Close enough
            if int(score) == int(max_score): # We did the best we could
                pass_value = analyze.IMP_difference(np.abs(score), 0)/24
            elif score < 0 and not achievable: # We simply bid too high
                pass_value = 0.3
            else: # We could have done better
                pass_value = analyze.IMP_difference(score, max_score)/24
        else: # Opp made the contract
            if achievable: # We could have done better
                pass_value = analyze.IMP_difference(score, max_score)/24
            else:
                pass_value = 0.3
    if duplicate_idx < len(scores)-1:
        return len(agent.m_actions[-1]) - 1, pass_value
    else:
        agent.save_pass_reward([last_pass_idx, len(agent.m_actions[-1]) -1],
                               [last_pass_value, pass_value])
        return None, None

def compute_and_save_IMP(agent, scores, duplicate_flag, duplicate_idx):
    if not duplicate_flag:
        IMP = analyze.IMP_difference(scores[0], 0)
    elif duplicate_idx == 1:
        IMP = analyze.IMP_difference(scores[0], scores[1])
    else:
        return None
    agent.save_reward(IMP/24)
    return IMP

def compute_and_save_rewards(env, agent, duplicate_idx, dds_table):
    env.state.set_contract_declarer(env.initial_player)
    # if env.state.contract is None:
        # agent.remove_last_trajectory()
        # return True, dds_table, None
    if dds_table is None:
        dds_table = dds.calc_table_PBN(env.state.hands)

    if env.state.contract is None:
        tricks_taken = score = 0
    else:
        tricks_taken = dds_table[env.state.declarer][env.state.contract.bid%5]
        score = analyze.score(env.state.contract, env.state.declarer,
                              env.state.vulnerability, tricks_taken)
        if env.state.declarer % 2 != duplicate_idx: # Rewards are from our perspective
            score *= -1

    return dds_table, score

def load_deal_dds():
    path = os.path.join('gym_bridge', 'data')
    dir = np.random.choice(os.listdir(path))
    file = np.random.choice(os.listdir(os.path.join(path, dir)))
    with open(os.path.join(path, dir, file), 'r') as f:
        data = json.loads(f.read())
        return analyze.serialize_hands(data['hands']), data['dds']

if __name__ == "__main__":

    DUPLICATE_BRIDGE = False
    PRINT = False
    SELFPLAY = False
    TRAIN = True
    LOAD = True
    SAVE = False
    DOUBLE = True
    SAMPLE_DEAL = True
    NUM_EPISODES = 1000000 if TRAIN else 2000

    env = gym.make('gym_bridge:bridge-v0')
    env.seed(0)

    lrs = [1e-3]
    fcs = [200]
    batches = [100]
    modes = ['multinomial']

    for lr in lrs:
        for fc in fcs:
            for batch in batches:
                for mode in modes:

                    agent = ActorCriticAgent(env.action_space.n, env.observation_space.n,
                                           critic_lr=lr, actor_lr=lr, fc_params=fc,
                                           batch_size=batch, mode=mode)
                    dummy_a2c = DummyActorCriticAgent(env.action_space.n,
                                                      env.observation_space.n,
                                                      mode='multinomial')

                    if DOUBLE and SELFPLAY and TRAIN:
                        n_agent = s_agent = agent
                        e_agent = w_agent = dummy_a2c
                    elif DOUBLE and SELFPLAY: # Evaluate against random (?)
                        n_agent = s_agent = agent
                        e_agent = w_agent = RandomAgent(env.action_space.n,
                                                        env.observation_space.n)
                        n_agent.mode = 'max'
                        s_agent.mode = 'max'
                    elif DOUBLE: # Opponents always pass
                        n_agent = s_agent = agent
                        e_agent = w_agent = PassingAgent(env.action_space.n,
                                                         env.observation_space.n)
                    else: # Singla agent, others pass.
                        n_agent = agent
                        e_agent = s_agent = w_agent = PassingAgent(env.action_space.n,
                                                                   env.observation_space.n)
                    controller = Controller([n_agent, e_agent, s_agent, w_agent])

                    info = "{}_{}_{}_{}_{}".format(controller.agents[0],
                                                   lr, fc, batch, mode)
                    if DOUBLE:
                        filename = "d_{}".format(info)
                    if SELFPLAY:
                        filename = "s_{}".format(info)
                    if not DOUBLE and not SELFPLAY:
                        filename = info
                    if LOAD:
                        load(agent.actor, 'models/{}_a'.format(filename))
                        load(agent.critic, 'models/{}_c'.format(filename))

                    repeats = 2 if DUPLICATE_BRIDGE else 1
                    if not TRAIN and not SAVE: # Just simulate
                        agent.mode='max'
                    IMPs = []
                    done = False


                    for i in range(NUM_EPISODES):
                        max_score = dds_table = last_pass_idx = last_pass_value = None
                        no_contract = False
                        if i % 1000 == 0:
                            print('====== FINISHED ITERATION {} ======'.format(i))
                        if SELFPLAY and TRAIN and i % 200 == 0:
                            print("Copying model at iteration", i)
                            load(dummy_a2c.actor, 'models/{}_a'.format(filename), weights_only=True)
                            load(dummy_a2c.critic, 'models/{}_c'.format(filename), weights_only=True)
                        # bids, declarers, contracts, tricks_takens = [], [], [], []
                        scores = np.zeros(repeats)
                        for duplicate_idx in range(repeats): # two sims if DUPLICATE_BRIDGE
                            # if no_contract:
                            #     continue
                            obs = env.reset() if duplicate_idx == 0 else env.reset(soft=True)
                            while True: # Until termination
                                valid_actions = env.get_valid_actions()
                                action = controller.act(env.acting_player, obs,
                                                         valid_actions, duplicate_idx)
                                obs, _, done, _ = env.step(action)
                                if done:
                                    # bids.append(env.state.history)
                                    dds_table, score = compute_and_save_rewards(
                                        env, agent, duplicate_idx, dds_table)

                                    scores[duplicate_idx] = score
                                    IMP = compute_and_save_IMP(agent, scores,
                                                               DUPLICATE_BRIDGE,
                                                               duplicate_idx)
                                    if IMP is not None:
                                        IMPs.append(IMP)

                                    last_pass_idx, last_pass_value = save_pass_reward(
                                          env, agent, last_pass_idx, last_pass_value,
                                          scores, duplicate_idx, dds_table)

                                    # declarers.append(env.state.declarer)
                                    # contracts.append(env.state.contract)
                                    # tricks_takens.append(tricks_taken)
                                    break
                        if PRINT:
                            print_sim_info(i, env, scores, bids, declarers, contracts, tricks_takens)
                        if i % 100 == 0:
                            print(np.mean(IMPs), np.std(IMPs))
                            print(analyze.deserialize_bids(env.state.history))
                            print(dds_table)
                            if i > 0:
                                print('pass_v: ', agent.m_rewards[-1][-1])
                        if TRAIN:
                            agent.train()
                            if i % 200 == 0 and SAVE:
                                plot_learning(IMPs, filename='plots/{}.png'.format(filename), window=100)
                                save(agent.actor, path='models/{}_a'.format(filename))
                                save(agent.critic, path='models/{}_c'.format(filename))

                    if SAVE and TRAIN:
                        save(agent.actor, path='models/{}_a'.format(filename))
                        save(agent.critic, path='models/{}_c'.format(filename))
                        plot_learning(IMPs, filename='plots/{}.png'.format(filename), window=100)
                    print(np.mean(IMPs), np.std(IMPs))
                    print(agent.m_rewards)
                    env.close()
