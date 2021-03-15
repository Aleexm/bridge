import gym_bridge.analyze as analyze
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras import backend as K
import pickle

def to_pbn(env, filename):
    PLAYERS = {0:'N', 1:'E', 2:'S', 3:'W'}
    if np.array_equal(env.state.vulnerability, np.array([0,0])):
        vuln = "None"
    elif np.array_equal(env.state.vulnerability, np.array([1,0])):
        vuln = "NS"
    elif np.array_equal(env.state.vulnerability, np.array([0,1])):
        vuln = "EW"
    elif np.array_equal(env.state.vulnerability, np.array([1,1])):
        vuln = "All"
    deal = analyze.deserialize_hands(env.state.hands)
    with open(filename, 'w') as f:
        f.write('[Dealer \"{}\"]\n'.format(PLAYERS[env.initial_player]))
        f.write('[Vulnerable \"{}\"]\n'.format(vuln))
        f.write('[Deal \"N:{}\"]'.format(deal))

def write_result(env, score, dds, filename, IMP=None):
    if IMP is None:
        mode = 'w'
    else:
        mode = 'a'
    with open(filename, mode) as f:
        f.write('{}\n'.format(str(env.initial_player)))
        f.write(str(env.state.declarer))
        f.write("\n")
        f.write(str(analyze.deserialize_bids(env.state.history)))
        f.write("\n")
        f.write(str(score))
        f.write("\n")
        f.write(str(dds))
        f.write("\n")
        if IMP is not None:
            f.write(str(IMP))

def print_scores_bidding(IMPs, state, dds, agent):
    print('Scores :', np.mean(IMPs), np.std(IMPs))
    print(analyze.deserialize_bids(state.history))
    print(dds)
    print('pass_v: ', agent.m_rewards[-1][-1])

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

def save(model, path):
    model.save_weights('{}_ws.h5'.format(path))
    symbolic_weights = getattr(model.optimizer, 'weights')
    weight_values = K.batch_get_value(symbolic_weights)
    with open('{}_optimizer.pkl'.format(path), 'wb') as f:
        pickle.dump(weight_values, f)

def load(model, path, weights_only=False):
    try:
        print(path)
        model.load_weights('{}_ws.h5'.format(path))
        # model._make_train_function()
        if not weights_only:
            with open('{}_optimizer.pkl'.format(path), 'rb') as f:
                weight_values = pickle.load(f)
            model.optimizer.set_weights(weight_values)
    except:
        print("=== NO MODEL LOADED ===")

def get_filename(agent, DUPLICATES, PASS_VALUE, FC_PARAMS, RESHAPE_OBS, RESHAPE_ENV=False):
    filename = 's_{}_{}_{}_{}_{}_{}_{}_{}'.format(
               agent,
               1e-3,
               FC_PARAMS,
               100,
               'multinomial',
               'd{}'.format(DUPLICATES),
               'p{}'.format(int(PASS_VALUE)),
               'r{}'.format(int(RESHAPE_OBS)))
    if RESHAPE_ENV:
        filename += "_e1"

    return filename
