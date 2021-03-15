from gym_bridge.analyze import serialize_bids, deserialize_bids
from gym_bridge.analyze import check_valid_bid_sequence
from gym_bridge.envs.bridge_env import BridgeEnv
import numpy as np
import gym

def test_valid_moves():
    """Tests whether bridge_env.get_valid_actions returns correct moves."""
    test_h = [
              ['1C', 'p','p','1D', '6N', 'd', 'p', 'p'],
              ['1C', 'p','p','1D', '6N', 'd'],
              ['p','p','p'],
              [],
              ['p','p','4D', 'p','p','d','p','p'],
              ['7N', 'd','r'],
              ['p','p','p','7S'],
              ['p','p','p','7S','p']
    ]
    test_a = [
              np.append(np.arange(30,36), 37),
              np.append(np.arange(30,36), 37),
              np.arange(0,36),
              np.arange(0,36),
              np.append(np.arange(17,36),37),
              np.array([35]),
              np.arange(34,37),
              np.arange(34,36)
    ]
    for i in range(len(test_h)):
        ser = serialize_bids(test_h[i])
        env = gym.make('gym_bridge:bridge-v0')
        env.state.history = ser
        actions = env.get_valid_actions()
        assert np.array_equal(actions, test_a[i])
