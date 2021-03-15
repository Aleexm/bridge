from gym_bridge.analyze import deserialize_bids, check_valid_bid_sequence
import gym
from gym_bridge.agents import RandomAgent, PassingAgent, DoublingRedoublingAgent
from gym_bridge.controller import Controller

def test_valid_played_games():
    """Tests whether the final obtained histories from simulation are valid."""
    env = gym.make('gym_bridge:bridge-v0')
    env.seed(0)
    controller = Controller(agents=[RandomAgent(env.action_space.n,
                                                env.observation_space.n),
                                    PassingAgent(env.action_space.n,
                                                 env.observation_space.n),
                                    DoublingRedoublingAgent(env.action_space.n,
                                                            env.observation_space.n),
                                    RandomAgent(env.action_space.n,
                                                env.observation_space.n)])
    episode_count = 10000
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            valid_actions = env.get_valid_actions()
            action = controller.act(env.acting_player, env.initial_player,
                                    ob, valid_actions)
            ob, _, done, _ = env.step(action)
            if done:
                assert check_valid_bid_sequence(deserialize_bids(ob.history))
                break
    env.close()
