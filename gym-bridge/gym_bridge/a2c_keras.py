from gym_bridge.agents import Agent
import numpy as np
import keras
from keras import backend as K
from keras.layers import Input, Dense, Multiply, Activation, Add, Lambda
from keras.models import Model
import tensorflow as tf
import random


def actor_loss(y_true, y_pred):
    '''
    Policy Gradient loss.

    args:
        - y_true: one-hot encoded numpy array containing the advantage
                  at the taken action's entry, and 0's elsewhere.
        - y_pred: logits: softmax'd action probabilities.

    returns:
        - K.sum(loss): loss function
    '''
    y_pred = tf.Print(y_pred, [y_pred], summarize=5)
    action_loss = K.sum(K.log(y_pred + 1e-10) * y_true * -1, axis=1)
    action_loss = K.print_tensor(action_loss, message='a_loss= ')
    entropy_loss = K.sum(K.log(y_pred + 1e-10) * y_pred * -1, axis=1)
    entropy_loss = K.print_tensor(entropy_loss, message='e_loss= ')
    loss = K.sum(K.sum(action_loss) - 0.01 * K.sum(entropy_loss))
    loss = K.print_tensor(loss, message='loss= ')
    return loss

class ActorCriticAgent(Agent):
    '''A2C with Entropy term.'''
    def __init__(self, action_space, observation_space, discount_factor=1,
                 batch_size=100, critic_lr=1e-3, actor_lr=1e-3, fc_params=200,
                 entropy_ratio=0.01, mode='multinomial', actor=None,
                 critic=None, RESHAPE_OBS=False):
        super().__init__(action_space, observation_space)
        self.discount_factor = discount_factor
        self.batch_size = batch_size
        self.entropy_ratio = entropy_ratio
        if actor is None or critic is None:
            self.actor, self.critic = self.build_actor_critic_model(fc_params,
                                            actor_lr, critic_lr)
        else:
            self.actor = actor
            self.critic = critic
        # Memory for trajectories
        self.m_observations, self.m_actions, self.m_rewards = [], [], []
        self.trajectory_done = True
        self.mode = mode # Exploration mode

        self.reshape_obs = RESHAPE_OBS

    def __repr__(self):
        return "a2c"

    def build_actor_critic_model(self, fc_params, actor_lr, critic_lr):
        inputs = Input(shape=(self.observation_space-38,))
        valid_actions = Input(shape=(38,))
        initial_fc = Dense(fc_params, activation='relu')(inputs)
        output_1 = Dense(fc_params, activation='relu')(initial_fc)
        output_2 = Dense(fc_params, activation='relu')(output_1)
        skip_1 = Add()([initial_fc, output_2])
        output_3 = Dense(fc_params, activation='relu')(skip_1)
        output_4 = Dense(fc_params, activation='relu')(output_3)
        skip_2 = Add()([skip_1, output_4])
        action_probs = Dense(units=self.action_space, activation='softmax')(skip_2)
        action_probs_masked = Multiply()([action_probs, valid_actions])
        layer = Lambda(lambda x: x / keras.backend.sum(x, axis=1)[:,None])
        actions = layer(action_probs_masked)

        value = Dense(1, activation='linear')(skip_2)

        actor = Model(inputs=[inputs, valid_actions], outputs=actions)
        actor_optimizer = keras.optimizers.RMSprop(lr=actor_lr)
        actor.compile(optimizer=actor_optimizer, loss=actor_loss)

        critic = Model(inputs=[inputs, valid_actions], outputs=value)
        critic_optimizer = keras.optimizers.RMSprop(lr=critic_lr)
        critic.compile(optimizer=critic_optimizer, loss='mean_squared_error')

        return actor, critic

    def act(self, observation, valid_actions):
        o = observation[:-38]
        valids = observation[-38:]
        observation = np.reshape(o, [1, self.observation_space-38])
        valid_actions = np.reshape(valids, [1, 38])
        logits = self.actor.predict([observation, valid_actions])[0]

        if self.mode == 'max':
            action = np.argmax(logits)
        elif self.mode == 'max_e':
            if random.random() < 0.2:
                return np.random.choice(valid_actions)
            else:
                action = np.argmax(logits)
        elif self.mode == 'multinomial':
            try: # Might contain NaN error
                action = np.random.choice(self.action_space, p=logits)
            except:
                return -1
        return action

    def train(self):
        if len(self.m_observations) < self.batch_size:
            return
        all_rewards = np.concatenate(np.array(self.m_rewards),axis=None)
        all_obs = np.zeros((len(all_rewards), self.observation_space-38))
        all_valid_actions = np.zeros((len(all_rewards), 38))
        all_advantages = np.zeros((len(all_rewards), self.action_space))
        c = 0
        # Memories contain list of trajectories.
        # Each trajectory contains multiple bids, states, etc.
        for i, (game_obs, game_act) in enumerate(zip(self.m_observations, self.m_actions)):
            for j, (o, a) in enumerate(zip(game_obs, game_act)):
                observation = o[:-38]
                valid_actions = o[-38:]
                all_obs[c,:] = observation
                all_valid_actions[c,:] = valid_actions
                critic_prediction = self.critic.predict(
                    [np.reshape(observation, [1,self.observation_space-38]),
                    np.reshape(valid_actions, [1, 38])]).flatten()
                all_advantages[c,a] = all_rewards[c] - critic_prediction
                c+=1
        all_obs = np.reshape(all_obs, [len(all_rewards), self.observation_space-38])
        all_valid_actions = np.reshape(all_valid_actions, [len(all_rewards), 38])
        all_advantages = np.reshape(all_advantages, [len(all_rewards), self.action_space])

        actor_loss = self.actor.fit([all_obs, all_valid_actions],
                                    all_advantages,
                                    batch_size=np.shape(all_obs)[0],
                                    verbose=0)
        critic_loss = self.critic.fit([all_obs, all_valid_actions],
                                      all_rewards,
                                      batch_size=np.shape(all_obs)[0],
                                      verbose=0)

        self._clear_trajectories()
        return (actor_loss.history, critic_loss.history)

    def save_observation_action(self, observation, action):
        '''
        Appends the observation to the current game, or creates a new game
        entry and appends the observation to that instead.
        '''
        if self.trajectory_done: # Enter new game
            self.m_observations.append([])
            self.m_actions.append([])
            self.trajectory_done = False
        self.m_observations[-1].append(observation)
        self.m_actions[-1].append(action)

    def save_model(self, file):
        self.actor.save('{}_actor.h5'.format(file))
        self.critic.save('{}_critic.h5'.format(file))

    def save_reward(self, reward):
        """Undiscounted rewards, so just repeat the final score."""
        self.trajectory_done = True
        self.m_rewards.append(np.repeat(reward, len(self.m_observations[-1])))

    def save_pass_reward(self, last_pass_idx, last_pass_values):
        '''
        Adjusts the score of the final pass, based on whether we passed
        too early or too late.

        args:
            - last_pass_idx(list(int)): Contains the indices of the last passes.
                                        Different in case of duplicate bridge,
                                        equal otherwise.
            - last_pass_values(list(int)): Contains the corresponding value of
                                           the passes.
        '''
        assert self.m_actions[-1][last_pass_idx[0]] == 35
        assert self.m_actions[-1][last_pass_idx[1]] == 35
        self.m_rewards[-1][last_pass_idx[0]] = last_pass_values[0]
        self.m_rewards[-1][last_pass_idx[1]] = last_pass_values[1]

    def _clear_trajectories(self):
        """Flush memory."""
        self.m_observations, self.m_actions, self.m_rewards = [], [], []
        self.trajectory_done = True

    def remove_last_trajectory(self):
        """If everyone passed, do not learn."""
        del self.m_observations[-1]
        del self.m_actions[-1]
        self.trajectory_done = True
