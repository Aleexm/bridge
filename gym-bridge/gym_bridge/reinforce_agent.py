from gym_bridge.agents import Agent
import keras
import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
import copy
import random

class REINFORCEAgent(Agent):
    def __init__(self, action_space, observation_space, discount_factor=1,
                 batch_size=1, epsilon=0.1):
        super().__init__(action_space, observation_space)
        self.model = self.create_model()
        self.discount_factor = discount_factor
        self.observations, self.actions, self.rewards = [], [], []
        self.batch_size = batch_size
        self.games_in_batch = 0
        self.epsilon = epsilon

    def __repr__(self):
        return "reinforce"

    def act(self, observation, valid_actions):
        # if random.random() < self.epsilon:
        #     return np.random.choice(valid_actions)
        observation = np.reshape(observation, [1, self.observation_space])
        policy = self.model.predict(observation, batch_size=1).flatten()
        # print("input: {}, output: {}".format(np.sum(np.nonzero(observation[0])[0], axis=None), np.argmax(policy)))
        illegal = np.setdiff1d(np.arange(self.action_space), valid_actions)
        for i in illegal: # Mask out illegal moves
            policy[i] = 0
        policy /= np.sum(policy) # Normalize policy for action selection
        action = np.random.choice(self.action_space, p=policy)
        if random.random() > 0.99:
            print(policy)
        return action

    def save_reward(self, reward):
        self.games_in_batch += 1
        new = len(self.observations) - len(self.rewards)
        if new == 0:
            return
        self.rewards = np.concatenate([self.rewards, np.zeros(new)], axis=None)
        self.rewards[-new:] = reward

    def save_observation_action(self, observation, action):
        self.observations.append(observation)
        self.actions.append(action)

    def discount_rewards(self, rewards):
        if self.discount_factor == 1: # No discountin
            discounted_rewards = np.repeat(rewards[-1], len(self.observations))
            return discounted_rewards
        else:
            discounted_rewards = np.zeros_like(rewards)
            running_add = 0
            for t in reversed(range(0, len(rewards))):
                running_add = running_add * self.discount_factor + rewards[t]
                discounted_rewards[t] = running_add
            return discounted_rewards

    def create_model(self):
        inputs = Input(shape=(self.observation_space,))
        initial_fc = Dense(200, activation='relu')(inputs)
        output_1 = Dense(200, activation='relu')(initial_fc)
        output_2 = Dense(200, activation='relu')(output_1)
        skip_1 = keras.layers.add([initial_fc, output_2])
        output_3 = Dense(200, activation='relu')(skip_1)
        output_4 = Dense(200, activation='relu')(output_3)
        skip_2 = keras.layers.add([skip_1, output_4])
        actions = Dense(self.action_space, activation='softmax')(skip_2)
        # value = Dense(1, activation='linear')(skip_2)

        # model = Model(inputs=inputs, outputs=[actions,value])
        model = Model(inputs=inputs, outputs=actions)
        adam = keras.optimizers.Adam(lr=0.0003)

        model.compile(optimizer=adam, loss='categorical_crossentropy')
        return model

    def train(self):
        trajectory_length = len(self.observations)
        if self.games_in_batch < self.batch_size:
            return

        # discounted_rewards = self.discount_rewards(self.rewards).astype('float64')
        # if self.discount_factor < 1: # If discount_factor 1, rewards will be equal
        #     discounted_rewards -= np.mean(discounted_rewards)
        #     if np.std(discounted_rewards):
        #         discounted_rewards /= np.std(discounted_rewards)
        #     else:
        #         self.observations, self.actions, self.rewards = [], [], []
        #         return None
        update_inputs = np.zeros((trajectory_length, self.observation_space))
        advantages = np.zeros((trajectory_length, self.action_space))

        for i in range(trajectory_length):
            update_inputs[i] = self.observations[i]
            advantages[i][self.actions[i]] = self.rewards[i]

        loss = self.model.fit(update_inputs, advantages,
                              batch_size=len(self.observations),
                              epochs=1, verbose=0)
        self._clear_trajectories()

        return loss.history

    def _clear_trajectories(self):
        self.observations, self.actions, self.rewards = [], [], []
        self.games_in_batch = 0

    def save_model(self, file):
        self.model.save('{}.h5'.format(file))

class DummyREINFORCEAgent(Agent):
    """Opponents. Does not store anything. Is never trained. Copies model at intervals."""
    def __init__(self, action_space, observation_space, model):
        super().__init__(action_space, observation_space)
        self.model = model

    def act(self, observation, valid_actions):
        observation = np.reshape(observation, [1, self.observation_space])
        policy = self.model.predict(observation, batch_size=1).flatten()
        illegal = np.setdiff1d(np.arange(self.action_space), valid_actions)
        for i in illegal: # Mask out illegal moves
            policy[i] = 0
        policy /= np.sum(policy) # Normalize policy for action selection
        return np.random.choice(self.action_space, p=policy)
