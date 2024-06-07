#
# Financial DDQ-Learning Agent
#
# (c) Oscar A. Rangel
#

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '4'

import random
import numpy as np
from pylab import plt
from collections import deque
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dense, Dropout, BatchNormalization, LeakyReLU
from keras.models import Sequential
from keras.optimizers import Adam
from keras.models import load_model
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from collections import deque

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

# plt.style.use('seaborn')

import seaborn as sns
sns.set(style="whitegrid")

from replaybuffer import *

def set_seeds(seed=100):
    ''' Function to set seeds for all
        random number generators.
    '''
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


class TradingBot:
    def __init__(self, hidden_units, learning_rate, learn_env, valid_env=None, val=True, batch_size=32):
        self.learn_env = learn_env
        self.valid_env = valid_env
        self.val = val
        self.epsilon = 1.0
        self.epsilon_min = 0.10
        self.epsilon_decay = 0.998
        self.learning_rate = learning_rate
        self.gamma = 0.95
        self.batch_size = batch_size #128
        self.memory = PrioritizedReplayBuffer(10000)
        # self.memory = deque(maxlen=10000)
        self.lr_schedule = ExponentialDecay(learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True)
        self.max_treward = 0
        self.averages = list()
        self.trewards = []
        self.performances = list()
        self.aperformances = list()
        self.vperformances = list()
       
        # existing initialization code...
        self.best_perf = float('-inf')  # Initialize the best performance as negative infinity

        self.lr_schedule = self._get_learning_rate_schedule(self.learning_rate)
        self.model = self._build_model(hidden_units)
        self.target_model = self._build_model(hidden_units)  # Create the target model    
        self.update_target_model()  # Initially set target model weights to be the same as the primary model
        
        # Load the best model if it exists before starting training
        self._load_best_model()

    def _build_model(self, hu):
        ''' Method to create a DNN model for trading data analysis, 
            incorporating advanced activation functions and more refined network architecture. '''
        model = Sequential()
        
        # Input layer with advanced activation
        model.add(Dense(hu, input_shape=(self.learn_env.lags, self.learn_env.n_features)))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))  # Using LeakyReLU for potentially better performance on non-linear data
    
        # First hidden layer
        model.add(Dense(hu * 2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.25))  # Slightly increase dropout for this layer
    
        # Second hidden layer
        model.add(Dense(hu * 2))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.01))
        model.add(Dropout(0.25))  # Consistent dropout to maintain regularization
    
        # Output layer for actions (assuming a decision on two possible actions)
        # model.add(Dense(2, activation='linear'))  # Linear output for continuous control or policy approximation

        # Output layer for actions
        model.add(Dense(self.learn_env.action_space.n, activation='linear'))  # Dynamic output based on action space size     
    
        # Compile the model with a specific optimizer configuration
        # Use the learning rate schedule in the optimizer
        optimizer = Adam(learning_rate=self.learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
        # optimizer = Adam(learning_rate=self.lr_schedule, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=True)
        model.compile(loss='mse', optimizer=optimizer)
        
        return model
        
    def act(self, state):
        ''' Method for taking action based on
            a) exploration
            b) exploitation
        '''
        if random.random() <= self.epsilon:
            return self.learn_env.action_space.sample()
            
        action = self.model.predict(state)[0, 0]
        
        return np.argmax(action)
    
    def update_target_model(self):
        # Update the target model weights to match the primary model
        self.target_model.set_weights(self.model.get_weights())

    def replay(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        experiences, indices, weights = self.memory.sample(self.batch_size, beta=0.4)
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = np.array(states).reshape((self.batch_size, self.learn_env.lags, self.learn_env.n_features))
        next_states = np.array(next_states).reshape((self.batch_size, self.learn_env.lags, self.learn_env.n_features))

        current_q_values = self.model.predict(states)
        next_q_values = self.target_model.predict(next_states)
        next_q_values_model = self.model.predict(next_states)

        targets = np.array(current_q_values)
        new_priorities = []

        for i, (state, action, reward, next_state, done) in enumerate(experiences):
            current_action_q_values = current_q_values[i, action]
            if done:
                targets[i, action] = reward
                td_error = np.abs(reward - current_action_q_values)
            else:
                selected_action = np.argmax(next_q_values_model[i])
                selected_action = np.clip(selected_action, 0, next_q_values.shape[1] - 1)
                target_q_value = reward + self.gamma * next_q_values[i, selected_action]
                targets[i, action] = target_q_value
                td_error = np.abs(target_q_value - current_action_q_values)

            # Calculate a scalar TD-error by taking the maximum of the errors
            scalar_td_error = np.max(td_error)
            new_priorities.append(scalar_td_error)

        self.model.fit(states, targets, sample_weight=np.array(weights), batch_size=self.batch_size, verbose=0, shuffle=False)
        self.memory.update_priorities(indices, new_priorities)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
   
    def learn(self, episodes):
        ''' Method to train the DQL agent.
        '''
        # Load the best model if it exists before starting training
        # self.load_best_model()

        for e in range(1, episodes + 1):
            state = self.learn_env.reset()
            state = np.reshape(state, [1, self.learn_env.lags, self.learn_env.n_features])

            for _ in range(10000):
                action = self.act(state)
                next_state, reward, done, info = self.learn_env.step(action)
                next_state = np.reshape(next_state,
                                        [1, self.learn_env.lags,
                                        self.learn_env.n_features])

                # Use an initial priority, e.g., the absolute reward, or simply a constant like 1.0
                initial_priority = abs(reward) + 1.0  # Basic example for initial priority

                # Correct usage: add experience with an initial priority
                self.memory.add((state, action, reward, next_state, done), initial_priority)

                state = next_state
                if done:
                    treward = _ + 1
                    self.trewards.append(treward)
                    av = sum(self.trewards[-25:]) / 25
                    perf = self.learn_env.performance
                    self.averages.append(av)
                    self.performances.append(perf)
                    self.aperformances.append(
                        sum(self.performances[-25:]) / 25)
                    self.max_treward = max(self.max_treward, treward)
                    templ = 'episode: {:2d}/{} | treward: {:4d} | '
                    templ += 'perf: {:5.3f} | av: {:5.1f} | max: {:4d}'
                    print(templ.format(e, episodes, treward, perf,
                                    av, self.max_treward), end='\r')
                    break

            if self.val:
                self.validate(e, episodes)
            if len(self.memory.buffer) > self.batch_size:  # Make sure to check the buffer size
                self.replay()
        print()

           
    def update_learning_rate(self, new_lr):
        # Update the learning rate for the optimizer
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
    
    def _get_learning_rate_schedule(initial_lr, decay_steps=1000, decay_rate=0.96):
        # Create a learning rate schedule
        lr_schedule = ExponentialDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            decay_rate=decay_rate,
            staircase=True  # If True, learning rate changes at discrete intervals
        )
        return lr_schedule
        
    def validate(self, e, episodes):
        ''' Method to validate the performance of the DQL agent and save the model if it achieves a new best performance. '''
        state = self.valid_env.reset()
        state = np.reshape(state, [1, self.valid_env.lags, self.valid_env.n_features])
        total_reward = 0
        
        for _ in range(10000):
            action = np.argmax(self.model.predict(state)[0, 0])
            next_state, reward, done, info = self.valid_env.step(action)
            state = np.reshape(next_state, [1, self.valid_env.lags, self.valid_env.n_features])
            total_reward += reward  # Summing up rewards to calculate total performance
    
            if done:
                treward = _ + 1
                perf = self.valid_env.performance
                self.vperformances.append(perf)
    
                if perf > self.best_perf:
                    self.best_perf = perf
                    
                    directory = "models"
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                        
                    model_path = f"{directory}/best_model_up_to_ep_{e}_perf_{perf:.3f}.h5"
                    self.model.save_weights(model_path)  # Save only the weights
                    print(f"New best performance: {perf:.3f} at episode {e}, model saved to {model_path}")
    
                if e % 100 == 0:  # You can adjust the frequency of these summaries as needed
                    templ = 70 * '='
                    templ += f'\nepisode: {e}/{episodes} | VALIDATION | treward: {treward} | perf: {perf:.3f} | eps: {self.epsilon:.2f}\n'
                    templ += 70 * '='
                    print(templ)
                    
                break

    def _load_best_model(self):
        directory = "models"
        best_model_path = None
        # best_perf = float('-inf')
    
        # Check if the models directory exists
        if os.path.exists(directory):
            # List all files in the directory
            files = os.listdir(directory)
            
            # Continue only if there are files in the directory
            if files:
                for filename in files:
                    if filename.startswith("best_model") and filename.endswith(".h5"):
                        # Extract performance from filename
                        parts = filename.split('_')
                        perf = float(parts[-1][:-3])  # Remove the '.h5' and convert to float
    
                        # Check if this model has the best performance
                        if perf > self.best_perf:
                            self.best_perf = perf
                            best_model_path = os.path.join(directory, filename)
    
                # If a best model was found, load it into the primary model and update the target model
                if best_model_path:
                    print(f"Loading best model from {best_model_path} with performance {self.best_perf}")
                    self.model.load_weights(best_model_path)
                    self.target_model.set_weights(self.model.get_weights())  # Ensure target model is synchronized
                    self.best_perf = self.best_perf  # Update the best performance tracked
                else:
                    print("No appropriate model found in the directory. Starting training from scratch.")
            else:
                print("Model directory is empty. Starting training from scratch.")
        else:
            print("Model directory does not exist. Starting training from scratch.")
    


    def update_learning_rate(self, new_lr):
        # Update the learning rate for the optimizer
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
    
def plot_treward(agent):
    ''' Function to plot the total reward
        per training eposiode.
    '''
    plt.figure(figsize=(10, 6))
    x = range(1, len(agent.averages) + 1)
    y = np.polyval(np.polyfit(x, agent.averages, deg=3), x)
    plt.plot(x, agent.averages, label='moving average')
    plt.plot(x, y, 'r--', label='regression')
    plt.xlabel('episodes')
    plt.ylabel('total reward')
    plt.legend()


def plot_performance(agent):
    ''' Function to plot the financial gross
        performance per training episode.
    '''
    plt.figure(figsize=(10, 6))
    x = range(1, len(agent.performances) + 1)
    y = np.polyval(np.polyfit(x, agent.performances, deg=3), x)
    plt.plot(x, agent.performances[:], label='training')
    plt.plot(x, y, 'r--', label='regression (train)')
    if agent.val:
        y_ = np.polyval(np.polyfit(x, agent.vperformances, deg=3), x)
        plt.plot(x, agent.vperformances[:], label='validation')
        plt.plot(x, y_, 'r-.', label='regression (valid)')
    plt.xlabel('episodes')
    plt.ylabel('gross performance')
    plt.legend()
