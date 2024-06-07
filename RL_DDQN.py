import os
import numpy as np
import pandas as pd
from pylab import plt
import mlflow
import mlflow.pyfunc

# Setting options and seeds
pd.set_option('mode.chained_assignment', None)
pd.set_option('display.float_format', '{:.4f}'.format)
np.set_printoptions(suppress=True, precision=4)
os.environ['PYTHONHASHSEED'] = '0'

# Importing custom modules
import finance
import tradingbot

# Configuration
symbol = 'EUR='
features = [symbol, 'r', 's', 'm', 'v']
a = 0
b = 1750
c = 250
episodes = 46

# Setup MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Change to your MLflow server URI
mlflow.set_experiment("TradingBotExperiment")

def reshape(s, env):
    return np.reshape(s, [1, env.lags, env.n_features])

def backtest(agent, env):
    done = False
    env.data['p'] = 0
    state = env.reset()
    while not done:
        action = np.argmax(agent.model.predict(reshape(state, env))[0, 0])
        position = 1 if action == 1 else -1
        env.data.loc[:, 'p'].iloc[env.bar] = position
        state, reward, done, info = env.step(action)
    env.data['s'] = env.data['p'] * env.data['r']

# MLflow tracking
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("symbol", symbol)
    mlflow.log_param("features", features)
    mlflow.log_param("window", 20)
    mlflow.log_param("lags", 6)
    mlflow.log_param("leverage", 1)
    mlflow.log_param("min_performance", 0.85)
    mlflow.log_param("train_start", a)
    mlflow.log_param("train_end", a + b)
    mlflow.log_param("validation_start", a + b)
    mlflow.log_param("validation_end", a + b + c)
    mlflow.log_param("episodes", episodes)

    # Initialize environments
    learn_env = finance.Finance(symbol, features, window=20, lags=6,
                     leverage=1, min_performance=0.85,
                     start=a, end=a + b, mu=None, std=None)
    
    valid_env = finance.Finance(symbol, features, window=learn_env.window,
                     lags=learn_env.lags, leverage=learn_env.leverage,
                     min_performance=learn_env.min_performance,
                     start=a + b, end=a + b + c,
                     mu=learn_env.mu, std=learn_env.std)
    
    # Initialize the trading agent
    tradingbot.set_seeds(100)
    agent = tradingbot.TradingBot(32, 0.00001, learn_env, valid_env)

    # Train the agent
    agent.learn(episodes)
    
    # Log the model
    mlflow.pyfunc.log_model("trading_model", python_model=agent.model)

    # Perform backtesting on the training environment
    env = agent.learn_env
    backtest(agent, env)
    
    # Log metrics
    performance = env.data[['r', 's']].iloc[env.lags:].sum().apply(np.exp) - 1
    mlflow.log_metric("training_performance_r", performance['r'])
    mlflow.log_metric("training_performance_s", performance['s'])

    # Perform backtesting on the validation environment
    env = valid_env
    backtest(agent, env)
    
    # Log metrics
    performance = env.data[['r', 's']].iloc[env.lags:].sum().apply(np.exp) - 1
    mlflow.log_metric("validation_performance_r", performance['r'])
    mlflow.log_metric("validation_performance_s", performance['s'])

    # Perform vectorized backtesting
    test_env = finance.Finance(symbol, features, window=learn_env.window,
                     lags=learn_env.lags, leverage=learn_env.leverage,
                     min_performance=learn_env.min_performance,
                     start=a + b + c, end=None,
                     mu=learn_env.mu, std=learn_env.std)
    
    env = test_env
    backtest(agent, env)
    
    # Log metrics
    performance = env.data[['r', 's']].iloc[env.lags:].sum().apply(np.exp) - 1
    mlflow.log_metric("test_performance_r", performance['r'])
    mlflow.log_metric("test_performance_s", performance['s'])

# Plots
env.data[['r', 's']].iloc[env.lags:].cumsum().apply(np.exp).plot(figsize=(10, 6))
plt.title("Performance over time")
plt.show()