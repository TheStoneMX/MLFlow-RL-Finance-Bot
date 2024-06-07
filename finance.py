#
# Finance Environment
#
# (c) Dr. Yves J. Hilpisch
# Artificial Intelligence in Finance
#
'''
For additional test data similar to aiif_eikon_eod_data.csv, you can explore various datasets available at URLs such as:

http://hilpisch.com/indices_eikon_eod_data.csv
http://hilpisch.com/dax_eikon_eod_data.csv
http://hilpisch.com/tr_eikon_option_data.csv

These datasets contain historical end-of-day financial time series data that you can use for testing and further analysis. 
You might find them within the context of the "Python for Algorithmic Trading" materials, or on websites associated with financial data science and algorithmic trading courses. 
If you're looking for specific types of data or instruments, make sure to check the supporting documents or resources section of the course you're following, 
as there might be additional links or references to data sources provided.

The format of a CSV (Comma-Separated Values) file is a plain text file that contains data separated by commas. Each line in a CSV file corresponds to a row in a table, 
and each item in a line, separated by a comma, corresponds to a cell in that row. CSV files are widely used for data exchange because they are relatively simple, 
can be read and written by most technological platforms, are human-readable, and can be easily imported into various applications like spreadsheet software, 
databases, and data analysis tools. Here's an example of what the content in a CSV file might look like:

column1,column2,column3
data1,data2,data3
data4,data5,data6

In the example above, the first line typically represents the headers for each column, and the subsequent lines represent the data rows. However, 
the delimiter can sometimes be different, such as a semicolon or tab, and this should be specified when reading or writing the file if it differs from the standard comma.

You can find Forex data from various sources, which often provide both historical and real-time data. Here are some platforms and methods through which you can access Forex data:

Oanda: Oanda provides access to a variety of trading instruments including Forex. You can access historical data and real-time Forex data through the Oanda API using Python wrappers like tpqoa.

FXCM: FXCM is another platform that offers free tick data for a number of currency pairs. They offer APIs that can be utilized for retrieving both historical and real-time Forex data.

Refinitiv Eikon: Refinitiv Eikon provides a vast universe of financial data that includes Forex data. You may need a subscription to access this data.

Broker Platforms: Many broker platforms, like Rwando or FXCM as mentioned, provide historical data for Forex and other financial instruments.

Free APIs: Certain free APIs such as the Yahoo! Finance and Google Finance APIs provide historical daily data for various instruments, including Forex pairs, though availability and reliability may vary.

Public Datasets: There are publicly available datasets and CSV files that contain historical Forex data, sometimes available through educational resources or trading platforms.

Remember that while some data may be freely accessible, other sources might require a subscription or an account with the platform, and usage may be subject to limitations set by the data provider. 
Always ensure you are compliant with terms of service and usage agreements when accessing and using financial data.
'''


import math
import random
import numpy as np
import pandas as pd


class observation_space:
    def __init__(self, n):
        self.shape = (n,)


class action_space:
    def __init__(self, n):
        self.n = n

    def sample(self):
        return random.randint(0, self.n - 1)


class Finance:
    url = 'http://hilpisch.com/aiif_eikon_eod_data.csv'
    # url = 'http://hilpisch.com/indices_eikon_eod_data.csv'
    # url = 'http://hilpisch.com/dax_eikon_eod_data.csv'
    # url = 'http://hilpisch.com/tr_eikon_option_data.csv'

    def __init__(self, symbol, features, window, lags,
                 leverage=1, min_performance=0.85,
                 start=0, end=None, mu=None, std=None):
        self.symbol = symbol
        self.features = features
        self.n_features = len(features)
        self.window = window
        self.lags = lags
        self.leverage = leverage
        self.min_performance = min_performance
        self.start = start
        self.end = end
        self.mu = mu
        self.std = std
        self.observation_space = observation_space(self.lags)
        self.action_space = action_space(2)
        self._get_data()
        self._prepare_data()

    def _get_data(self):
        self.raw = pd.read_csv(self.url, index_col=0,
                               parse_dates=True).dropna()

    def _prepare_data(self):
        self.data = pd.DataFrame(self.raw[self.symbol])
        self.data = self.data.iloc[self.start:]
        self.data['r'] = np.log(self.data / self.data.shift(1))
        self.data.dropna(inplace=True)
        self.data['m'] = self.data['r'].rolling(self.window).mean()
        self.data['s'] = self.data[self.symbol].rolling(self.window).mean()
        self.data['v'] = self.data['r'].rolling(self.window).std()
        self.data.dropna(inplace=True)
        if self.mu is None:
            self.mu = self.data.mean()
            self.std = self.data.std()
        self.data_ = (self.data - self.mu) / self.std
        self.data['d'] = np.where(self.data['r'] > 0, 1, 0)
        self.data['d'] = self.data['d'].astype(int)
        if self.end is not None:
            self.data = self.data.iloc[:self.end - self.start]
            self.data_ = self.data_.iloc[:self.end - self.start]

    def _get_state(self):
        return self.data_[self.features].iloc[self.bar -
                                              self.lags:self.bar]

    def get_state(self, bar):
        return self.data_[self.features].iloc[bar - self.lags:bar]

    def seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def reset(self):
        self.treward = 0
        self.accuracy = 0
        self.performance = 1
        self.bar = self.lags
        state = self.data_[self.features].iloc[self.bar -
                                               self.lags:self.bar]
        return state.values

    def step(self, action):
        correct = action == self.data['d'].iloc[self.bar]
        ret = self.data['r'].iloc[self.bar] * self.leverage
        reward_ = 1 if correct else 0
        reward = abs(ret) if correct else -abs(ret)
        factor = 1 if correct else -1
        self.treward += reward_
        self.bar += 1
        self.accuracy = self.treward / (self.bar - self.lags)
        self.performance *= math.exp(reward)
        if self.bar >= len(self.data):
            done = True
        elif reward == 1:
            done = False
        elif (self.performance < self.min_performance and
              self.bar > self.lags + 5):
            done = True
        else:
            done = False
        state = self._get_state()
        info = {}
        return state.values, reward, done, info
