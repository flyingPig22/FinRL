from __future__ import annotations

import gymnasium as gym
import numpy as np
from numpy import random as rd
import math


class StockTradingEnv(gym.Env):
    def __init__(
        self,
        config,
        initial_account=1e6,
        # gamma=0.99,
        gamma=1,
        turbulence_thresh=99,
        min_stock_rate=0.1,
        max_stock=5 * 1e2,
        initial_capital=1e6,
        buy_cost_pct=3e-2,
        sell_cost_pct=2e-2,
        reward_scaling=2**-14,
        short_premium_pct=0.5,
        initial_stocks=None,
    ):
        timestamp_array = config["timestamp_array"]
        price_ary = config["price_array"]
        tech_ary = config["tech_array"]
        turbulence_ary = config["turbulence_array"]
        if_train = config["if_train"]
        volume_array = config["volume_array"]
        open_price_array = config["open_price_array"]
        high_price_array = config["high_price_array"]
        low_price_array = config["low_price_array"]
        history_price_array = config["history_price_array"]
        history_volume_array = config["history_volume_array"]
        history_turbulence_array = config["history_turbulence_array"]
        history_tech_array = config["history_tech_array"]
        self.price_ary = price_ary.astype(np.float32)
        self.tech_ary = tech_ary.astype(np.float32)
        # self.turbulence_ary = turbulence_ary
        self.volume_array = volume_array.astype(np.float32)
        self.open_price_array = open_price_array.astype(np.float32)
        self.high_price_array = high_price_array.astype(np.float32)
        self.low_price_array = low_price_array.astype(np.float32)
        self.timestamp_array = timestamp_array
        self.history_price_array = history_price_array.astype(np.float32)
        self.history_volume_array = history_volume_array.astype(np.float32)
        self.history_turbulence_array = history_turbulence_array.astype(np.float32)
        self.days_diff = len(self.history_price_array) - len(self.price_ary)
        self.history_turbulence_array_normalized = self.history_turbulence_array * 1e-2
        self.history_tech_array = history_tech_array.astype(np.float32)

        self.history_rsi_array = self.history_tech_array[:, 0:1] * 1e-2

        # self.tech_ary = self.tech_ary * 2**-7
        self.turbulence_bool = (turbulence_ary > turbulence_thresh).astype(np.float32)
        # self.turbulence_ary = (
        #     self.sigmoid_sign(turbulence_ary, turbulence_thresh) * 2**-5
        # ).astype(np.float32)

        stock_dim = self.price_ary.shape[1]
        self.gamma = gamma
        self.max_stock = max_stock
        self.min_stock_rate = min_stock_rate
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.initial_capital = initial_capital
        # self.initial_stocks = (
        #     np.zeros(stock_dim, dtype=np.float32)
        #     if initial_stocks is None
        #     else initial_stocks
        # )

        # reset()
        self.day = None
        self.amount = None
        self.stocks = None
        self.total_asset = None
        self.gamma_reward = None
        self.initial_total_asset = None

        # environment information
        self.env_name = "StockEnv"
        # self.state_dim = 1 + 2 + 2 * stock_dim + self.tech_ary.shape[1]
        # # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        # self.state_dim = 1 + 2 + 3 * stock_dim + self.tech_ary.shape[1]
        # self.tech_ary first is RSI, removed later
        self.state_dim = 40
        # amount + (turbulence, turbulence_bool) + (price, stock) * stock_dim + tech_dim
        self.stocks_cd = None
        self.action_dim = stock_dim
        self.max_step = self.price_ary.shape[0] - 1
        self.if_train = if_train
        self.enable_logging = not if_train
        # self.enable_logging = True
        self.if_discrete = False
        self.target_return = 10.0
        self.episode_return = 0.0
        import math
        def next_power_of_10(A):
            exponent = math.floor(math.log10(A)) + 1
            return 10 ** exponent
        # self.amount_scaler = 1.0 / next_power_of_10(initial_capital)
        self.amount_scaler = 1e-8
        # self.stocks_scaler = 1.0 / (initial_capital * 10.0 / min(self.price_ary))
        self.stocks_scaler = 1e-6
        # price_min = np.min(self.price_ary)
        # price_max = np.max(self.price_ary)
        # # Normalize the prices to [-1, 1]
        # self.close_price_normalized = 2 * (self.price_ary - price_min) / (price_max - price_min) - 1
        self.close_price_normalized = self.price_ary * 1e-3
        # open_price_min = np.min(self.open_price_array)
        # open_price_max = np.max(self.open_price_array)
        # self.open_price_normalized = 2 * (self.open_price_array - open_price_min) / (open_price_max - open_price_min) - 1
        self.open_price_normalized = self.open_price_array * 1e-3
        # high_price_min = np.min(self.high_price_array)
        # high_price_max = np.max(self.high_price_array)
        # self.high_price_normalized = 2 * (self.high_price_array - high_price_min) / (
        #             high_price_max - high_price_min) - 1
        self.high_price_normalized = self.high_price_array * 1e-3
        # low_price_min = np.min(self.low_price_array)
        # low_price_max = np.max(self.low_price_array)
        # self.low_price_normalized = 2 * (self.low_price_array - low_price_min) / (
        #         low_price_max - low_price_min) - 1
        self.low_price_normalized = self.low_price_array * 1e-3
        self.vix_arr = turbulence_ary
        # vix_min = np.min(turbulence_ary)
        # vix_max = np.max(turbulence_ary)
        # self.vix_normalized = 2 * (turbulence_ary - vix_min) / (vix_max - vix_min) - 1
        self.vix_normalized = turbulence_ary * 1e-2

        # normalize tech array
        # tech_min = np.min(tech_ary, axis=0, keepdims=True)
        # tech_max = np.max(tech_ary, axis=0, keepdims=True)
        # # Avoid division by zero in case a column has constant value
        # denom = tech_max - tech_min
        # denom[denom == 0] = 1  # or choose another strategy to handle constant columns
        # self.tech_ary_normalized = 2 * (tech_ary - tech_min) / denom - 1
        # self.tech_ary_normalized = tech_ary * 1e-3

        self.rsi_array = self.tech_ary[:, 0:1]
        self.sma_array = self.tech_ary[:, 1:]

        self.rsi_array_normalized = self.rsi_array * 1e-2

        # hardcode for qqq
        self.action_units = 500

        self.observation_space = gym.spaces.Box(
            low=-10, high=10, shape=(self.state_dim,), dtype=np.float32
        )
        # self.action_space = gym.spaces.Box(
        #     low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        # )
        self.action_space = gym.spaces.Discrete(3)
        # for each buy/sell, allocate total assets into self.allocation parts
        self.allocations = 2

        # set cool down period, 5 days
        self.cool_down = 0

    def reset(self, *, seed=None, options=None):
        # return self._reset_v1()
        return self._reset_v2()

    def _reset_v2(self):
        self.day = 0
        price = self.price_ary[self.day][0]
        self.action_units = self.initial_capital // price // 100 * 100
        if self.if_train:
            self.stocks = rd.randint(0, 1) * self.action_units
            self.amount = self.initial_capital - self.stocks * price
        else:
            self.stocks = 0
            self.amount = self.initial_capital
        self.total_asset = self.amount + self.stocks * price
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(), {}

    def _reset_v1(self, *, seed=None, options=None):
        self.day = 0
        price = self.price_ary[self.day][0]
        self.action_units = self.initial_capital // price // 100 // self.allocations * 100
        if self.if_train:
            self.stocks = (rd.randint(0, self.allocations) * self.action_units)
            self.amount = self.initial_capital - self.stocks * price
        else:
            self.stocks = 0
            self.amount = self.initial_capital
        self.total_asset = self.amount + self.stocks * price
        self.initial_total_asset = self.total_asset
        self.gamma_reward = 0.0
        return self.get_state(), {}

    def _step_v6(self, action):
        reward_decay = 0.9
        earning_period = 11
        buy_pct = 1 / 2
        sell_pct = 1 / 2
        pre_price = self.price_ary[self.day][0]
        yesterday = self.day
        self.day += 1
        today = self.day
        cur_price = self.price_ary[today][0]
        if action == 0:
            # buy
            if self.enable_logging:
                print("\n")
                print("==========Action Start==========")
                print(f"Yesterday: {self.timestamp_array[yesterday]}, Price: {pre_price}, Holding shares: {self.stocks}, Holding cash: {self.amount}, Total asset: {self.total_asset}")
            buy_num_shares = (buy_pct * self.amount) // (pre_price * (1 + self.buy_cost_pct))
            self.amount = self.amount - buy_num_shares * pre_price * (1 + self.buy_cost_pct)
            self.stocks += buy_num_shares
            self.total_asset = self.amount + self.stocks * cur_price
            if self.enable_logging:
                print(f"Action is buy {buy_num_shares} shares, Holding shares: {self.stocks}, Holding cash: {self.amount}")
            reward = 0
            price_len = len(self.price_ary)
            if today < price_len:
                reward += self.price_ary[today][0] - pre_price
            for i in range(1, earning_period):
                if today + i < price_len:
                    price_now = self.price_ary[today + i][0]
                    price_prev = self.price_ary[today + i - 1][0]
                    reward += (price_now - price_prev) * (reward_decay ** i)
            reward = reward * self.stocks - buy_num_shares * pre_price * self.buy_cost_pct
            reward *= self.reward_scaling
            done = self.day == self.max_step
            state = self.get_state()
            if self.enable_logging:
                print(f"Today: {self.timestamp_array[today]}, Price: {cur_price}, Reward: {reward}, Total asset: {self.total_asset}")
                print("==========Action End==========")
            return state, float(reward), done, False, dict()
        elif action == 1:
            # sell
            if self.enable_logging:
                print("\n")
                print("==========Action Start==========")
                print(f"Yesterday: {self.timestamp_array[yesterday]}, Price: {pre_price}, Holding shares: {self.stocks}, Holding cash: {self.amount}, Total asset: {self.total_asset}")
            sell_num_shares = int(self.stocks * sell_pct)
            self.amount  = self.amount + sell_num_shares * pre_price
            self.stocks = self.stocks - sell_num_shares
            self.total_asset = self.amount + self.stocks * cur_price
            if self.enable_logging:
                print(f"Action is sell {sell_num_shares} shares, Holding shares: {self.stocks}, Holding cash: {self.amount}")
            reward = 0

            price_len = len(self.price_ary)
            if today < price_len:
                reward += self.price_ary[today][0] - pre_price
            for i in range(1, earning_period):
                if today + i < price_len:
                    price_now = self.price_ary[today + i][0]
                    price_prev = self.price_ary[today + i - 1][0]
                    reward += (price_now - price_prev) * (reward_decay ** i)
            reward *= self.stocks * self.reward_scaling

            done = self.day == self.max_step
            state = self.get_state()
            if self.enable_logging:
                print(f"Today: {self.timestamp_array[today]}, Price: {cur_price}, Reward: {reward}, Total asset: {self.total_asset}")
                print("==========Action End==========")
            return state, float(reward), done, False, dict()
        elif action == 2:
            # hold
            if self.enable_logging:
                print("\n")
                print("==========Action Start==========")
                print(f"Yesterday: {self.timestamp_array[yesterday]}, Price: {pre_price}, Holding shares: {self.stocks}, Holding cash: {self.amount}, Total asset: {self.total_asset}")
            hold_num_shares = self.stocks
            self.total_asset = self.amount + hold_num_shares * cur_price
            if self.enable_logging:
                print(f"Action is hold {hold_num_shares} shares, Holding shares: {hold_num_shares}, Holding cash: {self.amount}")
            reward = 0
            price_len = len(self.price_ary)
            if today < price_len:
                reward += self.price_ary[today][0] - pre_price
            for i in range(1, earning_period):
                if today + i < price_len:
                    price_now = self.price_ary[today + i][0]
                    price_prev = self.price_ary[today + i - 1][0]
                    reward += (price_now - price_prev) * (reward_decay ** i)
            reward *= self.stocks * self.reward_scaling
            done = self.day == self.max_step
            state = self.get_state()
            if self.enable_logging:
                print(f"Today: {self.timestamp_array[today]}, Price: {cur_price}, Reward: {reward}, Total asset: {self.total_asset}")
                print("==========Action End==========")
            return state, float(reward), done, False, dict()



    def _step_v5(self, action):
        reward_decay = 0.9
        reward_range = 7
        reward_penalty = 0.5 # hold vs buy in certain situations
        pre_price = self.price_ary[self.day][0]
        yesterday = self.day
        self.day += 1
        today = self.day
        cur_price = self.price_ary[today][0]
        if action == 0:
            # buy
            if self.enable_logging:
                print("\n")
                print("==========Action Start==========")
                print(f"Yesterday: {self.timestamp_array[yesterday]}, Price: {pre_price}, Holding shares: {self.stocks}, Holding cash: {self.amount}, Total asset: {self.total_asset}")
            buy_num_shares = self.amount // (pre_price * (1 + self.buy_cost_pct))
            self.amount = self.amount - buy_num_shares * pre_price * (1 + self.buy_cost_pct)
            self.stocks += buy_num_shares
            self.total_asset = self.amount + self.stocks * cur_price
            if self.enable_logging:
                print(f"Action is buy {buy_num_shares} shares, Holding shares: {self.stocks}, Holding cash: {self.amount}")
            left_reward, right_reward, reward = 0, 0, 0
            price_len = len(self.price_ary)
            if today < price_len:
                right_reward += self.price_ary[today][0] - pre_price
            for i in range(1, reward_range):
                if today + i < price_len:
                    price_now = self.price_ary[today + i][0]
                    price_prev = self.price_ary[today + i - 1][0]
                    right_reward += (price_now - price_prev) * (reward_decay ** i)
            if yesterday > 0:
                left_reward += pre_price - self.price_ary[yesterday - 1][0]
            for i in range(1, reward_range):
                if yesterday - i > 0:
                    price_now = self.price_ary[yesterday - i][0]
                    price_pre = self.price_ary[yesterday - i - 1][0]
                    left_reward += (price_now - price_pre) * (reward_decay ** i)
            if left_reward < 0 and right_reward > 0:
                reward = abs(left_reward) + abs(right_reward)
            elif left_reward >= 0 and right_reward >= 0 and abs(left_reward) < abs(right_reward):
                reward = abs(left_reward) + abs(right_reward)
            elif left_reward >= 0 and right_reward >=0 and abs(left_reward) >= abs(right_reward):
                reward = abs(right_reward) * reward_penalty
            elif left_reward > 0 and right_reward < 0:
                reward = -(abs(left_reward) + abs(right_reward))
            elif left_reward <= 0 and right_reward <= 0 and abs(left_reward) < abs(right_reward):
                reward = -(abs(left_reward) + abs(right_reward))
            elif left_reward <= 0 and right_reward <= 0 and abs(left_reward) >= abs(right_reward):
                reward = -abs(right_reward)
            reward = reward * self.stocks - buy_num_shares * pre_price * self.buy_cost_pct
            reward *= self.reward_scaling
            done = self.day == self.max_step
            state = self.get_state()
            if self.enable_logging:
                print(f"Today: {self.timestamp_array[today]}, Price: {cur_price}, Reward: {reward}, Total asset: {self.total_asset}")
                print("==========Action End==========")
            return state, float(reward), done, False, dict()
        elif action == 1:
            # sell
            if self.enable_logging:
                print("\n")
                print("==========Action Start==========")
                print(f"Yesterday: {self.timestamp_array[yesterday]}, Price: {pre_price}, Holding shares: {self.stocks}, Holding cash: {self.amount}, Total asset: {self.total_asset}")
            sell_num_shares = self.stocks
            self.amount = self.amount + sell_num_shares * pre_price
            self.stocks = 0
            self.total_asset = self.amount
            if self.enable_logging:
                print(f"Action is sell {sell_num_shares} shares, Holding shares: 0, Holding cash: {self.amount}")
            reward = 0
            done = self.day == self.max_step
            state = self.get_state()
            if self.enable_logging:
                print(f"Today: {self.timestamp_array[today]}, Price: {cur_price}, Reward: {reward}, Total asset: {self.total_asset}")
                print("==========Action End==========")
            return state, float(reward), done, False, dict()
        elif action == 2:
            # hold
            if self.enable_logging:
                print("\n")
                print("==========Action Start==========")
                print(f"Yesterday: {self.timestamp_array[yesterday]}, Price: {pre_price}, Holding shares: {self.stocks}, Holding cash: {self.amount}, Total asset: {self.total_asset}")
            hold_num_shares = self.stocks
            self.total_asset = self.amount + hold_num_shares * cur_price
            if self.enable_logging:
                print(f"Action is hold {hold_num_shares} shares, Holding shares: {hold_num_shares}, Holding cash: {self.amount}")
            left_reward, right_reward, reward = 0, 0, 0
            price_len = len(self.price_ary)
            if today < price_len:
                right_reward += self.price_ary[today][0] - pre_price
            for i in range(1, reward_range):
                if today + i < price_len:
                    price_now = self.price_ary[today + i][0]
                    price_prev = self.price_ary[today + i - 1][0]
                    right_reward += (price_now - price_prev) * (reward_decay ** i)
            if yesterday > 0:
                left_reward += pre_price - self.price_ary[yesterday - 1][0]
            for i in range(1, reward_range):
                if yesterday - i > 0:
                    price_now = self.price_ary[yesterday - i][0]
                    price_pre = self.price_ary[yesterday - i - 1][0]
                    left_reward += (price_now - price_pre) * (reward_decay ** i)
            if left_reward < 0 and right_reward > 0:
                reward = abs(right_reward)
            elif left_reward >= 0 and right_reward >= 0 and abs(left_reward) < abs(right_reward):
                reward = abs(right_reward)
            elif left_reward >= 0 and right_reward >= 0 and abs(left_reward) >= abs(right_reward):
                reward = abs(right_reward)
            elif left_reward > 0 and right_reward < 0:
                reward = -abs(right_reward)
            elif left_reward <= 0 and right_reward <= 0 and abs(left_reward) < abs(right_reward):
                reward = -abs(right_reward)
            elif left_reward <= 0 and right_reward <= 0 and abs(left_reward) >= abs(right_reward):
                reward = -abs(right_reward) * reward_penalty
            reward *= self.stocks * self.reward_scaling
            done = self.day == self.max_step
            state = self.get_state()
            if self.enable_logging:
                print(f"Today: {self.timestamp_array[today]}, Price: {cur_price}, Reward: {reward}, Total asset: {self.total_asset}")
                print("==========Action End==========")
            return state, float(reward), done, False, dict()

    def _step_v4(self, action):
        reward_decay = 0.9
        pre_price = self.price_ary[self.day][0]
        yesterday = self.day
        self.day += 1
        today = self.day
        cur_price = self.price_ary[today][0]
        if action == 0:
            # buy
            if self.enable_logging:
                print("\n")
                print("==========Action Start==========")
                print(f"Yesterday: {self.timestamp_array[yesterday]}, Price: {pre_price}, Holding shares: {self.stocks}, Holding cash: {self.amount}, Total asset: {self.total_asset}")
            buy_num_shares = self.amount // (pre_price * (1 + self.buy_cost_pct))
            self.amount = self.amount - buy_num_shares * pre_price * (1 + self.buy_cost_pct)
            self.stocks += buy_num_shares
            self.total_asset = self.amount + self.stocks * cur_price
            if self.enable_logging:
                print(f"Action is buy {buy_num_shares} shares, Holding shares: {self.stocks}, Holding cash: {self.amount}")
            reward = 0
            price_len = len(self.price_ary)
            if today < price_len:
                reward += self.price_ary[today][0] - pre_price
            for i in range(1, 17):
                if today + i < price_len:
                    price_now = self.price_ary[today + i][0]
                    price_prev = self.price_ary[today + i - 1][0]
                    reward += (price_now - price_prev) * (reward_decay ** i)
            reward = reward * self.stocks - buy_num_shares * pre_price * self.buy_cost_pct
            reward *= self.reward_scaling
            done = self.day == self.max_step
            state = self.get_state()
            if self.enable_logging:
                print(f"Today: {self.timestamp_array[today]}, Price: {cur_price}, Reward: {reward}, Total asset: {self.total_asset}")
                print("==========Action End==========")
            return state, float(reward), done, False, dict()
        elif action == 1:
            # sell
            if self.enable_logging:
                print("\n")
                print("==========Action Start==========")
                print(f"Yesterday: {self.timestamp_array[yesterday]}, Price: {pre_price}, Holding shares: {self.stocks}, Holding cash: {self.amount}, Total asset: {self.total_asset}")
            sell_num_shares = self.stocks
            self.amount  = self.amount + sell_num_shares * pre_price
            self.stocks = 0
            self.total_asset = self.amount
            if self.enable_logging:
                print(f"Action is sell {sell_num_shares} shares, Holding shares: 0, Holding cash: {self.amount}")
            reward = 0
            done = self.day == self.max_step
            state = self.get_state()
            if self.enable_logging:
                print(f"Today: {self.timestamp_array[today]}, Price: {cur_price}, Reward: {reward}, Total asset: {self.total_asset}")
                print("==========Action End==========")
            return state, float(reward), done, False, dict()
        elif action == 2:
            # hold
            if self.enable_logging:
                print("\n")
                print("==========Action Start==========")
                print(f"Yesterday: {self.timestamp_array[yesterday]}, Price: {pre_price}, Holding shares: {self.stocks}, Holding cash: {self.amount}, Total asset: {self.total_asset}")
            hold_num_shares = self.stocks
            self.total_asset = self.amount + hold_num_shares * cur_price
            if self.enable_logging:
                print(f"Action is hold {hold_num_shares} shares, Holding shares: {hold_num_shares}, Holding cash: {self.amount}")
            reward = 0
            price_len = len(self.price_ary)
            if today < price_len:
                reward += self.price_ary[today][0] - pre_price
            for i in range(1, 17):
                if today + i < price_len:
                    price_now = self.price_ary[today + i][0]
                    price_prev = self.price_ary[today + i - 1][0]
                    reward += (price_now - price_prev) * (reward_decay ** i)
            reward *= self.stocks * self.reward_scaling
            done = self.day == self.max_step
            state = self.get_state()
            if self.enable_logging:
                print(f"Today: {self.timestamp_array[today]}, Price: {cur_price}, Reward: {reward}, Total asset: {self.total_asset}")
                print("==========Action End==========")
            return state, float(reward), done, False, dict()

    def _step_v3(self, action):
        pre_price = self.price_ary[self.day][0]
        yesterday = self.day
        self.day += 1
        today = self.day
        cur_price = self.price_ary[today][0]

        if self.cool_down != 0 or action == 2:
            # hold
            original_coolDown = self.cool_down
            if self.cool_down != 0:
                self.cool_down -= 1
            reward = self.stocks * (cur_price - pre_price) * self.reward_scaling
            state = self.get_state()
            done = self.day == self.max_step
            self.gamma_reward = self.gamma_reward * self.gamma + reward
            total_asset = self.amount + self.stocks * cur_price
            if self.enable_logging:
                print(f"today is: {self.timestamp_array[today]}, cool_down is: {original_coolDown}, action is: holding, holding stocks: {self.stocks}, holding cash: {self.amount}, cur_price: {cur_price}, pre_price: {pre_price}, reward: {reward}, total asset: {total_asset}")
            return state, float(reward), done, False, dict()
        if action == 0:
            # buy
            buy_num_shares = self.amount // (pre_price * (1 + self.buy_cost_pct))
            operation_fee = buy_num_shares * pre_price * self.buy_cost_pct
            self.amount -= buy_num_shares * pre_price * (1 + self.buy_cost_pct)
            self.stocks += buy_num_shares
            reward = (self.stocks * (cur_price - pre_price) - operation_fee) * self.reward_scaling
            state = self.get_state()
            done = self.day == self.max_step
            self.gamma_reward = self.gamma_reward * self.gamma + reward
            total_asset = self.amount + self.stocks * cur_price
            if self.enable_logging:
                print(f"today is: {self.timestamp_array[today]}, action is buy, buy shares is: {buy_num_shares}, holding stocks: {self.stocks}, holding cash: {self.amount}, cur_price: {cur_price}, pre_price: {pre_price}, reward: {reward}, total asset: {total_asset}")
            self.cool_down = 0
            return state, float(reward), done, False, dict()
        elif action == 1:
            # cover
            reward = 0
            state = self.get_state()
            hold_stocks = self.stocks
            self.amount += self.stocks * pre_price
            self.stocks = 0
            done = self.day == self.max_step
            total_asset = self.amount + self.stocks * cur_price
            if self.enable_logging:
                print(f"today is: {self.timestamp_array[today]}, action is sell, sell shares is: {hold_stocks}, holding cash: {self.amount}, total asset: {total_asset}")
            self.cool_down = 0
            return state, float(reward), done, False, dict()

    def step(self, action):
        # return self._step_v1(action)
        return self._step_v4(action)

    def _step_v2(self, action):
        # This action corresponds to yesterday's state.
        pre_price = self.price_ary[self.day][0]
        # This is day/price after action
        yesterday = self.day
        self.day += 1
        cur_price = self.price_ary[self.day][0]

        # simple cool down
        if self.cool_down != 0:
            self.cool_down -= 1
            reward = (self.stocks * (cur_price - pre_price)) * self.reward_scaling
            done = self.day == self.max_step
            self.gamma_reward = self.gamma_reward * self.gamma + reward
            state = self.get_state()
            return state, float(reward), done, False, dict()


        # TODO(tianru): Add Logging
        if self.enable_logging:
            print(f"Previous date: {self.timestamp_array[yesterday]}, hold stocks: {self.stocks}, hold cash: {self.amount}, price: {pre_price}, total asset: {self.total_asset}")
        if action == 0:
            # buy as many as possible
            buy_num_shares = self.amount // (pre_price * (1 + self.buy_cost_pct))
            # operation fee to avoid frequent buy/sell
            if self.stocks >= 0:
                cover_shares = 0
            else:
                cover_shares = -self.stocks
            operation_shares = buy_num_shares - cover_shares
            operation_fee = operation_shares * pre_price * self.buy_cost_pct

            self.stocks += buy_num_shares
            self.amount -= (buy_num_shares * pre_price + operation_fee)

            reward = (self.stocks * (cur_price - pre_price) - operation_fee) * self.reward_scaling
            state = self.get_state()
            self.total_asset = self.amount + self.stocks * cur_price
            self.gamma_reward = self.gamma_reward * self.gamma + reward
            done = self.day == self.max_step
            # TODO(tianru): Add Logging
            if self.enable_logging:
                print(
                    f"Based on previous day's state, action is buy {buy_num_shares} shares, hold stocks: {self.stocks}, hold cash: {self.amount}, price: {cur_price}, reward: {reward / self.reward_scaling}, total asset: {self.total_asset}")
            self.cool_down = 5
            return state, float(reward), done, False, dict()
        elif action == 1:
            # sell as many as possible
            original_shares = self.stocks
            margin_requirement = -self.stocks * pre_price
            pre_asset = self.total_asset
            while margin_requirement < pre_asset:
                cover_unit = 100
                self.stocks -= cover_unit
                self.amount += pre_price * cover_unit
                margin_requirement = -self.stocks * pre_price
            sold_stocks = original_shares - self.stocks

            # only charge operation fees for short, cover long is free
            if original_shares <= 0:
                short_stocks = sold_stocks
            else:
                short_stocks = sold_stocks - original_shares # -self.stocks
            operation_fee = 0
            if short_stocks > 0:
                operation_fee = short_stocks * pre_price * self.sell_cost_pct
            reward = (self.stocks * (cur_price - pre_price) - operation_fee) * self.reward_scaling
            self.amount -= operation_fee
            self.total_asset = self.amount + self.stocks * cur_price
            margin_requirement = -self.stocks * cur_price
            while margin_requirement > self.total_asset:
                cover_unit = 100
                self.stocks += cover_unit
                self.amount -= cur_price * cover_unit
                margin_requirement = -self.stocks * cur_price
            state = self.get_state()
            self.gamma_reward = self.gamma_reward * self.gamma + reward
            done = self.day == self.max_step
            # TODO(tianru): Add Logging
            if self.enable_logging:
                print(
                    f"Based on previous day's state, action is sell {original_shares - self.stocks} shares, hold stocks: {self.stocks}, hold cash: {self.amount}, price: {cur_price}, reward: {reward / self.reward_scaling}, total asset: {self.total_asset}")
            self.cool_down = 5
            return state, float(reward), done, False, dict()
        elif action == 2:
            if self.stocks >= 0:
                self.stocks = self.stocks
                self.amount = self.amount

                reward = self.stocks * (cur_price - pre_price) * self.reward_scaling
                state = self.get_state()
                self.total_asset = self.amount + self.stocks * cur_price
                self.gamma_reward = self.gamma_reward * self.gamma + reward
                done = self.day == self.max_step
                # TODO(tianru): Add Logging
                if self.enable_logging:
                    print(
                        f"Based on previous day's state, action is hold shares, hold stocks: {self.stocks}, hold cash: {self.amount}, price: {cur_price}, reward: {reward / self.reward_scaling}, total asset: {self.total_asset}")
                return state, float(reward), done, False, dict()
            else:
                margin_requirement = -self.stocks * cur_price
                cur_asset = self.amount + self.stocks * cur_price
                while margin_requirement > cur_asset:
                    # buy 100 stocks each time
                    cover_unit = 100
                    self.stocks += cover_unit
                    self.amount -= cover_unit * cur_price
                    margin_requirement = -self.stocks * cur_price

                reward = self.stocks * (cur_price - pre_price) * self.reward_scaling
                state = self.get_state()
                self.total_asset = self.amount + self.stocks * cur_price
                self.gamma_reward = self.gamma_reward * self.gamma + reward
                done = self.day == self.max_step
                # TODO(tianru): Add Logging
                if self.enable_logging:
                    print(
                        f"Based on previous day's state, action is hold shares, hold stocks: {self.stocks}, hold cash: {self.amount}, price: {cur_price}, reward: {reward / self.reward_scaling}, total asset: {self.total_asset}")
                return state, float(reward), done, False, dict()
        elif action == 3:
            # cover is free
            reward = 0
            cover_shares = self.stocks
            self.stocks = 0
            self.amount = self.total_asset
            self.gamma_reward = self.gamma_reward * self.gamma + reward
            state = self.get_state()
            done = self.day == self.max_step
            if self.enable_logging:
                print(
                    f"Based on previous day's state, action is cover shares, cover shares: {cover_shares}, hold cash: {self.amount}, price: {cur_price}, reward: {reward / self.reward_scaling}, total asset: {self.total_asset}")
            self.cool_down = 5
            return state, float(reward), done, False, dict()


    def _step_v1(self, action):
        # This action corresponds to yesterday's state.
        pre_price = self.price_ary[self.day][0]
        action_units = self.total_asset // pre_price // 100 // self.allocations * 100
        # This is day/price after action
        yesterday = self.day
        self.day += 1
        cur_price = self.price_ary[self.day][0]

        # TODO(tianru): Add Logging
        if self.enable_logging:
            print(f"Previous date: {self.timestamp_array[yesterday]}, hold stocks: {self.stocks}, hold cash: {self.amount}, price: {pre_price}, total asset: {self.total_asset}")
        # buy stocks
        transaction_shares = 0
        action_str = ""
        if action == 0:
            # this is bought based on yesterday's price
            buy_num_shares = min(self.amount // pre_price, action_units)
            transaction_shares = buy_num_shares
            action_str = "buy"
            self.stocks += buy_num_shares
            self.amount -= buy_num_shares * pre_price
        # sell stocks
        elif action == 1:
            sell_num_shares = min(self.stocks, action_units)
            transaction_shares = sell_num_shares
            action_str = "sell"
            self.stocks -= sell_num_shares
            self.amount += sell_num_shares * pre_price
        # hold stocks
        else:
            action_str = "watch"
            self.stocks = self.stocks
            self.amount = self.amount

        reward = self.stocks * (cur_price - pre_price)
        state = self.get_state()
        self.total_asset = self.amount + self.stocks * cur_price
        if self.enable_logging:
            print(f"Based on previous day's state, action is {action_str} {transaction_shares} shares, hold stocks: {self.stocks}, hold cash: {self.amount}, price: {cur_price}, reward: {reward}, total asset: {self.total_asset}")
        self.gamma_reward = self.gamma_reward * self.gamma + reward
        done = self.day == self.max_step
        return state, float(reward), done, False, dict()

    def get_state(self):
        state = np.hstack(
            (
                # cash left
                # self.amount * self.amount_scaler,
                # # hold stocks
                # self.stocks * self.stocks_scaler,
                # (close - open) / (high - low)
                (self.close_price_normalized[self.day] - self.open_price_normalized[self.day]) / (self.high_price_normalized[self.day] - self.low_price_normalized[self.day]),
                (self.high_price_normalized[self.day] - max(self.open_price_normalized[self.day], self.close_price_normalized[self.day])) / (self.high_price_normalized[self.day] - self.low_price_normalized[self.day]),
                # # stock close price
                # self.close_price_normalized[self.day],
                # # stock open price
                # self.open_price_normalized[self.day],
                # # stock high price
                # self.high_price_normalized[self.day],
                # # stock low price
                # self.low_price_normalized[self.day],
                # price up/down percentage, take 10 days
                # 0 if self.day == 0 else (self.price_ary[self.day] - self.price_ary[self.day - 1]) / self.price_ary[self.day - 1],
                (self.history_price_array[self.day + self.days_diff] - self.history_price_array[
                    self.day + self.days_diff - 1]) / self.history_price_array[self.day + self.days_diff - 1],
                (self.history_price_array[self.day + self.days_diff - 1] - self.history_price_array[
                    self.day + self.days_diff - 2]) / self.history_price_array[self.day + self.days_diff - 2],
                (self.history_price_array[self.day + self.days_diff - 2] - self.history_price_array[
                    self.day + self.days_diff - 3]) / self.history_price_array[self.day + self.days_diff - 3],
                (self.history_price_array[self.day + self.days_diff - 3] - self.history_price_array[
                    self.day + self.days_diff - 4]) / self.history_price_array[self.day + self.days_diff - 4],
                (self.history_price_array[self.day + self.days_diff - 4] - self.history_price_array[
                    self.day + self.days_diff - 5]) / self.history_price_array[self.day + self.days_diff - 5],
                (self.history_price_array[self.day + self.days_diff - 5] - self.history_price_array[
                    self.day + self.days_diff - 6]) / self.history_price_array[self.day + self.days_diff - 6],
                (self.history_price_array[self.day + self.days_diff - 6] - self.history_price_array[
                    self.day + self.days_diff - 7]) / self.history_price_array[self.day + self.days_diff - 7],
                (self.history_price_array[self.day + self.days_diff - 7] - self.history_price_array[
                    self.day + self.days_diff - 8]) / self.history_price_array[self.day + self.days_diff - 8],
                (self.history_price_array[self.day + self.days_diff - 8] - self.history_price_array[
                    self.day + self.days_diff - 9]) / self.history_price_array[self.day + self.days_diff - 9],
                (self.history_price_array[self.day + self.days_diff - 9] - self.history_price_array[
                    self.day + self.days_diff - 10]) / self.history_price_array[self.day + self.days_diff - 10],
                # volume up/down percentage, take 10 days
                # 0 if self.day == 0 else (self.volume_array[self.day] - self.volume_array[self.day - 1]) / self.volume_array[self.day - 1],
                (self.history_volume_array[self.day + self.days_diff] - self.history_volume_array[
                    self.day + self.days_diff - 1]) / self.history_volume_array[self.day + self.days_diff - 1],
                (self.history_volume_array[self.day + self.days_diff - 1] - self.history_volume_array[
                    self.day + self.days_diff - 2]) / self.history_volume_array[self.day + self.days_diff - 2],
                (self.history_volume_array[self.day + self.days_diff - 2] - self.history_volume_array[
                    self.day + self.days_diff - 3]) / self.history_volume_array[self.day + self.days_diff - 3],
                (self.history_volume_array[self.day + self.days_diff - 3] - self.history_volume_array[
                    self.day + self.days_diff - 4]) / self.history_volume_array[self.day + self.days_diff - 4],
                (self.history_volume_array[self.day + self.days_diff - 4] - self.history_volume_array[
                    self.day + self.days_diff - 5]) / self.history_volume_array[self.day + self.days_diff - 5],
                (self.history_volume_array[self.day + self.days_diff - 5] - self.history_volume_array[
                    self.day + self.days_diff - 6]) / self.history_volume_array[self.day + self.days_diff - 6],
                (self.history_volume_array[self.day + self.days_diff - 6] - self.history_volume_array[
                    self.day + self.days_diff - 7]) / self.history_volume_array[self.day + self.days_diff - 7],
                (self.history_volume_array[self.day + self.days_diff - 7] - self.history_volume_array[
                    self.day + self.days_diff - 8]) / self.history_volume_array[self.day + self.days_diff - 8],
                (self.history_volume_array[self.day + self.days_diff - 8] - self.history_volume_array[
                    self.day + self.days_diff - 9]) / self.history_volume_array[self.day + self.days_diff - 9],
                (self.history_volume_array[self.day + self.days_diff - 9] - self.history_volume_array[
                    self.day + self.days_diff - 10]) / self.history_volume_array[self.day + self.days_diff - 10],
                # RSI value, take 10 values
                # self.rsi_array_normalized[self.day],
                self.history_rsi_array[self.day + self.days_diff],
                self.history_rsi_array[self.day + self.days_diff - 1],
                self.history_rsi_array[self.day + self.days_diff - 2],
                self.history_rsi_array[self.day + self.days_diff - 3],
                self.history_rsi_array[self.day + self.days_diff - 4],
                self.history_rsi_array[self.day + self.days_diff - 5],
                self.history_rsi_array[self.day + self.days_diff - 6],
                self.history_rsi_array[self.day + self.days_diff - 7],
                self.history_rsi_array[self.day + self.days_diff - 8],
                self.history_rsi_array[self.day + self.days_diff - 9],

                # VIX value, take 3 values
                self.vix_normalized[self.day],
                self.history_turbulence_array_normalized[self.day + self.days_diff - 1],
                self.history_turbulence_array_normalized[self.day + self.days_diff - 2],
                self.history_turbulence_array_normalized[self.day + self.days_diff - 3],
                self.history_turbulence_array_normalized[self.day + self.days_diff - 4],
                # VIX up/down percentage
                # 0 if self.day == 0 else (self.vix_arr[self.day] - self.vix_arr[self.day - 1]) / self.vix_arr[self.day - 1],
                # self.price_ary[self.day] * 1e-3,
                # self.tech_ary_normalized[self.day],
                (self.price_ary[self.day][0] - self.sma_array[self.day]) / self.price_ary[self.day][0],

                # is cool down allowed
                # 1 if self.cool_down == 0 else 0,
                # is holding stocks
                # 0 if self.stocks > 0 else 1
            )
        )  # state.astype(np.float32)
        return state

    @staticmethod
    def sigmoid_sign(ary, thresh):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x * np.e)) - 0.5

        return sigmoid(ary / thresh) * thresh
