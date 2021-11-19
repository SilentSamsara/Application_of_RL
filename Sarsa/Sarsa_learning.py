import numpy as np
import pandas as pd


class ReinforceLearning(object):
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)

    def read_q_table(self):
        self.q_table = pd.read_excel("./train_round-7000.xlsx")

    def check_state_exist(self, state):
        if state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=state
                )
            )

    def choose_action(self, observation):
        self.check_state_exist(observation)

        if np.random.rand() < self.epsilon:
            state_action = self.q_table.loc[observation, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(self.actions)
        return action

    def sarsa(self, state, action, reward, next_state, next_action, done):
        self.check_state_exist(next_state)
        q_predict = self.q_table.loc[state, action]
        if not done:
            q_target = reward + self.gamma * self.q_table.loc[next_state, next_action]
        else:
            q_target = reward
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)

    def q_learning(self, state, action, reward, next_state, next_action, done):
        self.check_state_exist(next_state)
        q_predict = self.q_table.loc[state, action]
        if not done:
            q_target = reward + self.gamma * max(self.q_table.loc[next_state, :])
        else:
            q_target = reward
        self.q_table.loc[state, action] += self.lr * (q_target - q_predict)
