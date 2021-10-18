import numpy as np
import math
import pandas as pd


def check_state_exist(state, agent):
    if state not in agent.q_table.index:
        agent.q_table = agent.q_table.append(
            pd.Series(
                [0] * len(agent.actions),
                index=agent.q_table.columns,
                name=state
            )
        )


K = 0.1
L = 100


class ReinforceLearning(object):
    def __init__(self, actions, learning_rate=0.5, reward_decay=0.9, e_greedy=0.02):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = e_greedy

    def choose_action(self, agent, agents):
        check_state_exist(agent.state, agent)
        if np.random.rand() > self.epsilon:
            state_action = agent.q_table.loc[agent.state, :]
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            action = np.random.choice(agent.actions)

        i = agent.study_x
        j = agent.study_y

        # 费米函数：是否进行学习并采取动作
        w = 1 / (1 + math.exp((agents[i][j].this_reward -
                               agents[agents[i][j].study_x][agents[i][j].study_y].this_reward) / K))
        if w > 0.5:
            # max(self.agents[i][j].q_table.loc[self.agents])
            # agents[i][j].is_collaborator = agents[agents[i][j].study_x][
            #     agents[i][j].study_y].is_collaborator
            if action == 0:
                agent.is_collaborator = agents[(agent.x - 1) % L][agent.y].is_collaborator
            if action == 1:
                agent.is_collaborator = agents[agent.x][(agent.y - 1) % L].is_collaborator
            if action == 2:
                agent.is_collaborator = agents[(agent.x + 1) % L][agent.y].is_collaborator
            if action == 3:
                agent.is_collaborator = agents[agent.x][(agent.y + 1) % L].is_collaborator
        agent.next_action = action

    def q_learning(self, agent, agents):
        agent.check_state_exist()
        q_predict = agent.q_table.loc[agent.state, agent.action]
        if np.random.rand() > self.epsilon:
            q_target = agent.this_reward + self.gamma * agents[agent.study_x][agent.study_y].this_reward
        else:
            q_target = agent.this_reward
        agent.q_table.loc[agent.state, agent.action] += self.lr * (q_target - q_predict)
