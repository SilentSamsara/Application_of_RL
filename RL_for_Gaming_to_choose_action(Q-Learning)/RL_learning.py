import math
import numpy as np

L = 100
K = 0.1


class ReinforceLearning(object):
    def __init__(self, actions, learning_rate=0.1, reward_decay=0.9, e_greedy=0.02):
        self.actions = actions
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon = 1 - e_greedy

    def choose_action(self, agent, agents):
        if agent.is_intelligence:
            agent.check_next_state()
            if np.random.rand() < self.epsilon:
                state_action = agent.q_table.loc[agent.this_state, :]
                action = np.random.choice(state_action[state_action == np.max(state_action)].index)
            else:
                action = np.random.choice(self.actions)
            agent.next_action = action
            if action == 'C':
                agent.is_collaborator = True
            else:
                agent.is_collaborator = False
        else:
            # 费米更新规则
            get_study_action = np.random.choice([0, 1, 2, 3])
            if get_study_action == 0:
                study_ob = agents[(agent.location_x - 1) % L][agent.location_y]
            elif get_study_action == 1:
                study_ob = agents[agent.location_x][(agent.location_y - 1) % L]
            elif get_study_action == 2:
                study_ob = agents[(agent.location_x + 1) % L][agent.location_y]
            elif get_study_action == 3:
                study_ob = agents[agent.location_x][(agent.location_y + 1) % L]
            if np.random.rand() <= 1 / (1 + math.exp((agent.next_reward - study_ob.next_reward)/K)):
                agent.is_collaborator = study_ob.is_collaborator

    def q_learning(self, agent):
        agent.check_next_state()
        q_predict = agent.q_table.loc[agent.this_state, agent.this_action]
        if np.random.rand() < self.epsilon:
            q_target = agent.this_reward + self.gamma * max(agent.q_table.loc[agent.next_state, :])
        else:
            q_target = agent.this_reward
        agent.q_table.loc[agent.this_state, agent.this_action] += self.lr * (q_target - q_predict)
