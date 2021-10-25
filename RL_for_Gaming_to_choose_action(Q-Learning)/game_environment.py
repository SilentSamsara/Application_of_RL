import time
import numpy as np
import pandas as pd
import tkinter as tk

UNIT = 6  # 像素边长
L = 100  # 网络规模: L × L
DS = 0.02  # 困境强度
D_g = DS  # D_g = T - R
D_r = DS  # D_r = P - S
rho = 0.1  # 真·智能体占比
init_collaborator = 0.3  # 初始合作者占比

'''
收益矩阵:
 R S
 T P
R:合作奖励
S:受骗成本
T:背叛诱惑
P:相互背叛

取值如下：
'''
R = 1
S = -D_r
T = 1 + D_g
P = 0
Q_actions_space = ['C', 'D']


def game_reward(agent1, agent2):
    if agent1.is_collaborator:
        if agent2.is_collaborator:
            return R
        else:
            return S
    else:
        if agent2.is_collaborator:
            return T
        else:
            return P


class Agent(object):
    def __init__(self, x, y, actions_space, collaborator=False):
        self.location_x = x
        self.location_y = y
        self.this_state = 0
        self.this_action = 0
        self.this_reward = 0
        self.next_state = 0
        self.next_action = 0
        self.next_reward = 0
        self.total_reward = 0
        self.is_intelligence = False
        self.is_collaborator = collaborator
        self.q_table = pd.DataFrame(columns=actions_space, dtype=np.float64)  # Q表（每个智能体单独一张Q表）

    def check_next_state(self):
        if self.next_state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(Q_actions_space),
                    index=self.q_table.columns,
                    name=self.next_state
                )
            )


class GameEnv(tk.Tk, object):

    def __init__(self):
        super(GameEnv, self).__init__()
        self.title('Choose Action')
        self.canvas = tk.Canvas(self, bg='white',
                                height=L * UNIT,
                                width=L * UNIT)
        self.rho = rho
        self.DS = DS
        self.agents = [[Agent(i, j, Q_actions_space) for i in range(L)] for j in range(L)]
        for i in range(L):
            for j in range(L):
                if np.random.rand() < self.rho:
                    self.agents[i][j].is_intelligence = True
        self.__build_env()
        print('init over')

    def __build_env(self):
        origin = [int(L / 2), int(L / 2)]
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                if np.random.rand() < init_collaborator:
                    self.agents[i][j].next_action = 'C'
                    self.agents[i][j].is_collaborator = True
                else:
                    self.agents[i][j].next_action = 'D'

                if self.agents[i][j].is_intelligence and self.agents[i][j].is_collaborator:
                    self.canvas.create_rectangle(
                        j * UNIT, i * UNIT,
                        j * UNIT + UNIT - 1, i * UNIT + UNIT - 1,
                        outline='red',
                        fill='blue')
                elif self.agents[i][j].is_intelligence:
                    self.canvas.create_rectangle(
                        j * UNIT, i * UNIT,
                        j * UNIT + UNIT - 1, i * UNIT + UNIT - 1,
                        outline='red',
                        fill='white')
                elif self.agents[i][j].is_collaborator:
                    self.canvas.create_rectangle(
                        j * UNIT, i * UNIT,
                        j * UNIT + UNIT - 1, i * UNIT + UNIT - 1,
                        outline='blue',
                        fill='blue')
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                self.agents[i][j].next_state = str(self.get_cooperation_num(self.agents[i][j].location_x,
                                                                            self.agents[i][j].location_y))
                self.agents[i][j].check_next_state()
                self.agents[i][j].this_state = self.agents[i][j].next_state
        self.canvas.pack()

    def draw_canvas(self):
        self.canvas.delete("all")
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                if self.agents[i][j].is_intelligence and self.agents[i][j].is_collaborator:
                    self.canvas.create_rectangle(
                        j * UNIT, i * UNIT,
                        j * UNIT + UNIT - 1, i * UNIT + UNIT - 1,
                        outline='red',
                        fill='blue')
                elif self.agents[i][j].is_intelligence:
                    self.canvas.create_rectangle(
                        j * UNIT, i * UNIT,
                        j * UNIT + UNIT - 1, i * UNIT + UNIT - 1,
                        outline='red',
                        fill='white')
                elif self.agents[i][j].is_collaborator:
                    self.canvas.create_rectangle(
                        j * UNIT, i * UNIT,
                        j * UNIT + UNIT - 1, i * UNIT + UNIT - 1,
                        outline='blue',
                        fill='blue')
        self.canvas.pack()

    def render(self, sp=True):
        if sp:
            time.sleep(0.5)
        self.update()

    def get_cooperation_num(self, x, y):
        num = 0
        if self.agents[(x - 1) % L][y].is_collaborator:
            num += 1
        if self.agents[x][(y - 1) % L].is_collaborator:
            num += 1
        if self.agents[(x + 1) % L][y].is_collaborator:
            num += 1
        if self.agents[x][(y + 1) % L].is_collaborator:
            num += 1
        return num / 4

    def one_time_step(self):
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                self.agents[i][j].total_reward += self.agents[i][j].this_reward
                self.agents[i][j].this_reward = self.agents[i][j].next_reward
                self.agents[i][j].this_state = self.agents[i][j].next_state
                self.agents[i][j].this_action = self.agents[i][j].next_action

    def set_next_reward(self):
        cooperator = 0
        intelligence = 0
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                if self.agents[i][j].is_collaborator:
                    cooperator += 1
                if self.agents[i][j].is_intelligence:
                    intelligence += 1
                self.agents[i][j].next_reward = 0
                self.agents[i][j].next_reward += game_reward(self.agents[i][j], self.agents[(i - 1) % L][j])
                self.agents[i][j].next_reward += game_reward(self.agents[i][j], self.agents[i][(j - 1) % L])
                self.agents[i][j].next_reward += game_reward(self.agents[i][j], self.agents[(i + 1) % L][j])
                self.agents[i][j].next_reward += game_reward(self.agents[i][j], self.agents[i][(j + 1) % L])
                self.agents[i][j].next_state = str((self.get_cooperation_num(i, j)))
        return cooperator / (L * L), intelligence

    def get_cooperation_rate(self):
        cooperator = 0
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                if self.agents[i][j].is_collaborator:
                    cooperator += 1
        return cooperator / (L*L)