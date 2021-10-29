import math
import time
import numpy as np
import pandas as pd
import operator
import tkinter as tk

UNIT = 6  # 像素边长
L = 100  # 网络规模: L × L
D_g = 0.03  # D_g = T - R
D_r = 0  # D_r = P - S

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


class Agent(object):

    def __init__(self, x, y, collaborator=False, cooperation_rate=0, actions=None):
        if actions is None:
            actions = [0, 1, 2, 3]
        self.actions = actions
        self.x = x
        self.y = y
        self.is_collaborator = collaborator  # 是否为合作者。True为合作者，False为背叛者
        self.next_is_collaborator = collaborator
        self.state = cooperation_rate  # 合作率（状态）
        self.action = 0  # 学习对象
        self.this_reward = 0  # 这个时间步的收益
        self.total_reward = 0  # 总收益
        self.study_x = (x - 1) % L
        self.study_y = y % L
        self.next_state = cooperation_rate  # 下个时间步状态
        self.next_action = 0
        self.next_reward = 0  # 下时间步的收益
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)  # Q表（每个智能体单独一张Q表）

    def check_state_exist(self):
        if self.next_state not in self.q_table.index:
            self.q_table = self.q_table.append(
                pd.Series(
                    [0] * len(self.actions),
                    index=self.q_table.columns,
                    name=self.next_state
                )
            )


def game_reward(agent1, agent2):
    if agent1.next_is_collaborator:
        if agent2.next_is_collaborator:
            return R
        else:
            return S
    else:
        if agent2.next_is_collaborator:
            return T
        else:
            return P


class GameEnv(tk.Tk, object):

    def __init__(self):
        super(GameEnv, self).__init__()
        self.action_space = [0, 1, 2, 3]  # 动作空间
        self.title('Gaming')
        self.geometry('{0}x{1}'.format(L * UNIT, L * UNIT))
        self.agents = [[Agent(i, j, actions=self.action_space) for j in range(L)] for i in range(L)]  # 创建LxL图
        self.agents[int(L / 2)][int(L / 2)].is_collaborator = True
        self.canvas = tk.Canvas(self, bg='white',
                                height=L * UNIT,
                                width=L * UNIT)
        self.__build_env()
        print("init over")

    # 初始化可视化界面
    def __build_env(self):
        origin = [int(L / 2), int(L / 2)]
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                # 中间为合作者
                self.agents[i][j].next_action = np.random.choice(self.agents[i][j].actions)
                if math.sqrt((origin[0] - i) * (origin[0] - i) + (origin[1] - j) * (origin[1] - j)) < (L / 18):
                    self.agents[i][j].next_is_collaborator = True
                # 初始随机分布
                # if np.random.rand() < 0.5:
                #     self.agents[i][j].is_collaborator = True
                if self.agents[i][j].next_is_collaborator:
                    self.canvas.create_rectangle(
                        j * UNIT, i * UNIT,
                        j * UNIT + UNIT - 1, i * UNIT + UNIT - 1,
                        outline='blue',
                        fill='blue')
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                self.agents[i][j].next_state = str((self.get_cooperation_num(i, j) / 4))
        self.determine_learning_objectives()
        self.canvas.pack()

    def draw_canvas(self):
        self.canvas.delete("all")
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                if self.agents[i][j].next_is_collaborator:
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

    # 获得智能体的学习对象
    def fill_study_agent(self, agent):
        x = agent.x
        y = agent.y
        agent_list = [self.agents[(x - 1) % L][y],
                      self.agents[x][(y - 1) % L],
                      self.agents[(x + 1) % L][y],
                      self.agents[x][(y + 1) % L]]
        study_agent = max(agent_list, key=operator.attrgetter('next_reward'))
        agent.study_x = study_agent.x
        agent.study_y = study_agent.y

    # 获取合作者个数
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
        # if self.agents[x][y].is_collaborator:
        #     num += 1
        return num

    # 一个时间步
    def one_time_step(self):
        # 计算前一轮的累计收益,确定当前状态
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                self.agents[i][j].total_reward += self.agents[i][j].this_reward
                self.agents[i][j].this_reward = self.agents[i][j].next_reward
                self.agents[i][j].state = self.agents[i][j].next_state
                self.agents[i][j].action = self.agents[i][j].next_action
                self.agents[i][j].is_collaborator = self.agents[i][j].next_is_collaborator

    # 确定学习目标对象
    def determine_learning_objectives(self):
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                self.fill_study_agent(self.agents[i][j])

    # 获取 next_reward（进行博弈）
    def set_next_reward(self):
        cooperator = 0
        for i in range(len(self.agents)):
            for j in range(len(self.agents[i])):
                if self.agents[i][j].next_is_collaborator:
                    cooperator += 1
                self.agents[i][j].next_reward = 0
                self.agents[i][j].next_reward += game_reward(self.agents[i][j], self.agents[(i - 1) % L][j])
                self.agents[i][j].next_reward += game_reward(self.agents[i][j], self.agents[i][(j - 1) % L])
                self.agents[i][j].next_reward += game_reward(self.agents[i][j], self.agents[(i + 1) % L][j])
                self.agents[i][j].next_reward += game_reward(self.agents[i][j], self.agents[i][(j + 1) % L])
                self.agents[i][j].next_state = str((self.get_cooperation_num(i, j) / 4))
        return cooperator / (L * L)

