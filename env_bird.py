import numpy as np
import time
import tkinter as tk

UNIT = 40
MAP_H = 10
MAP_W = 15
obstacle_right = [0, 0]
black = [[0] * 10 for i in range(20)]
# 格子的坐标原点（左上角）
origin = np.array([UNIT / 2, UNIT / 2])


class Env(tk.Tk, object):

    def __init__(self,):
        super(Env, self).__init__()
        self.end = False
        self.action_space = ['f', 'd']
        self.n_actions = len(self.action_space)
        self.title('Bird fly')
        self.geometry('{0}x{1}'.format(MAP_W * UNIT, MAP_H * UNIT))
        self.obstacle_right = [0, 0]
        self.canvas = tk.Canvas(self, bg='white',
                                height=MAP_H * UNIT,
                                width=MAP_W * UNIT)
        self.well_num = 0
        self.__build_env()

    def __build_env(self):
        gap = int(np.random.uniform(0, 7.9))  # 缝隙
        for i in range(0, 10):
            a = i - gap
            if not (0 <= a < 3):
                black[MAP_W-2][i] = 1
                black[MAP_W-1][i] = 1
            if a == 3:
                self.obstacle_right = [MAP_W-1, i]
        if gap == 7:
            self.obstacle_right = [MAP_W - 1, 10]
        self.bird = [2, 4]
        self.draw_game()
        self.canvas.pack()

    def draw_game(self):
        self.canvas.delete("all")
        for w in range(0, MAP_W * UNIT, UNIT):
            x0, y0, x1, y1 = w, 0, w, MAP_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for h in range(0, MAP_W * UNIT, UNIT):
            x0, y0, x1, y1 = 0, h, MAP_W * UNIT, h
            self.canvas.create_line(x0, y0, x1, y1)
        for i in range(len(black)):
            for j in range(len(black[i])):
                if black[i][j]:
                    self.canvas.create_rectangle(
                        origin[0] + i * UNIT - 15, origin[1] + j * UNIT - 15,
                        origin[0] + i * UNIT + 15, origin[1] + j * UNIT + 15,
                        fill='black')
        self.canvas.create_oval(
            origin[0] + self.bird[0] * UNIT - 15, origin[1] + self.bird[1] * UNIT - 15,
            origin[0] + self.bird[0] * UNIT + 15, origin[1] + self.bird[1] * UNIT + 15,
            fill='red')
        self.canvas.update()

    def reset(self, train_end):
        self.update()
        if train_end:
            time.sleep(0.5)
        for i in range(len(black)):
            for j in range(len(black[i])):
                black[i][j] = 0
        self.__build_env()
        self.canvas.delete("all")
        self.draw_game()
        return [self.obstacle_right[0] - self.bird[0], self.obstacle_right[1] - self.bird[1]]

    def one_step(self, action, train_end=False):
        reward = 0
        if black[0][0] == 1 or black[0][9] == 1:
            gap = int(np.random.uniform(0, 7.9))  # 缝隙
            if train_end:
                print('well done', self.well_num)
            self.well_num += 1
            reward = 10  # 通过柱子奖励
            for i in range(0, 10):
                a = i - gap
                black[0][i] = 0
                black[1][i] = 0
                if not (0 <= a < 3):
                    black[MAP_W-2][i] = 1
                    black[MAP_W-1][i] = 1
                if a == 3:
                    self.obstacle_right = [MAP_W-1, i]
            if gap == 7:
                self.obstacle_right = [MAP_W-1, 10]
        step = black.pop(0)
        black.append(step)
        if action == 0 and self.bird[1] > 0:
            self.bird[1] = self.bird[1] - 1
        elif action == 1 and self.bird[1] < 9:
            self.bird[1] = self.bird[1] + 1
        self.draw_game()
        self.obstacle_right[0] = self.obstacle_right[0] - 1
        next_state = [self.obstacle_right[0] - self.bird[0], self.obstacle_right[1] - self.bird[1]]
        if black[self.bird[0]][self.bird[1]]:  # 撞上柱子
            reward = -100
            done = True
            self.end = True
            # print(next_state)
            # next_state = 'terminal'
        else:  # 存活奖励
            reward = reward + 1
            done = False
        return next_state, reward, done

    def render(self, stop_time=0.01):
        time.sleep(stop_time)
        self.update()

