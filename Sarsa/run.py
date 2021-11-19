from env_bird import Env
from Sarsa_learning import ReinforceLearning
import pandas as pd

train_round = 7000  # 训练回合数
game_round = 500  # 演示回合数


def none_begin():
    train_end = False
    file_save = False
    if train_round == 0:
        train_end = True
        file_save = True
    for episode in range(train_round + game_round):
        observation = env.reset(train_end)

        action = RL.choose_action(str(observation))
        if episode % 100 == 0 or train_end:
            print('Game Round:', episode)
        if episode > train_round:
            train_end = True
            RL.epsilon = 1  # 训练结束不进行探索
            if not file_save:
                RL.q_table.to_excel("train_round-"+str(train_round)+".xlsx", encoding="utf-8")
                file_save = True
        while True:
            if not train_end:
                env.render(0)
            else:
                env.render(0.1)

            observation_, reward, done = env.one_step(action, train_end)

            action_ = RL.choose_action(str(observation_))

            if not train_end:  # 训练结束不进行更新
                RL.q_learning(str(observation), action, reward, str(observation_), action_, done)

            observation = observation_
            action = action_

            if done:
                break
        if episode == train_round - 1:
            print(RL.q_table)
    print('game over')
    env.destroy()


def read_begin():  # 读取数据开始
    global train_round
    train_round = 0
    test = pd.DataFrame(pd.read_excel("./train_round-7000.xlsx", index_col=0))
    for index, row in test.iterrows():
        RL.q_table = RL.q_table.append(
            pd.Series(
                [0] * len(RL.actions),
                index=RL.q_table.columns,
                name=row.name
            )
        )
        RL.q_table.loc[row.name, 0] = row[0]
        RL.q_table.loc[row.name, 1] = row[1]

    none_begin()


if __name__ == "__main__":
    env = Env()
    RL = ReinforceLearning(actions=list(range(env.n_actions)))
    env.after(50, read_begin)  # 读取训练7000轮后的数据后开始
    # env.after(50, none_begin)  # 训练 train_round 后再开始
    env.mainloop()
