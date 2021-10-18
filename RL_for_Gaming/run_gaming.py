import game_env
import RL_learning


def begin():
    ep = 0
    while True:
        env.render(False)
        # 累计收益
        env.one_time_step()

        # 获取动作
        for i in range(len(env.agents)):
            for j in range(len(env.agents[i])):
                RL.choose_action(env.agents[i][j], env.agents)

        # 获取当轮收益
        env.set_next_reward()

        # 确定学习对象
        env.determine_learning_objectives()

        for i in range(len(env.agents)):
            for j in range(len(env.agents[i])):
                RL.q_learning(env.agents[i][j], env.agents)
        env.draw_canvas()

        # for i in range(len(env.agents)):
        #     for j in range(len(env.agents[i])):
        #         print(env.agents[i][j].state, end=" | ")
        #     print()

        ep += 1
        print(ep)


if __name__ == "__main__":
    env = game_env.GameEnv()
    RL = RL_learning.ReinforceLearning(actions=list(range(len(env.action_space))))
    env.after(50, begin)
    env.mainloop()
