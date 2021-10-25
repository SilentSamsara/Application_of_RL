import matplotlib.pyplot as plt
import numpy as np
import game_environment
import RL_learning

rounds = 50000
data = np.arange(1, rounds + 1, 1.0)


def begin():
    ep = 0
    for r in range(rounds):
        env.draw_canvas()
        env.render(False)

        env.one_time_step()
        for i in range(len(env.agents)):
            for j in range(len(env.agents[i])):
                RL.choose_action(env.agents[i][j], env.agents)

        # for i in range(len(env.agents)):
        #     for j in range(len(env.agents[i])):
        #         print(env.agents[i][j].this_action, end=" | ")
        #     print()

        cooperation_rate, intelligence = env.set_next_reward()

        for i in range(len(env.agents)):
            for j in range(len(env.agents[i])):
                if env.agents[i][j].is_intelligence:
                    RL.q_learning(env.agents[i][j])

        data[r] = cooperation_rate
        ep += 1
        print(ep, intelligence, cooperation_rate)
    analysis()


def analysis():
    x = np.arange(1, rounds + 1, 1)
    print(data)
    if rounds < 10000:
        plt.title(str('ρ=' + str(env.rho) + ', DS=' + str(env.DS)))
    else:
        fig, ax1 = plt.subplots(1, 1)
        ax1.set_xscale("log")
        ax1.set_xlim(1, len(data))
        ax1.set_aspect(1)
        ax1.set_title(str('ρ=' + str(env.rho) + ', DS=' + str(env.DS)))
    plt.plot(x, data, label=str('ρ=' + str(env.rho) + ', DS=' + str(env.DS)))
    plt.legend()
    plt.yticks(np.arange(0, 1.05, 0.05))
    plt.xticks(x)
    plt.xlabel('round')
    plt.ylabel('Cooperation rate')
    plt.show()


if __name__ == '__main__':
    env = game_environment.GameEnv()
    RL = RL_learning.ReinforceLearning(actions=game_environment.Q_actions_space)
    env.after(0, begin)
    env.mainloop()
