from env_bird import Env
from Sarsa_learning import ReinforceLearning

train_round = 7000
game_round = 500


def begin():
    train_end = False
    for episode in range(train_round+game_round):
        observation = env.reset(train_end)

        action = RL.choose_action(str(observation))
        if episode % 100 == 0 or train_end:
            print('Game Round:', episode)
        if episode > train_round:
            train_end = True
            RL.epsilon = 0.99
        while True:
            if not train_end:
                env.render(0)
            else:
                env.render(0.1)

            observation_, reward, done = env.one_step(action, train_end)

            action_ = RL.choose_action(str(observation_))

            RL.sarsa(str(observation), action, reward, str(observation_), action_, env)

            observation = observation_
            action = action_

            if done:
                break

    print('game over')
    env.destroy()


if __name__ == "__main__":
    env = Env()
    RL = ReinforceLearning(actions=list(range(env.n_actions)))
    env.after(50, begin)
    env.mainloop()
