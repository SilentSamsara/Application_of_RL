import numpy as np

state_action = np.array([0, 1, 2, 3])
print(state_action[state_action == np.max(state_action)].index)

