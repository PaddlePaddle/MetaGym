from environment.mansion.utils import MansionAttribute, MansionState
from environment.mansion.utils import ElevatorAction
from environment.env import environmentEnv

env = environmentEnv()
env.seed(1998)
#iteration = env.iterations
state = env.reset()
action = [ElevatorAction(0, 1) for i in range(4)]
for i in range(100):
    next_state, reward, _, _ = env.step(action)

assert isinstance(env.attribute, MansionAttribute)
assert isinstance(env.state, MansionState)
print(env.statistics)
env.log_debug("This is a debug log")
env.log_notice("This is a notice log")
