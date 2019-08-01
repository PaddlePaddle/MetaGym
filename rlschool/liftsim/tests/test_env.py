from rlschool.liftsim.environment.mansion.utils import MansionAttribute, MansionState
from rlschool.liftsim.environment.env import LiftSim

env = LiftSim()
env.seed(1998)
#iteration = env.iterations
state = env.reset()
action = [0,1,0,1,0,1,0,1]
for i in range(100):
    next_state, reward, _, _ = env.step(action)

assert isinstance(env.attribute, MansionAttribute)
assert isinstance(env.state, MansionState)
print(env.statistics)
env.log_debug("This is a debug log")
env.log_notice("This is a notice log")
