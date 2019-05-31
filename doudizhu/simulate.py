import time

from apps.game.simulation import Simulator, Agent
from apps.game.policy.randomPolicy import RandomPolicy
from apps.game.policy.negativePolicy import NegativePolicy
from apps.game.policy.chooseMinPolicy import ChooseMinPolicy
from apps.game.policy.chooseMinWithRolePolicy import ChooseMinWithRolePolicy

from apps.game.policy.DRL.MLP import MultiLevelPerceptron
from apps.game.policy.DQNPolicy import DQNPolicy

# a1 = Agent('Stark', RandomPolicy(seed=0))
# a1 = Agent('Tully', ChooseMinWithRolePolicy(seed=0))

# a1_model = MultiLevelPerceptron(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(1000, 1000))
# a1_policy = DQNPolicy(a1_model, seed=0, comment="default")
# a1 = Agent('Apollo', a1_policy)

a1_model = MultiLevelPerceptron(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(1000, 1000), freeze=True)
a1_policy = DQNPolicy(a1_model, seed=0, comment="default", load_tag="DQNPolicy-sv1[MLP-lr(0.001)-h1(1000)-h2(1000)-e(1951)](0).default.v0")
a1 = Agent('Apollo', a1_policy)

a2 = Agent('Lannister', RandomPolicy(seed=1))
a3 = Agent('Targaryen', RandomPolicy(seed=2))

sim = Simulator([a1, a2, a3], display=False)
# sim = Simulator([a1, a2, a3], display=True)

time_start=time.time()

# results = sim.run(seeds=list(range(100)), mirror=True, save=True)
results = sim.run(seeds=list(range(2000)), mirror=False, save=False)

time_end=time.time()
print('Time(s)', time_end-time_start)