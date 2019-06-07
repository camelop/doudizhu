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

# a1_model = MultiLevelPerceptron(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(200, 200, 200, 200))
# a1_policy = DQNPolicy(a1_model, seed=0, comment="l4")
# a1 = Agent('Billy', a1_policy)

# a1_model = MultiLevelPerceptron(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(128,), learning_rate=1e-2)
# a1_policy = DQNPolicy(a1_model, seed=0, comment="negd", e_greedy=(0.0, -0.0002))
# a1 = Agent('Cathy8', a1_policy)

# a1_model = MultiLevelPerceptron(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(256, 256), learning_rate=1e-2)
# a1_policy = DQNPolicy(a1_model, seed=0, comment="egd5e-1", e_greedy=(5e-1, -5e-4))
# a1 = Agent('Dove', a1_policy)

# a1_model = MultiLevelPerceptron(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(1024, 1024, 1024, 1024), learning_rate=1e-3)
# a1_policy = DQNPolicy(a1_model, seed=0, comment="egd5e-1", e_greedy=(5e-1, -5e-4))
# a1 = Agent('Emma', a1_policy)

# a1_model = MultiLevelPerceptron(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(2048, 2048, 2048, 2048), learning_rate=1e-3)
# a1_policy = DQNPolicy(a1_model, seed=0, comment="egd5e-1", e_greedy=(5e-1, -5e-4), save_every=100)
# a1 = Agent('Fisher', a1_policy)

# a1_model = MultiLevelPerceptron(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(4096, )*8, learning_rate=1e-3)
# a1_policy = DQNPolicy(a1_model, seed=0, comment="egd4_vsChooseMinWithRole", e_greedy=(4e-1, -1e-4), save_every=500)
# a1 = Agent('Gilly', a1_policy)

a1_model = MultiLevelPerceptron(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(2048, )*4, learning_rate=1e-3)
a1_policy = DQNPolicy(a1_model, seed=0, comment="egd0_vsChooseMinWithRole", e_greedy=(0, 0), save_every=500)
a1 = Agent('Hill', a1_policy)
# a2 = Agent('Lannister', RandomPolicy(seed=1))
# a3 = Agent('Targaryen', RandomPolicy(seed=2))
a2 = Agent('Lazarus', ChooseMinWithRolePolicy(seed=1))
a3 = Agent('Tyrion', ChooseMinWithRolePolicy(seed=2))

sim = Simulator([a1, a2, a3], display=False)
# sim = Simulator([a1, a2, a3], display=True)

time_start=time.time()

# results = sim.run(seeds=list(range(100)), mirror=True, save=True)
results = sim.run(seeds=list(range(5000)), mirror=False, save=True)

time_end=time.time()
print('Time(s)', time_end-time_start)
