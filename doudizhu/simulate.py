import time

from apps.game.simulation import Simulator, Agent
from apps.game.policy.randomPolicy import RandomPolicy
from apps.game.policy.negativePolicy import NegativePolicy
from apps.game.policy.chooseMinPolicy import ChooseMinPolicy
from apps.game.policy.chooseMinWithRolePolicy import ChooseMinWithRolePolicy

from apps.game.policy.DRL.DQNMLP import DQNMLP
from apps.game.policy.DQNPolicy import DQNPolicy
from apps.game.policy.SA_DQNPolicy import SA_DQNPolicy

from apps.game.policy.DRL.REINFORCE_MLP import REINFORCE_MLP
from apps.game.policy.PGPolicy import PGPolicy


player = "Kate"
env = 'env1'
epoch_num = 5000  
display = False 
save = True
'''
# test setting
display = True 
save = False
'''


if player == 'Stark':
    a1 = Agent('Stark', RandomPolicy(seed=0))
elif player == 'Tully':
    a1 = Agent('Tully', ChooseMinWithRolePolicy(seed=0))
elif player == 'Apollo':
    a1_model = DQNMLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(1000, 1000))
    a1_policy = DQNPolicy(a1_model, seed=0, comment="default")
    a1 = Agent('Apollo', a1_policy)
elif player == 'Billy':
    a1_model = DQNMLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(200, 200, 200, 200))
    a1_policy = DQNPolicy(a1_model, seed=0, comment="l4")
    a1 = Agent('Billy', a1_policy)
elif player == "Cathy8":
    a1_model = DQNMLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(128,), learning_rate=1e-2)
    a1_policy = DQNPolicy(a1_model, seed=0, comment="negd", e_greedy=(0.0, -0.0002))
    a1 = Agent('Cathy8', a1_policy)
elif player == "Dove":
    a1_model = DQNMLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(256, 256), learning_rate=1e-2)
    a1_policy = DQNPolicy(a1_model, seed=0, comment="egd5e-1", e_greedy=(5e-1, -5e-4))
    a1 = Agent('Dove', a1_policy)
elif player == "Emma":
    a1_model = DQNMLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(1024, 1024, 1024, 1024), learning_rate=1e-3)
    a1_policy = DQNPolicy(a1_model, seed=0, comment="egd5e-1", e_greedy=(5e-1, -5e-4))
    a1 = Agent('Emma', a1_policy)
elif player == "Fisher":
    a1_model = DQNMLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(2048, 2048, 2048, 2048), learning_rate=1e-3)
    a1_policy = DQNPolicy(a1_model, seed=0, comment="egd5e-1", e_greedy=(5e-1, -5e-4), save_every=100)
    a1 = Agent('Fisher', a1_policy)
elif player == "Gilly":
    a1_model = DQNMLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(4096, )*8, learning_rate=1e-3)
    a1_policy = DQNPolicy(a1_model, seed=0, comment="egd4_vsChooseMinWithRole", e_greedy=(4e-1, -1e-4), save_every=500)
    a1 = Agent('Gilly', a1_policy)
elif player == "Hill":
    a1_model = DQNMLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(2048, )*4, learning_rate=1e-3)
    a1_policy = DQNPolicy(a1_model, seed=0, comment="egd0_vsChooseMinWithRole", e_greedy=(0, 0), save_every=500)
    a1 = Agent('Hill', a1_policy)
elif player == "Happy":
    a1_model = DQNMLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(1024, )*2, learning_rate=1e-3, activation="tanh")
    a1_policy = DQNPolicy(a1_model, seed=0, comment="edg4_tanh", e_greedy=(4e-1, -1e-4), save_every=500)
    a1 = Agent('Happy', a1_policy)
elif player == "Happy2":
    a1_model = DQNMLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(2048, )*4, learning_rate=1e-3, activation="tanh")
    a1_policy = DQNPolicy(a1_model, seed=0, comment="edg4_tanh", e_greedy=(4e-1, -1e-4), save_every=500)
    a1 = Agent('Happy2', a1_policy)
elif player == "Isaac":
    a1_model = REINFORCE_MLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(1024, )*2, learning_rate=1e-3, activation="tanh")
    a1_policy = PGPolicy(a1_model, seed=0, comment="REINFORCE", save_every=500)
    a1 = Agent('Isaac', a1_policy)
elif player == "Isaac2":
    a1_model = REINFORCE_MLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(2048, )*4, learning_rate=1e-3, activation="tanh")
    a1_policy = PGPolicy(a1_model, seed=0, comment="REINFORCE", save_every=500)
    a1 = Agent('Isaac2', a1_policy)
elif player == "Isaac3":
    a1_model = REINFORCE_MLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(2048, )*4, learning_rate=1e-3)
    a1_policy = PGPolicy(a1_model, seed=0, comment="REINFORCE-relu", save_every=500)
    a1 = Agent('Isaac3', a1_policy)
elif player == "Imp":
    a1_model = REINFORCE_MLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(4096, )*8, learning_rate=1e-3)
    a1_policy = PGPolicy(a1_model, seed=0, comment="REINFORCE-relu-env2", save_every=500)
    a1 = Agent('Imp', a1_policy)
elif player == "Imp2":
    a1_model = REINFORCE_MLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(2048, )*8, learning_rate=1e-3)
    a1_policy = PGPolicy(a1_model, seed=0, comment="REINFORCE-relu-env2", save_every=500)
    a1 = Agent('Imp2', a1_policy)

#  ---------------------- final tests ---------------------- 

elif player == "James":
    a1_model = DQNMLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(2048, )*8, learning_rate=1e-3)
    a1_policy = DQNPolicy(a1_model, seed=0, comment="final_"+env, e_greedy=(3e-1, -1e-4), save_every=1000)
    a1 = Agent(player, a1_policy)
elif player == "Johnny":
    a1_model = REINFORCE_MLP(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(2048, )*8, learning_rate=1e-3)
    a1_policy = PGPolicy(a1_model, seed=0, comment="final_"+env, save_every=1000)
    a1 = Agent(player, a1_policy)

elif player == "Kate":
    a1_model = DQNMLP(action_dim=DQNPolicy.S_ACTION_DIM, hidden_dims=(2048, )*8, learning_rate=1e-3)
    a1_policy = SA_DQNPolicy(a1_model, seed=0, comment="test_sa_"+env, e_greedy=(3e-1, -1e-4), save_every=1000)
    a1 = Agent(player, a1_policy)
else:
    raise NotImplementedError
    
# available: env1, env2

if env == 'env1':
    a2 = Agent('Lannister', RandomPolicy(seed=1))
    a3 = Agent('Targaryen', RandomPolicy(seed=2))
elif env == 'env2':
    a2 = Agent('Lazarus', ChooseMinWithRolePolicy(seed=1))
    a3 = Agent('Tyrion', ChooseMinWithRolePolicy(seed=2))
else:
    raise NotImplementedError

sim = Simulator([a1, a2, a3], display=display)
# sim = Simulator([a1, a2, a3], display=True)
time_start=time.time()
results = sim.run(seeds=list(range(epoch_num)), mirror=False, save=save)
time_end=time.time()
print('Time(s)', time_end-time_start)
