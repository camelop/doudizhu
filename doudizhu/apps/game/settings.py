from .policy.defaultPolicy import DefaultPolicy
from .policy.randomPolicy import RandomPolicy
from .policy.negativePolicy import NegativePolicy
from .policy.chooseMinPolicy import ChooseMinPolicy
from .policy.chooseMinWithRolePolicy import ChooseMinWithRolePolicy

from .policy.DRL.MLP import MultiLevelPerceptron
from .policy.DQNPolicy import DQNPolicy

# POLICY1 = DefaultPolicy
# POLICY1 = RandomPolicy

# POLICY1 = NegativePolicy()
# POLICY2 = NegativePolicy()

# POLICY1 = ChooseMinPolicy()
# POLICY2 = ChooseMinPolicy()

# POLICY1 = ChooseMinWithRolePolicy()
# POLICY2 = ChooseMinWithRolePolicy()

a1_model = MultiLevelPerceptron(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(1000, 1000), freeze=True)
a1_policy = DQNPolicy(a1_model, seed=0, comment="default", load_tag="DQNPolicy-sv1[MLP-lr(0.001)-h1(1000)-h2(1000)-e(1951)](0).default.v0")
POLICY1 = a1_policy

a2_model = MultiLevelPerceptron(action_dim=DQNPolicy.ACTION_DIM, hidden_dims=(1000, 1000), freeze=True)
a2_policy = DQNPolicy(a2_model, seed=0, comment="default", load_tag="DQNPolicy-sv1[MLP-lr(0.001)-h1(1000)-h2(1000)-e(1951)](0).default.v0")
POLICY2 = a2_policy