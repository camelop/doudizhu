from .policy.defaultPolicy import DefaultPolicy
from .policy.randomPolicy import RandomPolicy
from .policy.negativePolicy import NegativePolicy
from .policy.chooseMinPolicy import ChooseMinPolicy

# POLICY1 = DefaultPolicy
# POLICY1 = RandomPolicy

# POLICY1 = NegativePolicy()
# POLICY2 = NegativePolicy()

POLICY1 = ChooseMinPolicy()
POLICY2 = ChooseMinPolicy()
