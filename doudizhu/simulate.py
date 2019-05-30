import time

from apps.game.simulation import Simulator, Agent
from apps.game.policy.randomPolicy import RandomPolicy
from apps.game.policy.negativePolicy import NegativePolicy
from apps.game.policy.chooseMinPolicy import ChooseMinPolicy


# a1 = Agent('Stark', RandomPolicy(seed=0))
a1 = Agent('Bolton', ChooseMinPolicy(seed=0))
a2 = Agent('Lannister', RandomPolicy(seed=1))
a3 = Agent('Targaryen', RandomPolicy(seed=2))
sim = Simulator([a1, a2, a3], display=False)

time_start=time.time()
results = sim.run(seeds=list(range(100)), mirror=True, save=True)
time_end=time.time()
print('Time(s)', time_end-time_start)