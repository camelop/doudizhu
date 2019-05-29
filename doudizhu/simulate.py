import time

from apps.game.simulation import Simulator, Agent
from apps.game.policy.randomPolicy import RandomPolicy
from apps.game.policy.negativePolicy import NegativePolicy


a1 = Agent('Stark', RandomPolicy())
a2 = Agent('Lannister', RandomPolicy())
a3 = Agent('Targaryen', RandomPolicy())
sim = Simulator([a1, a2, a3], display=False)

time_start=time.time()
results = sim.run(seeds=list(range(100)))
time_end=time.time()
print('totally cost', time_end-time_start)
stat = {}
for r in results:
    for k,v in r.items():
        if k.uid not in stat:
            stat[k.uid] = [v]
        else:
            stat[k.uid] += [v]
from pprint import pprint
pprint(stat)
print('----------')
pprint({k: sum(v) for k,v in stat.items()})