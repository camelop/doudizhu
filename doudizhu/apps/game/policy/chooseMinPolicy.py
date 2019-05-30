import random, sys

from .basePolicy import BasePolicy, rule

def find_min(ts):
    if len(ts) < 2:
        return ts[0]
    nw = ts[0]
    for i in range(1, len(ts)):
        if len(ts[i]) == 0:
            continue
        if rule.compare_poker(nw, ts[i]) > 0 or len(nw) == 0:
            nw = ts[i]
    return nw
    

class ChooseMinPolicy(BasePolicy):

    def __init__(self, seed=None):
        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randint(0, sys.maxsize)
        random.seed(self.seed)
        self.state = random.getstate()

    def call_score(self, state, default_action=None):
        random.setstate(self.state)
        ret = random.sample(self._legal_call_score(state), 1)[0]
        self.state = random.getstate()
        return ret

    def shot_poker(self, state, default_action=None):
        first = state['first']
        if first:
            # single card
            legal = [t for t in list(self._legal_shot_poker(state)) if len(t) == 1]            
            ret = find_min(legal)
            # print(legal, '->', ret)
        else:
            legal = list(self._legal_shot_poker(state))       
            ret = find_min(legal)
            # print(legal, '->', ret)

        return ret
    
    def reset(self):
        random.seed(self.seed)
        self.state = random.getstate()
        
    def __str__(self):
        return "ChooseMinPolicy({})".format(self.seed)