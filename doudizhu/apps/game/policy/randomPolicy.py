import random, sys

from .basePolicy import BasePolicy

class RandomPolicy(BasePolicy):

    def __init__(self, seed=None):
        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randint(0, sys.maxsize)
        random.seed(self.seed)
        self.state = random.getstate()

    def call_score(self, state, default_action=None):
        random.setstate(self.state)
        random.seed(self.seed+hash(tuple(sorted(state["hand_pokers"]))))
        ret = random.sample(self._legal_call_score(state), 1)[0]
        self.state = random.getstate()
        return ret

    def shot_poker(self, state, default_action=None):
        random.setstate(self.state)
        ret = random.sample(self._legal_shot_poker(state), 1)[0]
        self.state = random.getstate()
        return ret
    
    def reset(self):
        random.seed(self.seed)
        self.state = random.getstate()
        
    def __str__(self):
        return "RandomPolicy({})".format(self.seed)
