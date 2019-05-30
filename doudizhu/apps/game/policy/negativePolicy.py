import random 

from .basePolicy import BasePolicy

class NegativePolicy(BasePolicy):

    def __init__(self, seed=None):
        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randint(0, sys.maxsize)
        random.seed(self.seed)
        self.state = random.getstate()

    def call_score(self, state, default_action=None):
        return 1

    def shot_poker(self, state, default_action=None):
        first = state['first']
        if first:
            random.setstate(self.state)
            ret = random.sample(self._legal_shot_poker(state), 1)[0]
            self.state = random.getstate()
        else:
            ret = []
        return ret

    def reset(self):
        random.seed(self.seed)
        self.state = random.getstate()


    def __str__(self):
        return "NegativePolicy({})".format(self.seed)