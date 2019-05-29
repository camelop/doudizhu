import random 

from .basePolicy import BasePolicy

class RandomPolicy(BasePolicy):

    def __init__(self, seed=None):
        if seed:
            random.seed(seed)

    def call_score(self, state, default_action=None):
        ret = random.sample(self._legal_call_score(state), 1)[0]
        return ret

    def shot_poker(self, state, default_action=None):
        ret = random.sample(self._legal_shot_poker(state), 1)[0]
        return ret