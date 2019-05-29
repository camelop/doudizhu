import random 

from .basePolicy import BasePolicy

class NegativePolicy(BasePolicy):

    def __init__(self, seed=None):
        if seed:
            random.seed(seed)

    def call_score(self, state, default_action=None):
        return 1

    def shot_poker(self, state, default_action=None):
        table = state['table']
        seat = state['seat']
        first = not table.last_shot_poker or table.last_shot_seat == seat        
        if first:
            ret = random.sample(self._legal_shot_poker(state), 1)
        else:
            ret = []
        return ret