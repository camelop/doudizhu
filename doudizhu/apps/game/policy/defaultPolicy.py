from .basePolicy import BasePolicy

class DefaultPolicy(BasePolicy):

    def call_score(self, state, default_action=None):
        return default_action

    def shot_poker(self, state, default_action=None):
        return default_action