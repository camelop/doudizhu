import sys
from .DQNPolicy import LearningPolicy, DQNPolicy, rule

'''
Use summary act instead of actual act.
In this way, we can cut down the output dim.

extra functions:
- learningPolicy._get_shot_poker_mask_sa(state) -> mask
- learningPolicy._random_shot_poker_sa(state, mask) -> prob, action, action_idx
- learningPolicy._sa_idx_to_pokers(action_idx, state) -> action
'''

class SA_DQNPolicy(DQNPolicy):

    def shot_poker(self, state, default_action=None, learning=True):
        '''
        shot a valid poker set, if learning is true, use epsilon greedy
        '''
        hand_pokers = state['hand_pokers']

        vec = LearningPolicy._get_state_vec_sv2(state)
        mask = self._get_shot_poker_mask_sa(state)
        self.memory.set_state((vec, mask))
        # epsilon-greedy
        prob, action, action_idx = self._random_shot_poker_sa(state, mask)
        if prob > self.e_greedy:
            action_idx = self.model.choose(vec, mask)
            action = self._sa_idx_to_pokers(action_idx, state)
        self.memory.set_action((action_idx, action))
        # reward: turn punish
        self.memory.set_reward(self.turn_reward)
        return action
    
    def __str__(self):
        if self.comment is not None:
            return "SA_DQNPolicy-sv2[{}]({}).{}.v{}".format(str(self.model),  self.seed, self.comment, self.generation)
        else:
            return "SA_DQNPolicy-sv2[{}]({}).v{}".format(str(self.model), self.seed, self.generation)