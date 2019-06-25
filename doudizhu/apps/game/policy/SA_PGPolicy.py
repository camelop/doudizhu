import sys
from .PGPolicy import LearningPolicy, PGPolicy, rule

'''
Use summary act instead of actual act......
(more) see SA_DQNPolicy.py

'''

class SA_PGPolicy(PGPolicy):

    def call_score(self, state, default_action=None, learning=True):
        '''
        call a score from 1-3, if learning is true, use epsilon greedy
        '''
        vec = LearningPolicy._get_state_vec_sv2(state)
        mask = self._get_call_score_mask_sa(state)
        self.memory.set_state((vec, mask))
        # epsilon-greedy
        # generate action_idx and action
        action_idx = self.model.choose(vec, mask, self._random_rand())
        action = LearningPolicy.CALL_SCORE_ACTIONS[action_idx]
        # record action
        self.memory.set_action((action_idx, action))
        # reward: turn punish - call is not involved so 0
        self.memory.set_reward(0)
        return action

    def shot_poker(self, state, default_action=None, learning=True):
        '''
        shot a valid poker set, if learning is true, use epsilon greedy
        '''
        hand_pokers = state['hand_pokers']

        vec = LearningPolicy._get_state_vec_sv2(state)
        mask = self._get_shot_poker_mask_sa(state)
        self.memory.set_state((vec, mask))
        # generate action_idx and action
        action_idx = self.model.choose(vec, mask, self._random_rand())
        action = self._sa_idx_to_pokers(action_idx, state)
        # record action
        self.memory.set_action((action_idx, action))
        # reward: turn punish
        self.memory.set_reward(self.turn_reward)
        return action

    def __str__(self):
        if self.comment is None:
            self.comment = 'NOCOMMENT'
        return "SA_PGPolicy-sv2[{}]-seed({}).{}.v{}".format(str(self.model),  self.seed, self.comment, self.generation)