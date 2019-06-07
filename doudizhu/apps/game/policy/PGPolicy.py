import sys
from .learningPolicy import LearningPolicy, rule
from .DQNPolicy import Memory

class PGPolicy(LearningPolicy):

    # copy and modify this part 

    def __init__(self, model, turn_reward=-0.01, gamma = 0.99, save_every=50, seed=None, load_tag=None, comment=None):
        self.model = model
        self.round = -2 # except Agent.__init__ and first_deal_poker
        self.gamma = gamma
        self.turn_reward = turn_reward
        self.memory = Memory()
        self.save_every = save_every
        self.comment = comment
        '''
        Assumptions about model
        - model.choose(vec, mask, rand) : return the chosen action_idx
        - model.train(states, action_idxs, vt)
        - model.save(tag)
        - model.load(tag)
        - model.__str__ : represent a unique model
        '''
        super().__init__(seed, load_tag)

    def call_score(self, state, default_action=None, learning=True):
        '''
        call a score from 1-3, if learning is true, use epsilon greedy
        '''
        vec = LearningPolicy._get_state_vec_sv2(state)
        mask = self._get_call_score_mask(state)
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
        mask = self._get_shot_poker_mask(state)
        self.memory.set_state((vec, mask))
        # generate action_idx and action
        action_idx = self.model.choose(vec, mask, self._random_rand())
        action = rule._to_pokers(hand_pokers, LearningPolicy.SHOT_POKER_ACTIONS[action_idx - LearningPolicy.CALL_SCORE_ACTION_DIM])
        # record action
        self.memory.set_action((action_idx, action))
        # reward: turn punish
        self.memory.set_reward(self.turn_reward)
        return action

    def finish(self, reward):
        '''
        record the reward
        '''
        vec = None
        mask = None
        self.memory.set_last_state((vec, mask), reward)
        self.train()

    def train(self):
        '''
        train the model
        '''
        states = []
        actions = []
        vts = []
        for _s, a, r, s_, isLast in reversed(self.memory.generate_sars()):
            states.append(_s[0]) # s: (vec, mask)
            actions.append(a[0]) # a: (idx, action)
            if isLast:
                vts.append(r)
            else:
                vts.append(r + self.gamma * vts[-1])
        states = list(reversed(states))
        actions = list(reversed(actions))
        vts = list(reversed(vts))
        loss = self.model.train(states, actions, vts)
        if self.round % self.save_every == 0:
            self.save()
        print('loss: ', loss, file=sys.stderr)

    def save(self):
        '''
        save the model by tag, called when upgrade triggered
        '''
        print("Save -> {}".format(str(self)), file=sys.stderr)
        self.model.save(tag=str(self))

    def load(self, tag):
        '''
        load the model by tag
        '''
        self.model.load(tag)
        if str(self) != tag:
            print('should be ->', tag)
            print('really be ->', str(self))
        assert str(self) == tag 
    
    def reset(self):
        super().reset()
        self.round += 1
        self.memory = Memory()
    
    def __str__(self):
        if self.comment is None:
            self.comment = 'NOCOMMENT'
        return "PGPolicy-sv2[{}]-seed({}).{}.v{}".format(str(self.model),  self.seed, self.comment, self.generation)