import sys
from .learningPolicy import LearningPolicy, rule

class DQNPolicy(LearningPolicy):

    # copy and modify this part 

    def __init__(self, model, e_greedy=(0.3, -0.001), turn_reward=-0.01, gamma = 0.99, save_every=50, seed=None, load_tag=None, comment=None):
        self.model = model
        self.round = -2 # except Agent.__init__ and first_deal_poker
        self.e_greedy, self.e_greedy_inc = e_greedy
        self.gamma = gamma
        self.turn_reward = turn_reward
        self.memory = Memory()
        self.save_every = save_every
        self.comment = comment
        '''
        Assumptions about model
        - model.choose(vec, mask) : return the chosen action_idx
        - model.value(vec, mask) : find the state value, usually max Q of allowed actions
        - model.train(states, action_idxs, qvalues)
        - model.save(tag)
        - model.load(tag)
        - model.__str__ : represent a unique model
        '''
        super().__init__(seed, load_tag)

    def call_score(self, state, default_action=None, learning=True):
        '''
        call a score from 1-3, if learning is true, use epsilon greedy
        '''
        vec = LearningPolicy._get_state_vec_sv1(state)
        mask = self._get_call_score_mask(state)
        self.memory.set_state((vec, mask))
        # epsilon-greedy
        prob, action = self._random_call_score(state)
        action_idx = LearningPolicy.CALL_SCORE_ACTIONS.index(action)
        if prob > self.e_greedy:
            action_idx = self.model.choose(vec, mask)
            assert action_idx < 3 # call action 
            action = LearningPolicy.CALL_SCORE_ACTIONS[action_idx]
        self.memory.set_action((action_idx, action))
        # reward: turn punish - call is not involved
        self.memory.set_reward(0)
        return action

    def shot_poker(self, state, default_action=None, learning=True):
        '''
        shot a valid poker set, if learning is true, use epsilon greedy
        '''
        hand_pokers = state['hand_pokers']

        vec = LearningPolicy._get_state_vec_sv1(state)
        mask = self._get_shot_poker_mask(state)
        self.memory.set_state((vec, mask))
        # epsilon-greedy
        prob, action = self._random_shot_poker(state)
        if ''.join(rule._to_cards(action)) == 'wW': # an exception in rule.json
            action_idx = LearningPolicy.SHOT_POKER_ACTIONS.index('Ww') + LearningPolicy.CALL_SCORE_ACTION_DIM
        else:
            action_idx = LearningPolicy.SHOT_POKER_ACTIONS.index(''.join(rule._to_cards(action))) + LearningPolicy.CALL_SCORE_ACTION_DIM
        if prob > self.e_greedy:
            action_idx = self.model.choose(vec, mask)
            '''
            print('idx:', action_idx)
            print('hand:', hand_pokers)
            print('mask:', mask[action_idx])
            print("type:", LearningPolicy.SHOT_POKER_ACTIONS[action_idx - LearningPolicy.CALL_SCORE_ACTION_DIM])
            '''
            action = rule._to_pokers(hand_pokers, LearningPolicy.SHOT_POKER_ACTIONS[action_idx - LearningPolicy.CALL_SCORE_ACTION_DIM])
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
        qvalues = []
        for _s, a, r, s_, isLast in self.memory.generate_sars():
            states.append(_s[0]) # s: (vec, mask)
            actions.append(a[0]) # a: (idx, action)
            if isLast:
                qvalues.append(r)
            else:
                qvalues.append(r + self.gamma * self.model.value(s_[0], s_[1]))
        loss = self.model.train(states, actions, qvalues)
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
        self.e_greedy += self.e_greedy_inc
        self.memory = Memory()
    
    def __str__(self):
        if self.comment is not None:
            return "DQNPolicy-sv1[{}]({}).{}.v{}".format(str(self.model),  self.seed, self.comment, self.generation)
        else:
            return "DQNPolicy-sv1[{}]({}).v{}".format(str(self.model), self.seed, self.generation)

class Memory(object):

    def __init__(self):
        self.next = 's'
        self.memory = []
    
    def set_state(self, s):
        assert self.next == 's'
        self.memory.append(s)
        self.next = 'a'
    
    def set_action(self, a):
        assert self.next == 'a'
        self.memory.append(a)
        self.next = 'r'
    
    def set_reward(self, r):
        assert self.next == 'r'
        self.memory.append(r)
        self.next = 's'
    
    def set_last_state(self, s, r):
        assert self.next == 's'
        self.memory[-1] = r
        self.memory.append(s)
        self.next = None

    def generate_sars(self):
        i = 0
        ret = []
        while i + 3 < len(self.memory):
            ret.append((self.memory[i], self.memory[i+1], self.memory[i+2], self.memory[i+3], i+6 >= len(self.memory)))
            i += 3
        return ret