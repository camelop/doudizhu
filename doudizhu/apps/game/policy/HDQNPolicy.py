import sys
from copy import deepcopy
from math import tanh
from .learningPolicy import LearningPolicy, rule

class HDQNPolicy(LearningPolicy):

    CALL_SCORE_OPTION_DIM = LearningPolicy.CALL_SCORE_ACTION_DIM
    CALL_SCORE_OPTIONS = LearningPolicy.CALL_SCORE_ACTIONS

    SHOT_POKER_OPTIONS = ["reduce_mycard", "take_next", "help_teammate", "stop_enemy"]
    SHOT_POKER_OPTION_DIM = 4

    OPTION_DIM = CALL_SCORE_OPTION_DIM + SHOT_POKER_OPTION_DIM

    def __init__(self, option_model, action_model, 
        e_greedy=(3e-1, -3e-4),
        turn_reward=-0.01,
        gamma=0.99,
        save_every=50,
        seed=None,
        load_tag=None,
        comment="NOCOMMENT"):

        self.option_model = option_model
        self.action_model = action_model
        self.round = -2
        self.e_greedy, self.e_greedy_inc = e_greedy
        self.gamma = gamma
        self.turn_reward = turn_reward
        self.memory = HMemory()
        self.save_every = save_every
        self.comment = comment
        super().__init__(seed, load_tag)

    def call_score(self, state):
        # controlled by option model
        vec = LearningPolicy._get_state_vec_sv2(state)
        legal = self._legal_call_score(state)
        mask = [1 if i in legal else 0 for i in HDQNPolicy.CALL_SCORE_OPTIONS] + [0] * HDQNPolicy.SHOT_POKER_OPTION_DIM
        self.memory.set_state((vec, mask), None)
        # epsilon_greedy
        prob, action = self._random_call_score(state)
        action_idx = HDQNPolicy.CALL_SCORE_OPTIONS.index(action)
        if prob > self.e_greedy:
            action_idx = self.option_model.choose(vec, mask)
            assert action_idx < 3
            action = HDQNPolicy.CALL_SCORE_OPTIONS[action_idx]
        self.memory.set_action((action_idx, action), None)
        self.memory.set_reward(0, None)
        return action

    def shot_poker(self, state):
        hand_pokers = state['hand_pokers']
        # update the option
        vec, option_mask, new_option = self._update_option(state)

        # action model generate actual action
        action_mask = self._get_shot_poker_mask(state)
        option_target_vec = [0] * HDQNPolicy.SHOT_POKER_OPTION_DIM
        option_target_vec[self.option] = 1
        action_vec = vec + option_target_vec

        self.memory.set_state((vec, option_mask), (action_vec, action_mask, state))
        # epsilon greedy
        prob, action = self._random_shot_poker(state)
        if ''.join(rule._to_cards(action)) == 'wW': # an exception in rule.json
            action_idx = LearningPolicy.SHOT_POKER_ACTIONS.index('Ww') + LearningPolicy.CALL_SCORE_ACTION_DIM
        else:
            action_idx = LearningPolicy.SHOT_POKER_ACTIONS.index(''.join(rule._to_cards(action))) + LearningPolicy.CALL_SCORE_ACTION_DIM
        if prob > self.e_greedy:
            action_idx = self.action_model.choose(action_vec, action_mask)
            action = rule._to_pokers(hand_pokers, LearningPolicy.SHOT_POKER_ACTIONS[action_idx - LearningPolicy.CALL_SCORE_ACTION_DIM])
        self.memory.set_action((new_option, None), (action_idx, action))
        # reward: turn punish
        self.memory.set_reward(self.turn_reward, None)
        return action

    def finish(self, reward):
        self.memory.set_last_state(None, None, reward, None)
        self.train()

    def train(self):
        option_sars, action_all_sars = self.memory.generate_sars()
        '''
        print("SARS:----------------------------")
        print("OP:")
        for s, a, r, s_, isLast in option_sars:
            print(len(s[0]), s[1], a, r, "X", isLast)
        '''
        # print("ACs:")
        # print(action_all_sars)

        option_loss = self._train_option(option_sars)
        action_loss = []
        for action_sars in action_all_sars:
            action_loss.append(self._train_action(action_sars))
        print('Option loss: {}, Action loss(ave): {}'.format(option_loss, sum(action_loss)* 1.0 / len(action_loss)), file=sys.stderr)
        if self.round % self.save_every == 0:
            self.save()

    def save(self):
        '''
        save the model by tag, called when upgrade triggered
        '''
        print("Save -> {}".format(str(self)), file=sys.stderr)
        self.option_model.save(tag=str(self)+"_option")
        self.action_model.save(tag=str(self)+"_action")

    def load(self, tag):
        '''
        load the model by tag
        '''
        if str(self) != tag:
            print('should be ->', tag)
            print('really be ->', str(self))
        assert str(self) == tag 
        self.option_model.load(str(self)+"_option")
        self.action_model.load(str(self)+"_action")

    def reset(self):
        super().reset()
        self.round += 1
        self.e_greedy += self.e_greedy_inc
        self.memory = HMemory()
        self.option = None
    
    def __str__(self):
        return "HDQNPolicy-sv2[{}][{}]({}).{}.v{}".format(str(self.option_model), str(self.action_model),  self.seed, self.comment, self.generation)
    
    def _update_option(self, state):
        '''
        return vec, option_mask, new_option
        '''
        detail = LearningPolicy._get_state_detail(state)
        role = detail['role']
        vec = LearningPolicy._get_state_vec_sv2(state)
        if self.option is not None:
            # check if quit
            if is_sub_trajectory_terminal(state):
                self.option = None
            else:
                return vec, None, None # no need to change
        # need a new option, first generate mask
        mask = [1] * HDQNPolicy.SHOT_POKER_OPTION_DIM
        if role == "LANDLORD":
            mask[HDQNPolicy.SHOT_POKER_OPTIONS.index('help_teammate')] = 0
        mask = [0] * HDQNPolicy.CALL_SCORE_OPTION_DIM + mask
        new_option = self.option_model.choose(vec, mask)
        self.option = new_option - HDQNPolicy.CALL_SCORE_OPTION_DIM
        # print("New Option -> ",self.SHOT_POKER_OPTIONS[self.option])
        return vec, mask, new_option

    def _train_option(self, sars):
        states, actions, qvalues = [], [], []
        for _s, a, r, s_, isLast in sars:
            states.append(_s[0])
            actions.append(a[0])
            if isLast:
                qvalues.append(r)
            else:
                qvalues.append(r + self.gamma * self.option_model.value(s_[0], s_[1]))
        loss = self.option_model.train(states, actions, qvalues)
        return loss

    def _train_action(self, sars):
        states, actions, qvalues = [], [], []
        for _s, a, r, s_, isLast in sars:
            states.append(_s[0])
            actions.append(a[0])
            if isLast:
                qvalues.append(r)
            else:
                qvalues.append(r + self.gamma * self.action_model.value(s_[0], s_[1]))
        loss = self.action_model.train(states, actions, qvalues)
        return loss

def is_sub_trajectory_terminal(state):
    if state is None:
        return True
    detail = LearningPolicy._get_state_detail(state)
    history = detail['history']
    if len(history) < 3:
        return False
    cnt = 0
    for who, what in reversed(history[-4:]):
        if len(what) == 0:
            cnt += 1
            if cnt >= 2:
                return True
        else:
            cnt = 0
    return False
    
class HMemory(object):

    def __init__(self):
        self.next = 's'
        self.option_memory = []
        self.action_memory = []
    
    def set_state(self, S, s):
        assert self.next == 's'
        self.option_memory.append(S)
        self.action_memory.append(deepcopy(s))
        self.next = 'a'
    
    def set_action(self, A, a):
        assert self.next == 'a'
        self.option_memory.append(A)
        self.action_memory.append(a)
        self.next = 'r'
    
    def set_reward(self, R, r):
        assert self.next == 'r'
        self.option_memory.append(R)
        self.action_memory.append(r)
        self.next = 's'
    
    def set_last_state(self, S, s, R, r):
        assert self.next == 's'
        self.option_memory[-1] = R
        self.action_memory[-1] = r
        self.option_memory.append(S)
        self.action_memory.append(s)
        self.next = None

    class Evaluator(object):
        # supported: ["reduce_mycard", "take_next", "help_teammate", "stop_enemy"]
        def __init__(self, subgoal):
            self.subgoal = subgoal
            self.first = True
            self.shots = []
        def update(self, state):
            if state is None:
                return
            detail = LearningPolicy._get_state_detail(state)
            history = detail['history']
            role = detail['role']
            assert history is not None
            if self.first:
                self.first = False
                self.role = role
                # add all that left
                cnt = 0 
                me = 2
                for who, what in reversed(history[:-3]):
                    self.shots.append((who, what, me))
                    me = (me - 1 + 3) % 3
                    if len(what) == 0:
                        cnt += 1
                        if cnt >= 2:
                            self.shots = self.shots[:-2]
                            break
                    else:
                        cnt = 0
                self.shots = list(reversed(self.shots))
            else:
                assert len(history) >= 3

            if len(self.shots) == 0:
                cnt = 0
            else:
                cnt = 1 if len(self.shots[-1][1]) == 0 else 0
            for i, (who, what) in enumerate(history[-3:]):
                self.shots.append((who, what, i))
                if len(what) == 0:
                    cnt += 1
                    if cnt >= 2:
                        break
                else:
                    cnt = 0

        def get_reward(self):
            SCALE = 1
            rew = self._get_reward()
            '''
            print("Subgoal: {}".format(self.subgoal))
            print("Shots: {}".format(str(self.shots)))
            print("Reward: {}".format(str(rew)))
            '''
            return rew * SCALE

        def _get_reward(self):
            if self.subgoal == "reduce_mycard":
                TANH_SCALE = 16.0
                cnt = 0
                for who, what, me in self.shots:
                    if me == 0:
                        cnt += len(what)
                    else:
                        cnt -= len(what)
                return tanh(1.0 * cnt / TANH_SCALE)
            elif self.subgoal == "take_next":
                return 1 if self.shots[-1][2] == 2 else -1
            elif self.subgoal == "help_teammate":
                assert self.role != "LANDLORD"
                teammate = None
                if self.role == "FARMER1":
                    teammate = 1
                elif self.role == "FARMER2":
                    teammate = 2
                TANH_SCALE = 16.0
                cnt = 0
                for who, what, me in self.shots:
                    if me == teammate:
                        cnt += len(what)
                    elif me != 0 and me != teammate:
                        cnt -= len(what)
                return tanh(1.0 * cnt / TANH_SCALE)
            elif self.subgoal == "stop_enemy":
                enemy = []
                if self.role == 'LANDLORD':
                    enemy = [1, 2]
                elif self.role == 'FARMER1':
                    enemy = [2]
                else: # self.role == 'FARMER2'
                    enemy = [1]
                TANH_SCALE = 16.0
                cnt = 0
                for who, what, me in self.shots:
                    if me in enemy:
                        cnt += len(what)
                return 1 - 2 * tanh(1.0 * cnt / TANH_SCALE)
            else:
                raise NotImplementedError

    def corrected_action_all_sars(self):
        '''
        return all action's sars according to their different goals
        '''
        # slice trajectory into sub-trajectories
        i = 0
        ret = []
        current_sars = []
        evaluator = None
        while i + 3 < len(self.action_memory):
            s, a, r, s_, isLast = self.action_memory[i], self.action_memory[i+1], self.action_memory[i+2], self.action_memory[i+3], i+6 >= len(self.action_memory)
            # for each sub-trajectory, figure out its sub-goal
            if evaluator is None: # a new turn or error
                if i == 0: # call_score
                    i += 3
                    continue
                else:
                    new_option = self.option_memory[i+1][0]
                    assert new_option is not None
                    evaluator = HMemory.Evaluator(HDQNPolicy.SHOT_POKER_OPTIONS[new_option - HDQNPolicy.CALL_SCORE_OPTION_DIM])
                    if i == 3:
                        evaluator.update(s[2])
            if s_ is None:
                # for the last
                SCALE = 32
                current_sars.append((s, a, self.option_memory[i+2] / SCALE, s_, True))
                ret.append(current_sars)
                current_sars = []
                break
            else:
                evaluator.update(s_[2])
            if is_sub_trajectory_terminal(s_[2] if s_ is not None else None):
                # call their evaluator to give out its reward
                r = evaluator.get_reward()
                isLast = True
                current_sars.append((s, a, r, s_, isLast))
                ret.append(current_sars)
                current_sars = []
                evaluator = None
            else:
                current_sars.append((s, a, 0, s_, isLast))
            i += 3
        if len(current_sars) > 0:
            ret.append(current_sars)
        # return all sars
        return ret

    def generate_sars(self):
        i = 0
        option_ret = []
        _s = None
        _a = None
        accum_r = 0
        while i + 3 < len(self.option_memory):
            s, a, r, s_, isLast = self.option_memory[i], self.option_memory[i+1], self.option_memory[i+2], self.option_memory[i+3], i+6 >= len(self.option_memory)
            if _s is None:
                _s = s
            if _a is None:
                _a = a
            if isLast or self.option_memory[i+4][0] is not None:
                option_ret.append((_s, _a, accum_r+r, s_, isLast))
                _s, _a, accum_r = None, None, 0
            else:
                accum_r += r
            i += 3
        action_ret = self.corrected_action_all_sars()
        return option_ret, action_ret
