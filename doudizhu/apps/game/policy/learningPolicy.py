import random, sys

from ..protocol import Protocol as Pt
from .basePolicy import BasePolicy, rule
from ..rule import Rule

class LearningPolicy(BasePolicy):

    SHOT_POKER_ACTION_CATAGORY_NUM = 38
    SHOT_POKER_ACTION_CATAGORY = list(rule.rules.keys()).copy()
    SHOT_POKER_ACTION_DIM = 13998 + 1
    SHOT_POKER_ACTIONS = sum([v.copy() for k,v in rule.rules.items()], [''])
    CALL_SCORE_ACTION_DIM = 3
    CALL_SCORE_ACTIONS = [1, 2, 3]
    ACTION_DIM = CALL_SCORE_ACTION_DIM + SHOT_POKER_ACTION_DIM
    # copy and modify this part 

    def __init__(self, seed=None, load_tag=None):
        if seed is not None:
            self.seed = seed
        else:
            self.seed = random.randint(0, sys.maxsize)
        random.seed(self.seed)
        self.state = random.getstate()
        self.generation = 0
        if load_tag is not None:
            self.load(load_tag)

    def call_score(self, state, default_action=None, learning=True):
        '''
        call a score from 1-3, if learning is true, use epsilon greedy
        '''
        raise NotImplementedError

    def shot_poker(self, state, default_action=None, learning=True):
        '''
        shot a valid poker set, if learning is true, use epsilon greedy
        '''
        raise NotImplementedError

    def finish(self, reward):
        '''
        record the reward
        '''
        raise NotImplementedError

    def save(self):
        '''
        save the model by tag, called when upgrade triggered
        '''
        raise NotImplementedError

    def load(self, tag):
        '''
        load the model by tag
        '''
        raise NotImplementedError

    def __str__(self):
        return "LearningPolicy({}).v{}".format(self.seed, self.generation)

    def reset(self):
        random.seed(self.seed)
        self.state = random.getstate()

    # util methods
    def _get_state_detail(state):
        history = state['history']
        me = state['me']
        hand_pokers = state['hand_pokers']

        landlord = None
        last = None
        last_shot = None
        myseat = None
        seats = None
        used = []
        shot_history = []
        for h in history:
            if h[0] == Pt.RSP_JOIN_TABLE:
                seats = [uid for uid, name in h[2]]
                for i, uid in enumerate(seats):
                    if uid == me:
                        myseat = i
            elif h[0] == Pt.RSP_SHOW_POKER:
                landlord = h[1]
            elif h[0] == Pt.RSP_SHOT_POKER:
                shot_history.append((h[1], h[2]))
                if len(h[2]) > 0:
                    last = h[1]
                    last_shot = h[2]
                    used = used + list(h[2])
        uid_to_roleid = {} # roleid LANDLORD-0, FARMER1-1, FARMER2-2
        if landlord is not None:
            if me == landlord:
                uid_to_roleid[seats[myseat]] = 0
                uid_to_roleid[seats[(myseat+1)%3]] = 1
                uid_to_roleid[seats[(myseat+2)%3]] = 2
            elif seats[(myseat+1)%3] == landlord:
                uid_to_roleid[seats[myseat]] = 2
                uid_to_roleid[seats[(myseat+1)%3]] = 0
                uid_to_roleid[seats[(myseat+2)%3]] = 1
            else:
                uid_to_roleid[seats[myseat]] = 1
                uid_to_roleid[seats[(myseat+1)%3]] = 2
                uid_to_roleid[seats[(myseat+2)%3]] = 0
            shot_history = [(uid_to_roleid[uid], poker) for uid, poker in shot_history]
        if last is not None:
            last = uid_to_roleid[last]

        detail = {}
        # depend on cards in hand
        detail['hand_cards'] = list(hand_pokers)
        # depend on what role that we're playing
        if landlord is None:
            detail['role'] = None # not decided yet
        elif landlord == me:
            detail['role'] = 'LANDLORD' # LANDLORD -> FARMER1 -> FARMER2
        elif landlord == seats[(myseat+1)%3]:
            detail['role'] = 'FARMER2'
        else:
            detail['role'] = 'FARMER1'
        # depend on what is on the deck and who shotted it
        detail['last_shot'] = (last, last_shot)
        # depend on cards that have been used
        detail['used_cards'] = used
        # depend on what cards might others hold, which may be deduced by history
        detail['history'] = None if landlord is None else shot_history
        # depend on who you're playing with (or should we consider this?)
        detail['opponent'] = seats 
        return detail

    def _get_call_score_mask(self, state):
        legal = self._legal_call_score(state)
        ret = []
        for i in LearningPolicy.CALL_SCORE_ACTIONS:
            ret.append(1 if i in legal else 0)
        ret = ret + [0] * LearningPolicy.SHOT_POKER_ACTION_DIM
        return ret
    
    def _get_shot_poker_mask(self, state):
        hand_pokers = state['hand_pokers']
        legal = self._legal_shot_poker(state)
        ret = [0] * LearningPolicy.CALL_SCORE_ACTION_DIM
        for i in LearningPolicy.SHOT_POKER_ACTIONS:
            pokers = rule._to_pokers(hand_pokers, i)
            if len(pokers) != len(i):
                ret.append(0)
                continue
            ret.append(1 if tuple(pokers) in legal else 0)
        return ret

    def _get_state_vec_sv1(state):
        # basic and raw
        return LearningPolicy._get_state_vec(state, content=['hand_cards', 'role'], convert_pokers=LearningPolicy._pokers_to_raw_vec)
        
    def _get_state_vec_sv2(state):
        # basic and raw
        return LearningPolicy._get_state_vec(state, content=['hand_cards', 'role', 'last_shot', 'used_cards'], convert_pokers=LearningPolicy._pokers_to_cnt_vec)

    def _get_state_vec(state, content='all', convert_pokers=None):
        if convert_pokers is None:
            convert_pokers = LearningPolicy._pokers_to_raw_vec
        detail = LearningPolicy._get_state_detail(state)
        if content == 'all':
            content = ['hand_cards', 'role', 'last_shot', 'used_cards', 'history', 'opponent']
        assert isinstance(content, list)
        ret = []
        if 'hand_cards' in content:
            ret = ret + convert_pokers(detail['hand_cards'])
        if 'role' in content:
            ret = ret + LearningPolicy._role_to_vec(detail['role'])
        if 'last_shot' in content:
            last_shot_who, last_shot_what = detail['last_shot']
            temp = [0] * 4
            if last_shot_who is not None:
                temp[last_shot_who] = 1
            ret = ret + temp
            ret = ret + convert_pokers(last_shot_what)
        if 'used_cards' in content:
            ret = ret + convert_pokers(detail['used_cards'])
        if 'history' in content:
            raise NotImplementedError
        if 'opponent' in content:
            raise NotImplementedError
        return ret

    def _role_to_vec(role):
        if role is None:
            return [0, 0, 0]
        elif role == 'LANDLORD':
            return [1, 0, 0]
        elif role == 'FARMER1':
            return [0, 1, 0]
        elif role == 'FARMER2':
            return [0, 0, 1]

    def _pokers_to_raw_vec(pokers):
        if pokers is None:
            return [0] * 54
        return [1 if i in pokers else 0 for i in range(54)]

    def _pokers_to_cnt_vec(pokers):
        # A 2 3 4 5 6 7 8 9 0 J Q K W w
        if pokers is None:
            return [0] * 67
        cnt = [0] * 13
        for i in range(52):
            if i in pokers:
                cnt[i % 13] += 1
        ret = []
        for i in range(13):
            temp = [0] * 5
            temp[cnt[i]] = 1
            ret = ret + temp
        ret = ret + [1 if 52 in pokers else 0] # W
        ret = ret + [1 if 53 in pokers else 0] # w
        return ret

    def _random_call_score(self, state):
        random.setstate(self.state)
        random.seed(self.seed+hash(tuple(sorted(state["hand_pokers"]))))
        prob = random.random()
        ret = random.sample(self._legal_call_score(state), 1)[0]
        self.state = random.getstate()
        return prob, ret

    def _random_shot_poker(self, state):
        random.setstate(self.state)
        prob = random.random()
        ret = random.sample(self._legal_shot_poker(state), 1)[0]
        self.state = random.getstate()
        return prob, ret

    def _random_rand(self):
        random.setstate(self.state)
        rand = random.random()
        self.state = random.getstate()
        return rand
        
    # util about summary act
    '''
    Summary Act:
        0 skip
        1-38 use the smallest i'th-catagory
        39-76 use the largest i'th-catagory
    '''
    SHOT_POKER_S_ACTION_DIM = 1 + 38 * 2
    S_ACTION_DIM = CALL_SCORE_ACTION_DIM + SHOT_POKER_S_ACTION_DIM

    def _get_shot_poker_mask_sa(self, state):
        ret = [0] * (LearningPolicy.CALL_SCORE_ACTION_DIM + LearningPolicy.SHOT_POKER_S_ACTION_DIM)

        hand_pokers = state['hand_pokers']
        first = state['first']
        last_shot_poker = state['last_shot_poker']

        turn_pokers = last_shot_poker
        hand_cards = Rule._to_cards(hand_pokers)
        turn_cards = Rule._to_cards(turn_pokers)
        
        def mark(ret, card_type):
            type_idx = LearningPolicy.SHOT_POKER_ACTION_CATAGORY.index(card_type)
            ret[1 + LearningPolicy.CALL_SCORE_ACTION_DIM + type_idx] = 1
            ret[1 + LearningPolicy.CALL_SCORE_ACTION_DIM + LearningPolicy.SHOT_POKER_ACTION_CATAGORY_NUM + type_idx] = 1

        if not first:
            ret[LearningPolicy.CALL_SCORE_ACTION_DIM] = 1 # 'skip'
            card_type, card_value = rule._cards_value(turn_cards)
            if not card_type:
                raise NotImplementedError

            one_rule = rule.rules[card_type]
            for i, t in enumerate(one_rule):
                if i > card_value and Rule.is_contains(hand_cards, t):
                    mark(ret, card_type)
                    break 
            if card_value < 1000:
                one_rule = rule.rules['bomb']
                for t in one_rule:
                    if Rule.is_contains(hand_cards, t):
                        mark(ret, 'bomb')
                        break
                if Rule.is_contains(hand_cards, 'wW'): # ROCKET
                    mark(ret, 'rocket')
            return ret
        else:
            for card_type, one_rule in rule.rules.items():
                for t in one_rule:
                    if Rule.is_contains(hand_cards, t):
                        mark(ret, card_type)
                        break
            return ret

    def _random_shot_poker_sa(self, state, mask):
        '''
        return prob, action, action_idx
        NOTICE: action_idx is starting from 3 
                because of call score actions
        '''
        random.setstate(self.state)
        prob = random.random()
        # print("mask", [LearningPolicy.SHOT_POKER_ACTION_CATAGORY[(i-4)%38] for i, m in enumerate(mask) if m != 0 and i > 3])
        action_idx = random.sample([i for i, m in enumerate(mask) if m != 0], 1)[0]
        self.state = random.getstate()
        return prob, self._sa_idx_to_pokers(action_idx, state), action_idx
    
    def _sa_idx_to_pokers(self, action_idx, state):
        '''
        return action
        '''
        hand_pokers = state['hand_pokers']
        first = state['first']
        last_shot_poker = state['last_shot_poker']

        turn_pokers = last_shot_poker
        hand_cards = Rule._to_cards(hand_pokers)
        turn_cards = Rule._to_cards(turn_pokers)
        shot_poker_idx = action_idx - LearningPolicy.CALL_SCORE_ACTION_DIM

        if shot_poker_idx == 0:
            assert not first
            return () # shot nothing
        
        type_idx = shot_poker_idx - 1
        small = type_idx < LearningPolicy.SHOT_POKER_ACTION_CATAGORY_NUM
        type_idx %= LearningPolicy.SHOT_POKER_ACTION_CATAGORY_NUM
        target_type = LearningPolicy.SHOT_POKER_ACTION_CATAGORY[type_idx]

        if not first:
            card_type, card_value = rule._cards_value(turn_cards)
            if not card_type:
                raise NotImplementedError

            one_rule = rule.rules[card_type]
            if card_type == target_type: # if not bomb
                large_ret = None
                for i, t in enumerate(one_rule):
                    if i > card_value and Rule.is_contains(hand_cards, t):
                        if small:
                            return tuple(Rule._to_pokers(hand_pokers, t))
                        else:
                            large_ret = tuple(Rule._to_pokers(hand_pokers, t))
                if large_ret is not None:
                    return large_ret

            if target_type == 'rocket' and Rule.is_contains(hand_cards, 'wW'): # ROCKET
                return (52, 53)
            if target_type == 'bomb':
                one_rule = rule.rules['bomb']
                large_ret = None
                for t in one_rule:
                    if Rule.is_contains(hand_cards, t):
                        if small:
                            return tuple(Rule._to_pokers(hand_pokers, t))
                        else:
                            large_ret = tuple(Rule._to_pokers(hand_pokers, t))
                if large_ret is not None:
                    return large_ret
            print("target_type", target_type)
            raise NotImplementedError
        else:
            for card_type, one_rule in rule.rules.items():
                if card_type != target_type:
                    continue
                large_ret = None
                for t in one_rule:
                    if Rule.is_contains(hand_cards, t):
                        if small:
                            return tuple(Rule._to_pokers(hand_pokers, t))
                        else:
                            large_ret = tuple(Rule._to_pokers(hand_pokers, t))
                if large_ret != None:
                    return large_ret
            print("target_type", target_type)
            raise NotImplementedError