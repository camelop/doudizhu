from ..rule import Rule, rule

class BasePolicy(object):
    
    
    def call_score(self, state, default_action=None):
        raise NotImplementedError

    def shot_poker(self, state, default_action=None):
        raise NotImplementedError
    
    # Util methods

    def _legal_call_score(self, state):
        max_call_score = state['max_call_score']
        base_score = max_call_score + 1
        ret = [0]
        if base_score <= 3:
            for i in range(base_score, 4):
                ret.append(i)
        return ret

    def _legal_shot_poker(self, state):
        '''
        notice! return a set of tuples (list is unhashable)
        '''
        hand_pokers = state['hand_pokers']
        first = state['first']
        last_shot_poker = state['last_shot_poker']

        turn_pokers = last_shot_poker
        hand_cards = Rule._to_cards(hand_pokers)
        turn_cards = Rule._to_cards(turn_pokers)
        ret = set()
        
        if not first:
            ret.add(()) # pass
            card_type, card_value = rule._cards_value(turn_cards)
            if not card_type:
                raise NotImplementedError

            one_rule = rule.rules[card_type]
            for i, t in enumerate(one_rule):
                if i > card_value and Rule.is_contains(hand_cards, t):
                    ret.add(tuple(Rule._to_pokers(hand_pokers, t)))

            if card_value < 1000:
                one_rule = rule.rules['bomb']
                for t in one_rule:
                    if Rule.is_contains(hand_cards, t):
                        ret.add(tuple(Rule._to_pokers(hand_pokers, t)))
                if Rule.is_contains(hand_cards, 'wW'): # ROCKET
                    ret.add((52, 53))

            return ret
        else:
            for card_type, one_rule in rule.rules.items():
                for t in one_rule:
                    if Rule.is_contains(hand_cards, t):
                        ret.add(tuple(Rule._to_pokers(hand_pokers, t)))
            return ret

    def finish(self, reward):
        pass
    
    def reset(self):
        pass
        
    def __str__(self):
        return self.__class__.__name__