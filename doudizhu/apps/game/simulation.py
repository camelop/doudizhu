import random, sys, os

from .rule import rule
from .protocol import Protocol as Pt
from .policy.randomPolicy import RandomPolicy

class Agent(object):
    FARMER = 1
    LANDLORD = 2
    def __init__(self, name, policy):
        self.uid = name
        self.name = name
        self.policy = policy
        self.reset()
    def reset(self):
        self.policy.reset()
        self.history = []
        self.hand_pokers = []
        self.role = Agent.FARMER
    def rsp_join_table(self, room: int, players):
        self.history.append([Pt.RSP_JOIN_TABLE, room, players.copy()])
    def rsp_deal_poker(self, name: str, poker):
        self.history.append([Pt.RSP_DEAL_POKER, name, list(poker).copy()])
    def rsp_call_score(self, who: str, score: int, call_end: bool):
        self.history.append([Pt.RSP_CALL_SCORE, who, score, call_end])
    def rsp_show_poker(self, who: str, poker):
        self.history.append([Pt.RSP_SHOW_POKER, who, list(poker).copy()])
    def rsp_shot_poker(self, who: str, poker):
        self.history.append([Pt.RSP_SHOT_POKER, who, list(poker).copy()])
    def call_score(self, simulator):
        return self.policy.call_score(self._get_state(simulator))
    def shot_poker(self, simulator):
        return self.policy.shot_poker(self._get_state(simulator))
    def finish(self, reward):
        self.policy.finish(reward)
    def _get_state(self, simulator):
        return {
            "history": self.history,
            "hand_pokers": self.hand_pokers,
            "last_shot_poker": simulator.last_shot_poker,
            "max_call_score": simulator.max_call_score,
            "first": not simulator.last_shot_poker or simulator.last_shot_seat == simulator.whose_turn,
        }
        
class Simulator(object):

    def __init__(self, players, display=True):
        '''
        set necessary configurations for related simulations
        '''
        self.players = players
        self.display = display
        pass

    def run(self, seeds=None):
        '''
        run the simulations, return the trajectories
        '''
        if isinstance(seeds, list):
            return [self.run(seeds=seed) for seed in seeds]
        elif isinstance(seeds, int) or seeds is None:
            self._restart(seed=seeds)
            # run deterministic simulation
            # call phase
            for i in range(3):
                agent = self.players[(self.whose_turn+i)%3]
                score = agent.call_score(self)
                assert 0 <= score <= 3
                if score > self.max_call_score:
                    self.max_call_score = score
                    self.max_call_score_turn = (self.whose_turn+i)%3
                call_end = i == 2 or score == 3
                for p in self.players:
                    p.rsp_call_score(agent.uid, score, call_end)
                if call_end:
                    break
            self.whose_turn = self.max_call_score_turn
            self.last_shot_seat = self.whose_turn
            self.players[self.whose_turn].hand_pokers += self.pokers[-3:]
            self.players[self.whose_turn].role = Agent.LANDLORD
            for p in self.players:
                p.rsp_show_poker(self.players[self.whose_turn].uid, self.pokers[-3:])
            # shot phase
            while True:
                agent = self.players[self.whose_turn]
                pokers = agent.shot_poker(self)
                if self.display:
                    print("{} \tby {}".format("".join(rule._to_cards(pokers)) if pokers else '-', agent.uid), file=sys.stderr)
                self.turn += 1
                assert isinstance(pokers, list) or isinstance(pokers, tuple)
                if pokers:
                    assert rule.is_contains(agent.hand_pokers, pokers)
                    assert self.last_shot_seat == self.whose_turn or rule.compare_poker(pokers, self.last_shot_poker) >= 0
                    self.last_shot_seat = self.whose_turn
                    self.last_shot_poker = pokers
                    for poker in pokers:
                        agent.hand_pokers.remove(poker)
                    card_type, card_value = rule._cards_value(rule._to_cards(pokers))
                    if card_value >= 1000:
                        self.multiple *= 2
                self.whose_turn = (self.whose_turn + 1) % 3
                for p in self.players:
                    p.rsp_shot_poker(agent.uid, pokers)
                
                if len(agent.hand_pokers) == 0:
                    # game over
                    result = {}
                    point = self.max_call_score * self.multiple
                    if agent.role == Agent.FARMER:
                        for p in self.players:
                            result[p] = point if p.role == Agent.FARMER else -2*point
                    else:
                        for p in self.players:
                            result[p] = 2*point if p.role == Agent.LANDLORD else -point
                    for p in self.players:
                        p.finish(result[p])
                    print(self.turn, "turns", {k.uid:('FARMER' if k.role == 1 else 'LANDLORD',v) for k,v in result.items()}, file=sys.stderr)
                    return result
        else:
            raise NotImplementedError

    def _restart(self, seed=None):
        for agent in self.players:
            assert isinstance(agent, Agent)
            agent.reset()
            agent.rsp_join_table(1, [(p.uid, p.name) for p in self.players])
        self.multiple = 1
        self.call_score = 0
        self.max_call_score = 0
        self.max_call_score_turn = 0
        self.last_shot_seat = 0
        self.turn = 0
        self.last_shot_poker = []
        self.pokers = list(range(54))
        if seed:
            random.seed(seed)
        self.whose_turn = random.randint(0, 2)
        random.shuffle(self.pokers)
        for i, agent in enumerate(self.players):
            agent.hand_pokers = self.pokers[17*i:17*i+17]
            agent.hand_pokers.sort()        
            agent.rsp_deal_poker(self.players[self.whose_turn].uid, agent.hand_pokers)