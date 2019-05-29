import json
import logging

from tornado.ioloop import IOLoop

from ..player import Player
from ..protocol import Protocol as Pt
from ..rule import rule

from ..policy.basePolicy import BasePolicy

class AiPlayer(Player):

    def __init__(self, uid: str, username: str, player: Player, Policy: BasePolicy):
        from ..views import LoopBackSocketHandler
        super().__init__(uid, username, LoopBackSocketHandler(self))
        self.room = player.room
        self.history = []
        self.policy = Policy()

    def to_server(self, message):
        packet = json.dumps(message)
        IOLoop.current().add_callback(self.socket.on_message, packet)
        logging.info('AI[%s] REQ: %s', self.uid, message)

    def from_server(self, packet):
        self.history.append(packet)
        logging.info('AI[%s] ON: %s', self.uid, packet)
        code = packet[0]
        if code == Pt.RSP_LOGIN:
            pass
        elif code == Pt.RSP_TABLE_LIST:
            pass
        elif code == Pt.RSP_JOIN_TABLE:
            pass
        elif code == Pt.RSP_DEAL_POKER:
            if self.uid == packet[1]:
                self.auto_call_score()
        elif code == Pt.RSP_CALL_SCORE:
            if self.table.turn_player == self:
                # caller = packet[1]
                # score = packet[2]
                call_end = packet[3]
                if not call_end:
                    self.auto_call_score()
                else:
                    pass
                    # self.auto_shot_poker()
        elif code == Pt.RSP_SHOW_POKER:
            if self.table.turn_player == self:
                self.auto_shot_poker()
        elif code == Pt.RSP_SHOT_POKER:
            if self.table.turn_player == self:
                self.auto_shot_poker()
        elif code == Pt.RSP_GAME_OVER:
            winner = packet[1]
            coin = packet[2]
        else:
            logging.info('AI ERROR PACKET: %s', packet)

    def auto_call_score(self, score=0):
        # millis = random.randint(1000, 2000)
        # score = random.randint(min_score + 1, 3)
        packet = [Pt.REQ_CALL_SCORE, self._call_score()]
        IOLoop.current().add_callback(self.to_server, packet)

    def auto_shot_poker(self):
        packet = [Pt.REQ_SHOT_POKER, self._shot_poker()]
        # IOLoop.current().add_callback(self.to_server, packet)
        IOLoop.current().call_later(2, self.to_server, packet)

    def _get_state(self):
        return {
            "history": self.history,
            "hand_pokers": self.hand_pokers,
            "last_shot_poker": self.table.last_shot_poker,
            "max_call_score": self.table.max_call_score,
            "first": not self.table.last_shot_poker or self.table.last_shot_seat == self.seat,
        }

    def _call_score(self):
        print(str(self)+"callscore")
        default_score = self.table.call_score + 1
        score = self.policy.call_score(self._get_state(), default_action=default_score)
        print(str(self)+"callscore finish")
        return score
    
    def _shot_poker(self):
        default_pokers = []
        if not self.table.last_shot_poker or self.table.last_shot_seat == self.seat:
            default_pokers.append(self.hand_pokers[0])
        else:
            default_pokers = rule.cards_above(self.hand_pokers, self.table.last_shot_poker)
        pokers = self.policy.shot_poker(self._get_state(), default_action=default_pokers)
        print("\n".join([str(h) for h in self.history]))
        return pokers

    def __str__(self):
        return "AI[" + self.uid+ "]{"+ self.name + "}"