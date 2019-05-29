# Reinforcement learning for doudizhu game

## Game Policy

A doudizhu policy should determine:
- what score to call with cards on hand and others score
- what card to play with history info

## Problems

1. Should we change our policy when playing with different opponents?

## History example

```
[<Protocol.RSP_JOIN_TABLE: 20>, 1, [('haha', 'haha'), ('Alice', 'NegativePolicy-I'), ('Bob', 'NegativePolicy-II')]]
[<Protocol.RSP_DEAL_POKER: 32>, 'Alice', [1, 2, 9, 14, 16, 20, 23, 25, 29, 31, 36, 37, 42, 45, 46, 48, 53]]
[<Protocol.RSP_CALL_SCORE: 34>, 'Alice', 1, False]
[<Protocol.RSP_CALL_SCORE: 34>, 'Bob', 1, False]
[<Protocol.RSP_CALL_SCORE: 34>, 'haha', 0, True]
[<Protocol.RSP_SHOW_POKER: 36>, 'Alice', [7, 50, 28]]
[<Protocol.RSP_SHOT_POKER: 38>, 'Alice', [5, 6, 7, 34, 22, 10]]
[<Protocol.RSP_SHOT_POKER: 38>, 'Bob', []]
[<Protocol.RSP_SHOT_POKER: 38>, 'haha', []]
[<Protocol.RSP_SHOT_POKER: 38>, 'Alice', [50]]
[<Protocol.RSP_SHOT_POKER: 38>, 'Bob', []]
[<Protocol.RSP_SHOT_POKER: 38>, 'haha', [0]]
[<Protocol.RSP_SHOT_POKER: 38>, 'Alice', []]
```