# Reinforcement learning for doudizhu game

## Game Policy

A doudizhu policy should determine:
- what score to call with cards on hand and others score
- what card to play with history info

What should it consider:
- call score
  - depend on cards in hand
  - depend on who you're playing with (or should we consider this?)  
- shot card
  - depend on what role that we're playing
  - depend on what is on the deck and who shotted it
  - depend on cards in hand
  - depend on cards that have been used
  - depend on what cards might others hold, which may be deduced by history
  - depend on who you're playing with (or should we consider this?)  

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