from __future__ import annotations

import datetime
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import pandas as pd

# os.environ["PYTHONHASHSEED"] = '0'
# random.seed(42)
COLUMNS = 7
ROWS = 6

Action = int


def printf(*args, **kwargs):
    print(*args, **kwargs, flush=True)


class BitMap:
    def __init__(self, x: int = 0):
        self.x = x

    def __hash__(self):
        return self.x

    def __eq__(self, other):
        return isinstance(other, type(self)) and other.x == self.x

    def __and__(self, other):
        return type(self)(self.x & other.x)

    def __or__(self, other):
        return type(self)(self.x | other.x)

    def __getitem__(self, item):
        return bool(self.x & self.mask(*item))

    def __bool__(self):
        return bool(self.x)

    def __repr__(self):
        return f"BitMap({self.x})"

    def put(self, i, j) -> BitMap:
        return type(self)(self.x | self.mask(i, j))

    def show(self):
        return str(bin(self.x))[2:].rjust(COLUMNS * ROWS, '0')

    @classmethod
    def from_indices(cls, indices):
        x = BitMap()
        for i, j in indices:
            x = x.put(i, j)
        return x

    @staticmethod
    def mask(i: int, j: int) -> int:
        assert i < COLUMNS and j < ROWS
        return 1 << ((j % ROWS) + (i % COLUMNS) * ROWS)


class Connect4Board:
    _term = {}

    def __init__(self, g: BitMap, b: BitMap):
        assert not g & b, "green and blue in same spot"
        self.g = g
        self.b = b
        # s = self.to_str()
        # assert (s.count('G') - s.count('B')) in {0, 1}

    def __eq__(self, other):
        return isinstance(other, type(self)) and other.g == self.g and other.b == self.b

    def __hash__(self):
        return hash((self.g.x, self.b.x))

    def __getitem__(self, item):
        return ['-', 'G', 'B'][self.g[item] + 2 * self.b[item]]

    def to_str(self):
        return "".join(self[i, j] for i in range(COLUMNS) for j in range(ROWS))

    def free_columns(self):
        taken = self.g | self.b
        return [i for i in range(COLUMNS) if not taken[i, ROWS - 1]]

    def show(self):
        print('_' * (COLUMNS * 2 - 1))
        for j in reversed(range(ROWS)):
            print('|'.join(self[i, j] for i in range(COLUMNS)))

    def put(self, i: int, is_green: bool):
        taken = self.g | self.b
        j = [j for j in range(ROWS) if not taken[i, j]][0]
        if is_green:
            return type(self)(self.g.put(i, j), self.b)

        return type(self)(self.g, self.b.put(i, j))

    def terminal(self):
        if self not in self._term:
            if any((self.g & x) == x for x in self.SET):
                self._term[self] = 1
            elif any((self.b & x) == x for x in self.SET):
                self._term[self] = -1
            elif '-' not in self.to_str():
                self._term[self] = 0
            else:
                self._term[self] = None

        return self._term[self]

    @classmethod
    def new(cls):
        return cls(BitMap(), BitMap())

    @classmethod
    def from_str(cls, string: str):
        g = BitMap(sum((x == 'G') << i for i, x in enumerate(string.upper())))
        b = BitMap(sum((x == 'B') << i for i, x in enumerate(string.upper())))
        return cls(g, b)

    @staticmethod
    def win_masks():
        vertical = [BitMap(0b1111 << (x + c * ROWS)) for c in range(COLUMNS) for x in range(ROWS - 3)]

        horizontal = [
            BitMap.from_indices([(i, j) for i in range(i_s, i_s + 4)])
            for j in range(ROWS)
            for i_s in range(COLUMNS - 3)
        ]

        diag_pos = [
            BitMap.from_indices([(i + x, j + x) for x in range(4)])
            for i in range(COLUMNS - 3)
            for j in range(ROWS - 3)
        ]

        diag_neg = [
            BitMap.from_indices([(i + x, j - x) for x in range(4)])
            for i in range(COLUMNS - 3)
            for j in range(3, ROWS)
        ]

        return vertical + horizontal + diag_neg + diag_pos

    SET = set(win_masks())


@dataclass
class Res:
    games: int
    wins: int
    r: int


class MCTS:
    state_action: Dict[Connect4Board, Dict[Action, Res]]

    def __init__(self, root_board: Optional[Connect4Board] = None, c: float = 2 ** .5, gamma: float = 0.99):
        self.root_board = root_board or self.default_board()
        self.state_action = {}
        self.c = c
        self.gamma = gamma

    @staticmethod
    def default_board() -> Connect4Board:
        return Connect4Board.from_str(
            "gbgbgg"
            "gbbgbg"
            "bbbgbg"
            "ggbgbb"
            "------"
            "bggbgb"
            "------"
        )

    @classmethod
    def default_board_2(cls) -> Connect4Board:
        d1 = cls.default_board()
        return Connect4Board(d1.b, d1.g)

    def propagate(self, history: List[Tuple[Connect4Board, Action]], r: int):
        n = len(history)
        for i, (board, action) in enumerate(history):
            gamma = pow(self.gamma, n - i - 1)
            self.state_action[board][action].games += 1
            self.state_action[board][action].wins += (r > 0)
            self.state_action[board][action].r += r * gamma
        return r

    def step(self):
        board = self.root_board
        history = []
        ucb = (lambda res_, n_: res_.r / res_.games + self.c * pow(math.log(n_) / res_.games, .5))

        while board in self.state_action:
            acts = self.state_action[board]
            n = sum([res.games for res in acts.values()])
            best_action = max(acts, key=lambda a: ucb(acts[a], n))
            # if random.random() > 0.01:
            # best_action = min(acts, key=lambda a: abs(3-a))
            history.append((board, best_action))

            nb = board.put(best_action, True)
            if (r := nb.terminal()) is not None:
                return self.propagate(history, r)

            board = nb.put(random.choice(nb.free_columns()), False)
            if (r := board.terminal()) is not None:
                return self.propagate(history, r)

        actions = board.free_columns()
        self.state_action[board] = {a: Res(0, 0, 0) for a in actions}

        for a in actions:
            i = 0
            nb = board.put(a, True)
            ig = False
            while (r := nb.terminal()) is None:
                nb = nb.put(random.choice(nb.free_columns()), ig)
                ig = not ig
                i += 1
            self.propagate(history + [(board, a)], r)  # * pow(.99, i)

    def test_step(self, show: bool = False):
        board = self.root_board

        while True:
            show and board.show()
            if board in self.state_action:
                actions = self.state_action[board]
                action = max(actions, key=lambda a: actions[a].r / actions[a].games)
            else:
                action = random.choice(board.free_columns())

            board = board.put(action, True)
            if (r := board.terminal()) is not None:
                show and board.show()
                return r

            board = board.put(random.choice(board.free_columns()), False)
            if (r := board.terminal()) is not None:
                show and board.show()
                return r

    def test(self, steps: int):
        res = [self.test_step() for _ in range(steps)]
        return sum(res) / len(res), sum(r > 0 for r in res) / len(res)

    def main(self, steps: int, batch_size: int, test_steps: int = 10_000):
        for i in range(steps):
            self.step()
            (i % batch_size) or self.state_res(self.root_board)

        tr, tw = self.test(test_steps)
        print(f"testing: {tw}, {tr}")

    def state_res(self, board: Connect4Board):
        ucb = (lambda res_, n_: res_.r / res_.games + self.c * pow(math.log(n_) / res_.games, .5))
        n = sum(res.games for res in self.state_action[board].values())
        w = sum(res.wins for res in self.state_action[board].values())
        r = sum(res.r for res in self.state_action[board].values())
        astr = ' | '.join(
            f"{a} {int(100 * res.games / n)}%@{100 * res.wins / res.games:.1f} {ucb(res, n):.2f}"
            for a, res in self.state_action[board].items()
        )
        print(f"{w=} {n=} {r / n:.2f} | {astr}")

    def play_cpu(self):
        board = self.root_board

        while (r := board.terminal()) is None:
            # board.show()
            # a = int(input(f"pick: {board.free_columns()}"))
            a = min(board.free_columns(), key=lambda x: abs(3 - x))
            assert a in board.free_columns()
            board = board.put(a, True)
            if (r := board.terminal()) is not None:
                break
            board = board.put(random.choice(board.free_columns()), False)
        return r

    def rx(self, train_steps: int, test_steps: int, batches: int):
        results = []
        for batch in range(batches):
            print(f"{self.gamma} {self.c} | {batch}/{batches}")
            for _ in range(train_steps):
                self.step()
            reward_mu, win_mu = self.test(test_steps)
            results.append({"step": batch * train_steps, "reward_mu": reward_mu, "win_mu": win_mu})
        return results


def _grid_inner(c: float, gamma: float):
    f = Path(__file__).parent / f"c_{c}_gamma_{gamma}.csv"
    res = MCTS(Connect4Board.new(), c=c ** .5, gamma=gamma).rx(train_steps=5000, test_steps=1000, batches=20)
    # printf(f"completed {c=}, {gamma=}, writing to f={f}")
    pd.DataFrame(res).to_csv(f)
    # printf(f"done: {f}")


def grid():
    hypers = [
        (c, gamma)
        for c in (2, 10, 100, 625, 2, 10, 100, 625, 2, 10, 100, 625)
        for gamma in (1, .99, .95, .9, .5)
    ]

    with ProcessPoolExecutor(max_workers=4) as ppool:
        procs = [ppool.submit(_grid_inner, c=c, gamma=gamma) for c, gamma in hypers]

        for i in range(len(procs)):
            procs[i].result()
            (c, gamma) = hypers[i]
            printf(f"COMPLETED! {c=} {gamma=} {datetime.datetime.now()}")


def test_mp_inner():
    time.sleep(1 + random.random())
    printf("finished task inner")


def testing_mp():
    with ProcessPoolExecutor(max_workers=4) as ppool:
        procs = [ppool.submit(test_mp_inner) for _ in range(400)]

        for proc in procs:
            proc.result()
            printf(f"fin {datetime.datetime.now()}")


if __name__ == '__main__':
    grid()
    # testing_mp()
    # CDIR = Path(__file__).parent
    # grid()
    # MCTS(MCTS.default_board()).main(200, 100, 1000)
    # MCTS(MCTS.default_board_2()).main(200, 100, 1000)
    # MCTS(Connect4Board.new()).main(steps=1000, batch_size=499, test_steps=1)
    # MCTS(Connect4Board.new()).main(steps=200_000, batch_size=1000, test_steps=1000)
    # results = [MCTS(Connect4Board.new()).play_cpu() for _ in range(10_000)]
    # print(sum(results) / len(results))
