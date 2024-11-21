import json
import os
import pickle
import time
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from concurrent.futures.process import ProcessPoolExecutor as PPool
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Set, List, Dict, Optional, cast, Callable

import pandas as pd

from DIMACS_parser import DIMACS_reader, DIMACS_reader_Sixteen

Literal = int


class Unsolvable(Exception):
    pass


class Heuristic(ABC):

    @abstractmethod
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal: ...


class Rand(Heuristic):
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        return min(clauses[min(clauses)])


class Max(Heuristic):
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        return max(lit for cl in clauses.values() for lit in cl)


class JWOneSide(Heuristic):

    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        lit_weights = defaultdict(int)
        for cl in clauses.values():
            for lit in cl:
                lit_weights[lit] += 2 ** -len(cl)

        return max(lit_weights, key=lit_weights.get)


class JWTwoSide(Heuristic):

    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        lit_weights = defaultdict(int)
        for cl in clauses.values():
            for lit in cl:
                lit_weights[lit] += 2 ** -len(cl)

        lit = max(lit_weights, key=lambda x: lit_weights[x] + lit_weights[-x])
        return max([lit, -lit], key=lit_weights.get)


class JWTwoSideMin(Heuristic):

    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        lit_weights = defaultdict(int)
        for cl in clauses.values():
            for lit in cl:
                lit_weights[lit] += 2 ** -len(cl)

        lit = min(lit_weights, key=lambda x: lit_weights[x] + lit_weights[-x])
        return min([lit, -lit], key=lit_weights.get)


class DLCS(Heuristic):

    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        occ = Counter(lit for lit_l in clauses.values() for lit in lit_l)
        pos_lit = set(map(abs, occ))
        max_pos_lit = max(pos_lit, key=lambda lit: occ[lit] + occ[-lit])
        return (-max_pos_lit, max_pos_lit)[occ[max_pos_lit] > occ[-max_pos_lit]]


class DLIS(Heuristic):

    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        occ = Counter(lit for lit_l in clauses.values() for lit in lit_l)
        pos_lit = set(map(abs, occ))
        max_pos_lit = max(pos_lit, key=lambda lit: max(occ[lit], occ[-lit]))
        return (-max_pos_lit, max_pos_lit)[occ[max_pos_lit] > occ[-max_pos_lit]]


class MOM(Heuristic):
    k = 2
    
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        f = defaultdict(int)
        formula = cast(Callable, lambda x: (f[x] + f[-x]) * 2 ** self.k + f[x] * f[-x])
        len_dict = {i: len(cl) for i, cl in clauses.items()}
        min_len = min(len_dict.values())

        min_cl = [i for i, len_ in len_dict.items() if len_ == min_len]

        for cl in min_cl:
            for lit in clauses[cl]:
                f[lit] += 1

        return max(list(f), key=formula)


class SodokuSolver:
    clauses: Dict[int, List[Literal]]
    solution: Dict[Literal, bool]
    backtracks: int
    max_depth: int
    solved_literals: int
    n_steps: int

    def __init__(self, filename: str, heuristic: Heuristic):
        all_lit_pos, clauses = (DIMACS_reader, DIMACS_reader_Sixteen)['16by16' in filename](filename)
        clauses = [{int(lit) for lit in cl} for cl in clauses]

        self.clauses = {i: [lit for lit in cl] for i, cl in enumerate(clauses)}
        self.lit_where = defaultdict(list)
        self.solution = {}
        self.heuristic = heuristic
        self._populate_lit_where()
        self.backtracks = 0
        self.max_depth = 0
        self.pure_literals = 0
        self.solved_literals = 0
        self.n_steps = 0

    def _populate_lit_where(self):
        for i, cl in self.clauses.items():
            for lit in cl:
                self.lit_where[lit].append(i)

    def solve_literal(self, literal: Literal) -> None:
        if -literal in self.solution:
            raise Unsolvable()
        assert literal not in self.solution

        for clause_idx in self.lit_where[literal]:
            if clause_idx not in self.clauses:
                continue
            del self.clauses[clause_idx]

        for clause_idx in self.lit_where[-literal]:
            if clause_idx not in self.clauses:
                continue

            self.clauses[clause_idx].remove(-literal)
            if not self.clauses[clause_idx]:
                raise Unsolvable()

        self.solution[abs(literal)] = literal > 0
        self.solved_literals += 1

    def simplify(self) -> None:
        change = True
        while change:
            change = False
            for cl_idx, cl in self.clauses.items():
                if len(cl) == 1:
                    del self.clauses[cl_idx]
                    self.solve_literal(cl[0])
                    change = True
                    break

    def remove_pure_literals(self):
        for literal in self.lit_where:
            if abs(literal) in self.solution:
                continue
            if not any(map(self.clauses.__contains__, self.lit_where[-literal])):
                self.solve_literal(literal)
                self.pure_literals += 1

    def solve(self, depth: int = 0):
        self.n_steps += 1
        try:
            self.remove_pure_literals()
        except Unsolvable as e:
            raise Unsolvable(f"Unsolvable in `remove_pure_literals` at depth {depth}") from e

        self.max_depth = max(self.max_depth, depth)
        self.simplify()

        if not self.clauses:
            return self.solution

        lit = self.heuristic.choose(self.clauses)
        sol = self.solution.copy()
        clauses = deepcopy(self.clauses)

        try:
            self.solve_literal(lit)
            return self.solve(depth + 1)

        except Unsolvable:
            self.solution = sol
            self.clauses = clauses
            self.backtracks += 1
            self.solve_literal(-lit)

            return self.solve(depth + 1)


@dataclass
class ExperimentResult:
    filename: str
    solvable: bool
    time_elapsed: float
    # literals: str
    max_depth: int
    backtracks: int
    pure_literals: int
    solved_literals: int
    n_steps: int


def run_experiment(filename: str, heuristic: Optional[Heuristic] = None, verbose: bool = False):
    h_name = type(heuristic).__name__
    out_file = (Path(__file__).parent / f"16x16_res/{h_name}_{filename}").with_suffix('.json')
    os.makedirs(out_file.parent, exist_ok=True)
    if out_file.exists():
        with open(out_file, 'r') as f:
            return ExperimentResult(**json.load(f))

    solver = SodokuSolver(filename, heuristic or Rand())
    t = time.perf_counter()

    # try:
    solution = solver.solve()
    solvable = True

    # except Unsolvable:
    #     solvable = False
    #     solution = {}

    res = ExperimentResult(
        filename=filename,
        solvable=solvable,
        time_elapsed=time.perf_counter() - t,
        # literals='|'.join(map(str, {literal for literal, truth in solution.items() if truth})),
        max_depth=solver.max_depth,
        backtracks=solver.backtracks,
        pure_literals=solver.pure_literals,
        solved_literals=solver.solved_literals,
        n_steps=solver.n_steps,
    )

    if verbose:
        print(
            f"{h_name}"
            f" | {filename=}"
            f" | {solvable=}"
            f" | {res.time_elapsed=:6f}"
            f" | {res.backtracks=}"
            f" | {res.pure_literals=}"
            f" | {res.max_depth=}"
            )
    else:
        print(f"{h_name} | {filename[5:-4]}")

    out_file.touch(exist_ok=False)
    with open(out_file, 'w') as f:
        json.dump(asdict(res), f)
    return res


def _get_files(directory: str= "4by4_cnf"):
    return [f'{directory}/{file}' for file in sorted(os.listdir(directory), key=lambda x: int(x[5:-4]))]


def main():
    cpu_count = 4
    map_ = (lambda x: x)(lambda f, *iters: [f(*args) for args in zip(*iters)])
    # avg = lambda x: sum(x) / len(x)

    with PPool(max_workers=cpu_count) as pool:
        for directory in ["4by4_cnf", "9by9_cnf", "16by16_cnf"][2:]:
            files = _get_files(directory)

            for h in [Max, Rand, MOM, JWOneSide, JWTwoSide, DLCS, DLIS]:
                h_name = h.__name__
                results: List[ExperimentResult] = list(
                    (map_, pool.map)[cpu_count > 1](run_experiment, files, [h()] * len(files))
                )
                results_df = pd.DataFrame({
                    "filename": [res.filename for res in results],
                    "solvable": [res.solvable for res in results],
                    "time_elapsed": [res.time_elapsed for res in results],
                    # "literals": [list(res.literals) for res in results],
                    "max_depth": [res.max_depth for res in results],
                    "backtracks": [res.backtracks for res in results],
                    # "pure_literals": [res.pure_literals for res in results],
                    "solved_literals": [res.solved_literals for res in results],
                    "n_steps": [res.n_steps for res in results],
                    "heuristic_used": [h_name] * len(results)
                })
                results_df.to_csv(f"results/{directory}_{h_name}.csv")


def find_hard_files():
    for directory in ["4by4_cnf", "9by9_cnf", "16by16_cnf"][2:]:
        files = _get_files(directory)
        for file in files:

            for h in [Rand, MOM, JWOneSide, JWTwoSide, DLCS, DLIS][:1]:
                h_name = h.__name__
                out_file = (Path(__file__).parent / f"16x16_res/{h_name}_{file}").with_suffix('.json')
                if not out_file.exists():
                    print(out_file)


def read_easy_files():
    for directory in ["4by4_cnf", "9by9_cnf", "16by16_cnf"][2:]:
        files = _get_files(directory)
        for file in files:

            for h in [Max, Rand, MOM, JWOneSide, JWTwoSide, DLCS, DLIS][1:2]:
                h_name = h.__name__
                out_file = (Path(__file__).parent / f"16x16_res/{h_name}_{file}").with_suffix('.json')
                if not out_file.exists():
                    continue
                try:
                    with open(out_file, 'r') as f:
                        _ = json.load(f)
                except Exception as e:
                    # raise e
                    print(f"file {out_file} had problem")


    # files = _get_files("9by9_cnf")
    # heuristics = [DLCS, DLIS, Rand]  # Rand, MOM, JWOneSide, JWTwoSide, JWTwoSideMin,
    #
    # t = time.perf_counter()
        # result_futures = {
        #     h: (map_, pool.map)[cpu_count > 1](run_experiment, files, [h()] * len(files))
        #     for h in heuristics
        # }
        # results = {k: list(v) for k, v in result_futures.items()}
    #
    # for h, res_list in results.items():
    #     res_list: List[ExperimentResult]
    #     print(f"{h_name}: {avg([r.backtracks for r in res_list])}")

if __name__ == '__main__':
    # find_hard_files()
    # read_easy_files()
    main()



