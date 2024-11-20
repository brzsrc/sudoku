import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures.process import ProcessPoolExecutor as PPool
from copy import deepcopy
from dataclasses import dataclass
from typing import Set, List, Dict, Optional, cast, Callable

from DIMACS_parser import DIMACS_reader

Literal = int


class Heuristic(ABC):

    @abstractmethod
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal: ...


class Rand(Heuristic):
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        return min(clauses[min(clauses)])


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
    depth_profile: List[int]

    def __init__(self, filename: str, heuristic: Heuristic):
        all_lit_pos, clauses = DIMACS_reader(filename)
        clauses = [{int(lit) for lit in cl} for cl in clauses]

        self.clauses = {i: [int(lit) for lit in cl] for i, cl in enumerate(clauses)}
        self.lit_where = defaultdict(list)
        self.solution = {}
        self.heuristic = heuristic
        self._populate_lit_where()
        self.backtracks = 0
        self.depth_profile = []

    def _populate_lit_where(self):
        for i, cl in self.clauses.items():
            for lit in cl:
                self.lit_where[lit].append(i)

    def solve_literal(self, literal: Literal) -> bool:
        for clause_idx in self.lit_where[literal]:
            if clause_idx not in self.clauses:
                continue
            del self.clauses[clause_idx]

        for clause_idx in self.lit_where[-literal]:
            if clause_idx not in self.clauses:
                continue

            self.clauses[clause_idx].remove(-literal)
            if not self.clauses[clause_idx]:
                return False

        self.solution[abs(literal)] = literal > 0
        return True

    def simplify(self) -> bool:
        change = True
        while change:
            change = False
            for cl_idx, cl in self.clauses.items():
                if len(cl) == 1:
                    lit = cl[0]
                    del self.clauses[cl_idx]
                    status = self.solve_literal(lit)
                    if not status:
                        return False
                    change = True
                    break
        return True

    def solve(self, depth: int = 0) -> bool:
        self.depth_profile.append(depth)
        status = self.simplify()

        if not status:
            return False

        if not self.clauses:
            return True

        lit = self.heuristic.choose(self.clauses)
        sol = self.solution.copy()
        clauses = deepcopy(self.clauses)

        if self.solve_literal(lit) and self.solve(depth + 1):
            return True

        self.solution = sol
        self.clauses = clauses
        self.backtracks += 1
        return self.solve_literal(-lit) and self.solve(depth + 1)


@dataclass
class ExperimentResult:
    filename: str
    solvable: bool
    time_elapsed: float
    literals: Set[int]
    depth_profile: List[int]
    backtracks: int


def run_experiment(filename: str, heuristic: Optional[Heuristic] = None, verbose: bool = True):
    solver = SodokuSolver(filename, heuristic or Rand())
    t = time.perf_counter()
    solvable = solver.solve()
    result = ExperimentResult(
        filename=filename,
        solvable=solvable,
        time_elapsed=time.perf_counter() - t,
        literals={literal for literal, truth in solver.solution.items() if truth},
        depth_profile=solver.depth_profile,
        backtracks=solver.backtracks
    )
    verbose and print(f"{type(heuristic).__name__}{filename} {solvable} {result.time_elapsed:6f} | {result.backtracks}")
    return result


def _get_files(directory: str= "4by4_cnf"):
    return [f'{directory}/{file}' for file in sorted(os.listdir(directory), key=lambda x: int(x[5:-4]))]


def main():
    files = _get_files("9by9_cnf")
    heuristics = [Rand, MOM]  # JWOneSide, JWTwoSide,
    cpu_count = os.cpu_count()
    print(f"running experiment with {cpu_count} cores")
    map_ = lambda f, *iters: [f(*args) for args in zip(*iters)]

    # t = time.perf_counter()
    with PPool(max_workers=cpu_count) as pool:
        result_futures = {
            h: (map_, pool.map)[cpu_count > 1](run_experiment, files, [h()] * len(files))
            for h in heuristics
        }
        results = {k: list(v) for k, v in result_futures.items()}

    import matplotlib.pyplot as plt

    for h, res_list in results.items():
        h_name = h.__name__
        res_list: List[ExperimentResult]
        elapsed = [r.time_elapsed for r in res_list]
        backtracks = [r.backtracks for r in res_list]
        n_steps = [len(r.depth_profile) for r in res_list]

        plt.bar(list(range(len(elapsed))), elapsed, label='time elapsed')
        plt.title(f"{h_name}: elapsed")
        plt.show()

        plt.bar(list(range(len(backtracks))), backtracks, label='num of backtracks')
        plt.title(f"{h_name}: backtracks")
        plt.show()



    # avg = sum(res.time_elapsed for res in results) / len(results)
    # print(f"time elapsed per result: {avg:3f}")
    # print(f"time for the entire sim {time.perf_counter() - t}")


if __name__ == '__main__':
    main()



