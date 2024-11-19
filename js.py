import os
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import as_completed
from copy import deepcopy
from typing import Set, List, Dict

from concurrent.futures.process import ProcessPoolExecutor as PPool

from DIMACS_parser import DIMACS_reader


Literal = int


class Heuristic(ABC):

    @abstractmethod
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal: ...


class Rand(Heuristic):

    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        return min(clauses[min(clauses)])


class SodokuSolver:
    clauses: Dict[int, List[Literal]]

    def __init__(self, filename: str, heuristic: Heuristic):
        all_lit_pos, clauses = DIMACS_reader(filename)
        clauses = [{int(lit) for lit in cl} for cl in clauses]

        self.clauses = {i: [int(lit) for lit in cl] for i, cl in enumerate(clauses)}
        self.lit_where = defaultdict(list)
        self.solution = {}
        self.heuristic = heuristic
        self._populate_lit_where()

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

    def simplify(self):
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

    def solve(self, depth: int = 0):
        status = self.simplify()

        if not status:
            return False

        if not self.clauses:
            return True

        lit_ = self.heuristic.choose(self.clauses)
        sol = self.solution.copy()
        clauses = deepcopy(self.clauses)

        for lit in lit_, -lit_:
            try:
                assert self.solve_literal(lit)
                assert self.solve(depth + 1)
                return True

            except AssertionError:
                self.solution = sol
                self.clauses = clauses

        return False


def _log(*args, **kwargs):
    0 and print(*args, **kwargs)


def _solve_file(filename: str) -> Set[Literal]:
    solver = SodokuSolver(filename, Rand())
    assert solver.solve()
    print(f"solved {filename}")
    return {literal for literal, truth in solver.solution.items() if truth}


def main():
    # directory = "4by4_cnf"
    directory = "16by16.cnf"
    cnf_files = os.listdir(directory)
    files = [f'{directory}/{file}' for file in sorted(cnf_files, key=lambda x: int(x[5:-4]))]

    with PPool() as pool:
        pool.map(_solve_file, files)
    # check_file('9by9_cnf/9by9_3.cnf')
    # solver = SodokuSolver("9by9_cnf/9by9_3.cnf", Rand())
    # print(solver.solve())
    # print(solver.solution)
    # with PPool() as pool:
    #     futs = pool.map(check_file, files)


if __name__ == '__main__':
    main()



