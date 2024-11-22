import json
import os
import random
import time
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from concurrent.futures import as_completed, Future
from concurrent.futures.process import ProcessPoolExecutor as PPool
from copy import deepcopy
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Optional, cast, Callable

import pandas as pd

from DIMACS_parser import DIMACS_reader, DIMACS_reader_Sixteen

Literal = int


class Heuristic(ABC):

    @abstractmethod
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal: ...


class Rand(Heuristic):
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        return random.choice([lit for cl in clauses.values() for lit in cl])


class RandPos(Heuristic):
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        return random.choice([lit for cl in clauses.values() for lit in cl if lit >= 0])


class Max(Heuristic):
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        return max(lit for cl in clauses.values() for lit in cl)


class FirstPos(Heuristic):
    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        return [lit for cl in clauses.values() for lit in cl][0]


class JWOneSide(Heuristic):

    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        lit_weights = defaultdict(int)
        for cl in clauses.values():
            for lit in cl:
                lit_weights[lit] += 2 ** -len(cl)

        return max(lit_weights, key=lit_weights.get)


class JWOneSidePos(Heuristic):

    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        lit_weights = defaultdict(int)
        for cl in clauses.values():
            for lit in cl:
                if lit < 0:
                    continue
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


class JWTwoSidePos(Heuristic):

    def choose(self, clauses: Dict[int, List[Literal]]) -> Literal:
        lit_weights = defaultdict(int)
        for cl in clauses.values():
            for lit in cl:
                lit_weights[lit] += 2 ** -len(cl)

        return max([lit for lit in lit_weights if lit >= 0], key=lambda x: lit_weights[x] + lit_weights[-x])


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


class MOMPos(Heuristic):
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

        return max([lit for lit in f if lit >= 0], key=formula)


class SodokuSolver:
    clauses: Dict[int, List[Literal]]
    solution: Dict[Literal, bool]
    backtracks: int
    max_depth: int
    solved_literals: int
    n_steps: int
    pos_lit_chosen: int
    neg_lit_chosen: int

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
        self.pos_lit_chosen = 0
        self.neg_lit_chosen = 0

    def _populate_lit_where(self):
        for i, cl in self.clauses.items():
            for lit in cl:
                self.lit_where[lit].append(i)

    def solve_literal(self, literal: Literal) -> bool:
        if -literal in self.solution:
            return False
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
                return False

        self.solution[abs(literal)] = literal > 0
        self.solved_literals += 1
        return True

    def simplify(self) -> bool:
        change = True
        while change:
            change = False
            for cl_idx, cl in self.clauses.items():
                if len(cl) == 1:
                    del self.clauses[cl_idx]
                    if not self.solve_literal(cl[0]):
                        return False

                    change = True
                    break
        return True

    def solve(self, depth: int = 0) -> bool:
        self.n_steps += 1
        self.max_depth = max(self.max_depth, depth)

        if not self.simplify():
            return False

        if not self.clauses:
            return True

        lit = self.heuristic.choose(self.clauses)
        sol = self.solution.copy()
        clauses = deepcopy(self.clauses)

        if lit > 0:
            self.pos_lit_chosen += 1
        else:
            self.neg_lit_chosen += 1

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
    # literals: str
    max_depth: int
    backtracks: int
    pure_literals: int
    solved_literals: int
    n_steps: int
    pos_lit_chosen: int
    neg_lit_chosen: int


ALL_HEURISTICS = [
    # RandPos(),
    Max(),
    RandPos(),
    Rand(),
    MOM(),
    # JWTwoSide(),
    # JWOneSide(),
    # JWOneSidePos(),
    # JWTwoSidePos(),
    # DLCS(),
    # DLIS(),
    # MOMPos(),
]
JSON_DIR = Path(__file__).parent / "JSON_RESULTS"
CSV_DIR = Path(__file__).parent / "CSV_RESULTS"
INPUT_DIRS = ["4by4_cnf", "9by9_cnf", "16by16_cnf"]


def _get_json_file(filename: str, heuristic: Heuristic):
    return (JSON_DIR / f"{type(heuristic).__name__}/{filename}").with_suffix('.json')


def _get_files(directory: str):
    return [f'{directory}/{file}' for file in sorted(os.listdir(directory), key=lambda x: int(x[5:-4]))]


def run_experiment(filename: str, heuristic: Heuristic, verbose: bool = False):
    h_name = type(heuristic).__name__
    file = _get_json_file(filename, heuristic)

    if file.exists():
        return ExperimentResult(**json.loads(file.read_text()))

    solver = SodokuSolver(filename, heuristic or Rand())
    t = time.perf_counter()
    solvable = solver.solve()
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
        pos_lit_chosen=solver.pos_lit_chosen,
        neg_lit_chosen=solver.neg_lit_chosen,
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

    os.makedirs(file.parent, exist_ok=True)
    file.touch()
    file.write_text(json.dumps(asdict(res)))
    return res


def run_all_experiments(directory: str, first_n_files: Optional[int] = None):
    cpu_count = max(1, os.cpu_count() - 4)
    with PPool(max_workers=cpu_count) as pool:
        tasks = []
        files = _get_files(directory)
        for file in files[:first_n_files]:
            for heuristic in ALL_HEURISTICS:
                tasks.append(pool.submit(run_experiment, file, heuristic))

        for task in as_completed(tasks):
            task.result()


def collect(directory: str, heuristic: Heuristic):
    return pd.DataFrame([json.loads(_get_json_file(file, heuristic).read_text()) for file in _get_files(directory)])


# def run_and_collect(directory: str, heuristic: Heuristic):
#     cpu_count = max(1, os.cpu_count() - 4)
#     files = _get_files(directory)
#     with PPool(max_workers=cpu_count) as pool:
#         results = pool.map(run_experiment, files, [heuristic] * len(files))
#         return pd.DataFrame(list(map(asdict, results)))


def count_missing():
    for directory in INPUT_DIRS:
        files = _get_files(directory)
        print(directory.center(80, '='))
        for heuristic in ALL_HEURISTICS:
            missing = 0
            for file in files:
                missing += not _get_json_file(file, heuristic).exists()
            print(f"{type(heuristic).__name__}/{directory}: {missing} / {len(files)} missing")


def run_directory(directory: str):
    run_all_experiments(directory)

    for heuristic in ALL_HEURISTICS:
        csv_file = (CSV_DIR / f"{type(heuristic).__name__}/{directory}").with_suffix('.csv')
        os.makedirs(csv_file.parent, exist_ok=True)
        collect(directory, heuristic).to_csv(csv_file)


# def main():
#     cpu_count = max(1, os.cpu_count() - 4)
#     map_ = (lambda x: x)(lambda f, *iters: [f(*args) for args in zip(*iters)])
#     # avg = lambda x: sum(x) / len(x)
#
#     with PPool(max_workers=cpu_count) as pool:
#         for directory in ["4by4_cnf", "9by9_cnf", "16by16_cnf"][2:]:
#             files = _get_files(directory)
#
#             for h in [Max, Rand, MOM, JWOneSide, JWTwoSide, DLCS, DLIS]:
#                 h_name = h.__name__
#                 results: List[ExperimentResult] = list(
#                     (map_, pool.map)[cpu_count > 1](run_experiment, files, [h()] * len(files))
#                 )
#                 results_df = pd.DataFrame({
#                     "filename": [res.filename for res in results],
#                     "solvable": [res.solvable for res in results],
#                     "time_elapsed": [res.time_elapsed for res in results],
#                     # "literals": [list(res.literals) for res in results],
#                     "max_depth": [res.max_depth for res in results],
#                     "backtracks": [res.backtracks for res in results],
#                     # "pure_literals": [res.pure_literals for res in results],
#                     "solved_literals": [res.solved_literals for res in results],
#                     "n_steps": [res.n_steps for res in results],
#                     "heuristic_used": [h_name] * len(results)
#                 })
#                 results_df.to_csv(f"results/{directory}_{h_name}.csv")


# def main_v2():
#     cpu_count = max(1, os.cpu_count() - 4)
#     with PPool(max_workers=cpu_count) as pool:
#         for directory in ["4by4_cnf", "9by9_cnf", "16by16_cnf"][2:]:
#             files = _get_files(directory)
#             for h in [Max, Rand, MOM, JWOneSide, JWTwoSide, DLCS, DLIS]:
#                 for file in files:
#                     pool.submit(run_experiment, file, h())


# def find_hard_files():
#     for directory in ["4by4_cnf", "9by9_cnf", "16by16_cnf"][2:]:
#         files = _get_files(directory)
#         for file in files:
#
#             for h in [Rand, MOM, JWOneSide, JWTwoSide, DLCS, DLIS]:
#                 h_name = h.__name__
#                 out_file = (Path(__file__).parent / f"16x16_res/{h_name}_{file}").with_suffix('.json')
#                 if not out_file.exists():
#                     print(out_file)


# def count_missing_files():
#     for directory in ["4by4_cnf", "9by9_cnf", "16by16_cnf"][2:]:
#         files = _get_files(directory)
#         for h in [Rand, MOM, JWOneSide, JWTwoSide, DLCS, DLIS]:
#             h_name = h.__name__
#             missing = 0
#             for file in files:
#                 out_file = (Path(__file__).parent / f"16x16_res/{h_name}_{file}").with_suffix('.json')
#                 missing += not out_file.exists()
#             print(f"{h_name} missing {missing}")


# def read_easy_files():
#     for directory in ["4by4_cnf", "9by9_cnf", "16by16_cnf"][2:]:
#         files = _get_files(directory)
#         for file in files:
#
#             for h in [Max, Rand, MOM, JWOneSide, JWTwoSide, DLCS, DLIS][:2]:
#                 h_name = h.__name__
#                 out_file = (Path(__file__).parent / f"16x16_res/{h_name}_{file}").with_suffix('.json')
#                 if not out_file.exists():
#                     continue
#                 try:
#                     with open(out_file, 'r') as f:
#                         _ = json.load(f)
#                 except Exception as e:
#                     # raise e
#                     print(f"file {out_file} had problem")
#                     os.remove(out_file)


if __name__ == '__main__':
    run_all_experiments(INPUT_DIRS[2], first_n_files=100)
    # count_missing()
    # _run_9x9()
