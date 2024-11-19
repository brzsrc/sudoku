import os
import time
from concurrent.futures import as_completed
from typing import Set, List, Dict

from concurrent.futures.process import ProcessPoolExecutor as PPool

from DIMACS_parser import DIMACS_reader


def _batches(li: List, n: int):
    return [li[i:i+n] for i in range(0, len(li), n)]


def _neg(literal: str) -> str:
    return literal[1:] if literal.startswith('-') else '-' + literal


def _is_neg(literal: str) -> bool:
    return literal.startswith('-')


def _as_pos(literal: str, value: bool):
    return (literal[1:], not value) if _is_neg(literal) else (literal, value)


def _copy_clauses(clauses: List[Set[str]]) -> List[Set[str]]:
    return [clause.copy() for clause in clauses]


def _log(*args, **kwargs):
    print(*args, **kwargs)


def _show_soduku(literals: List[str]):
    sudoku = [list('#' * 9) for _ in range(9)]
    for lit in literals:
        sudoku[int(lit[0]) - 1][int(lit[1]) - 1] = lit[2]
    print('\n'.join(' '.join(row) for row in sudoku))


def f_sol(clauses: List[Set[str]], cur_sol: Dict[str, bool], pool, depth: int = 0):
    indent = ' ' * depth

    while True:
        all_literals = {lit for cl in clauses for lit in cl}
        unit_clauses = {i for i, cl in enumerate(clauses) if len(cl) == 1}
        pure_literals = {lit for lit in all_literals if _neg(lit) not in all_literals}
        tauts = {i for i, cl in enumerate(clauses) if any(_neg(lit) in cl for lit in cl)}

        if not (tauts or unit_clauses or pure_literals):
            break
        _log(f"{indent}crunching cl: {len(clauses)} | sol: {len(cur_sol)} | {len(unit_clauses)} ...")
        # print(f"units: {len(unit_clauses)}, pures: {len(pure_literals)}, cl: {len(clauses)}, taut: {len(tauts)}")

        true_lit = pure_literals.copy()
        for i in unit_clauses:
            true_lit.update(clauses[i])

        false_lit = {_neg(lit) for lit in true_lit}

        _log(f"{indent} true_lit={len(true_lit)}")
        cur_sol.update(dict(_as_pos(lit, True) for lit in true_lit))

        for i, idx in enumerate(sorted(tauts | unit_clauses)):
            del clauses[idx - i]

        to_rm = []
        for i, cl in enumerate(clauses):
            tr = cl.intersection(true_lit)
            if tr:
                to_rm.append(i)
            else:
                cl.difference_update(false_lit)
                if not cl:
                    _log(f"{indent}nope!")
                    return {}, False

        for i, idx in enumerate(to_rm):
            del clauses[idx - i]

        if not clauses:
            return cur_sol, True

    _log(f"{indent}hard...")
    all_literals = {lit for cl in clauses for lit in cl}
    assert not any(map(cur_sol.__contains__, all_literals)), "literal exists which has value already determined"

    if not depth:
        with PPool() as pool:
            futs = [
                pool.submit(f_sol, [{lit_}] + _copy_clauses(clauses), cur_sol.copy(), None, depth + 1)
                for lit in all_literals
                for lit_ in (lit, _neg(lit))
            ]
            for fut in as_completed(futs):
                sol, status = fut.result()
                if status:
                    return sol, status

    return {}, False


def main():
    cnf_files = os.listdir("9by9_cnf")
    for filename in sorted(cnf_files, key=lambda x: int(x[5:-4])):
        print(filename)
        symbols, clauses = DIMACS_reader(f"9by9_cnf/{filename}")
        t = time.perf_counter()
        sol, status = f_sol(clauses, {}, None)
        print(f"{filename}: {status}, {round(time.perf_counter() - t, 3)}")

if __name__ == '__main__':
    main()



