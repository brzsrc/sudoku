import os
import time
from typing import List, Dict, Set

from heuristics import jw_os, jw_ts, mom
from DIMACS_parser import tt_to_dimacs, save_dimacs, DIMACS_reader, to_DIMACS, to_DIMACS_Sixteen, DIMACS_reader_Sixteen
from measure import Metrics

def dpll(solver, clauses, symbols,metrics =None):
    if metrics is None:
        metrics = Metrics()
    if not clauses:
        return solver, True
    if any(len(clause) == 0 for clause in clauses):
        return solver, False

    # Unit Propagation and Pure Literal Elimination
    unit_clauses, pure_literals = find_unit_and_pure_literals(clauses)
    while unit_clauses or pure_literals:
        for literal in unit_clauses:
            if literal.startswith('-'):
                solver[literal[1:]] = False
            else:
                solver[literal] = True
            symbols.discard(literal.strip('-'))
        for literal in pure_literals:
            if literal.startswith('-'):
                solver[literal[1:]] = False
            else:
                solver[literal] = True
            symbols.discard(literal.strip('-'))

        clauses = simplify_clauses(clauses, solver)
        unit_clauses, pure_literals = find_unit_and_pure_literals(clauses)

    if not symbols:
        return solver, True

    symbol = symbols.pop()
    new_solver = solver.copy()

    # Recur with symbol set to True
    new_solver[symbol] = True
    metrics.increment_backtrack_counter()
    result_solver, result = dpll(new_solver, clauses, set(symbols),metrics)
    if result:
        return result_solver, True

    # Recur with symbol set to False
    new_solver = solver.copy()
    new_solver[symbol] = False
    print("Backtracking...")
    metrics.increment_backtrack_counter()
    return dpll(new_solver, clauses, set(symbols),metrics)

def find_unit_and_pure_literals(clauses):
    unit_clauses = set()
    literal_count = {}
    for clause in clauses:
        if len(clause) == 1:
            unit_clauses.update(clause)
        for literal in clause:
            if literal in literal_count:
                literal_count[literal] += 1
            else:
                literal_count[literal] = 1

    pure_literals = {literal for literal, count in literal_count.items() if literal.strip('-') not in literal_count}
    return unit_clauses, pure_literals

def simplify_clauses(clauses, solver):
    new_clauses = []
    for clause in clauses:
        new_clause = set()
        for literal in clause:
            var = literal.strip('-')
            is_neg = literal.startswith('-')
            if var in solver:
                if solver[var] is not (is_neg):
                    break  # Clause is satisfied
            else:
                new_clause.add(literal)
        else:
            new_clauses.append(new_clause)
    return new_clauses

if __name__ == '__main__':
# Initialize and load problems as required, then solve using dpll.
    # cnf_files = os.listdir("9by9_cnf")
    # for i in cnf_files:
    #     symbols, clauses = DIMACS_reader(f"9by9_cnf/{i}")
    #     # print(len(symbols), len(clauses))
    #     metrics = Metrics()
    #     metrics.start_timing()
    #     solver, if_solved = dpll({}, clauses, symbols, metrics)
    #     metrics.end_timing()
    #     print(f"if_solved: {if_solved}, time elapse: {metrics.get_time_interval()}, "
    #           f"# of bt: {metrics.get_backtrack_counter()}")
    #     import csv
    #     with open('outputs/9by9/data.csv', mode='a+', newline='') as file:
    #         writer = csv.writer(file)
    #         if file.tell() == 0:
    #             writer.writerow(['file name', 'time elapse', 'backtrack counter'])
    #
    #         writer.writerow([i, metrics.get_time_interval(), metrics.get_backtrack_counter()])
    #         file.flush()
    #
    #     print(f"Data for Person{i} written to CSV.")
    #     #
    #     dimacs_content = tt_to_dimacs(solver, if_solved)
    #     save_dimacs(dimacs_content, f'9by9/{i}_solution')
    #
    #     cnf_files = os.listdir("16by16_cnf")
    cnf_files = os.listdir("16by16_cnf")
    for i in cnf_files:
        symbols, clauses = DIMACS_reader_Sixteen(f"16by16_cnf/{i}")

        metrics = Metrics()
        metrics.start_timing()
        solver, if_solved = dpll({}, clauses, symbols, metrics)
        metrics.end_timing()
        print(f"if_solved: {if_solved}, time elapse: {metrics.get_time_interval()}, "
             f"# of bt: {metrics.get_backtrack_counter()}")
        dimacs_content = tt_to_dimacs(solver, if_solved)
        save_dimacs(dimacs_content, f'16by16/{i}_solution')
