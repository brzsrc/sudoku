
import os
import time
from typing import List, Dict, Set
from heuristics import jw_os, jw_ts, mom
from DIMACS_parser import tt_to_dimacs, save_dimacs, DIMACS_reader,to_DIMACS,to_DIMACS_Sixteen,DIMACS_reader_Sixteen
from measure import Metrics

def _inverse(literal: str) -> str:
    return literal.removeprefix('-') if literal.startswith('-') else f'-{literal}'

def is_tautology(clause):
    return any(_inverse(literal) in clause for literal in clause)

def check_pure_literal(clauses, symbols, solver):
    easy_cases = []
    for symbol in symbols.copy():
        is_contained_pos = False
        is_contained_neg = False

        for clause in clauses:
            if (symbol in clause)and not(is_contained_pos):
                is_contained_pos = True
            if ('-'+symbol in clause)and not(is_contained_neg):
                is_contained_neg = True
        
        if (is_contained_pos and not(is_contained_neg)):
            if symbol not in solver:
                solver[symbol] = True
                symbols.remove(symbol)
            easy_cases.append(symbol)
            
        if (not(is_contained_pos) and is_contained_neg):    
            if symbol not in solver:
                solver[symbol] = False
                symbols.remove(symbol)
            easy_cases.append(symbol)
    return easy_cases
        

def check_easy_cases(clauses, symbols, solver):
    #a list of symbols
    easy_cases = []
    for clause in clauses.copy():
        #tautology rule
        if(is_tautology(clause)):
            clauses.remove(clause)

        #unit clause: {literal}
        elif len(clause) == 1:
            literal = clause.copy().pop()    
            if literal[0] == '-':
                if literal[1:] not in solver:
                    solver[literal[1:]] = False
                    symbols.remove(literal[1:])
                easy_cases.append(literal[1:])
            else:
                if literal not in solver:
                    solver[literal] = True
                    symbols.remove(literal)
                easy_cases.append(literal)
    return  easy_cases


def split(clauses: List[Set], symbols: Set[str], heuristics:str = 'rand'):
    symbol_list = list(symbols)
    if(heuristics == 'jw_os'):
        symbol, is_true = jw_os(clauses, symbol_list)
    elif(heuristics == 'jw_ts'):
        symbol, is_true = jw_ts(clauses, symbol_list)
        print(symbol, is_true)
    elif(heuristics == 'mom'):    
        symbol, is_true = mom(clauses, symbol_list, 2)
        pass
    else:
        #heuristics is 'rand':
        symbol = symbols.copy().pop()
        is_true = True
    if(is_true):
        clauses_fst = clauses + [{symbol}]
        clauses_snd = clauses + [{'-'+symbol}]
    else:
        clauses_fst = clauses + [{'-'+symbol}]
        clauses_snd = clauses + [{symbol}]
    
    return clauses_fst, clauses_snd


def dpll(solver: Dict, clauses: List[Set], symbols: Set[str], heuristics: str = 'rand', metrics =None):
    if metrics is None:
        metrics = Metrics()
    if len(clauses) == 0:
        return solver, True

    if len(clauses[0]) == 0:
        return solver, False

    #a list of symbols
    easy_cases = check_easy_cases(clauses, symbols, solver)
    easy_cases = easy_cases + check_pure_literal(clauses, symbols, solver)

    while(len(easy_cases) != 0):
        #simplify
        for easy_case in easy_cases:
            #clauses.copy(): keep the old clauses for lopp while we need to remove literal from clauses
            for clause in clauses.copy():
                #check if this clause contains the easy_case literal
                if(easy_case in clause):
                    if(solver[easy_case]):
                        clauses.remove(clause)
                    else:
                        clause.remove(easy_case)

                easy_case_neg = '-'+easy_case
                if(easy_case_neg in clause):
                    if not solver[easy_case]:
                        clauses.remove(clause)
                    else:
                        clause.remove(easy_case_neg)
                if len(clause) == 0:
                    return solver, False

        easy_cases = check_easy_cases(clauses, symbols, solver)

    #split
    if(len(symbols) != 0):
        clauses_fst, clauses_snd = split(clauses, symbols, heuristics)
    else:
        assert len(clauses) == 0
        clauses_fst = clauses.copy()
        clauses_snd = clauses.copy()

    solver_fst, res_fst = dpll(solver, clauses_fst, symbols, heuristics, metrics)
    if(res_fst == True):
        return solver_fst, True
    else:
        print("Backtracking...")
        metrics.increment_backtrack_counter()
        return dpll(solver, clauses_snd, symbols,heuristics, metrics)
    

if __name__ == '__main__':
    # to_DIMACS_Sixteen("16by16_online/1.txt","sudoku-rules-16x16.txt")
    # symbols, clauses = DIMACS_reader_Sixteen(f"16by16_cnf2/16by16_1.cnf")
    # solver, if_solved = dpll({}, clauses, symbols, 'jw_os')
    # print(if_solved)
    # # print(solver)
    # dimacs_content = tt_to_dimacs(solver)
    # save_dimacs(dimacs_content, f'16by16_online/solution')


    ## 4*4
    # to_DIMACS(4,"4x4.txt","sudoku-rules-4x4.txt")    
    # cnf_files = os.listdir("4by4_cnf")
    # for i in cnf_files:
    #     symbols, clauses = DIMACS_reader(f"4by4_cnf/{i}")
    #     solver, if_solved = dpll({}, clauses, symbols, 'jw_ts')
    #     dimacs_content = tt_to_dimacs(solver)
    #     save_dimacs(dimacs_content, f'4by4/{i}_solution')

    # 9*9
    # to_DIMACS(9,"testsets/1000 sudokus.txt","sudoku-rules-9x9.txt")
    cnf_files = os.listdir("9by9_cnf")
    for i in cnf_files:
        symbols, clauses = DIMACS_reader(f"9by9_cnf/{i}")
        # print(len(symbols), len(clauses))
        metrics = Metrics()
        metrics.start_timing()
        solver, if_solved = dpll({}, clauses, symbols, 'jw_os', metrics)
        metrics.end_timing()
        print(f"if_solved: {if_solved}, time elapse: {metrics.get_time_interval()}, "
              f"# of bt: {metrics.get_backtrack_counter()}")
        # print(solver)
        dimacs_content = tt_to_dimacs(solver, if_solved)
        save_dimacs(dimacs_content, f'9by9/{i}_solution')



    # cnf_files = os.listdir("16by16_cnf")
    # for i in cnf_files:
    #     symbols, clauses = DIMACS_reader_Sixteen(f"16by16_cnf/{i}")
    #     solver, if_solved = dpll({}, clauses, symbols, 'mom')
    #     print(if_solved)
    #     # print(solver)
    #     dimacs_content = tt_to_dimacs(solver)
    #     save_dimacs(dimacs_content, f'16by16/{i}_solution')
        

