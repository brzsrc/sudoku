from typing import List, Dict, Set
from heuristics import jw_os, jw_ts, mom
import os
from DIMACS_parser import tt_to_dimacs, save_dimacs, DIMACS_reader,to_DIMACS,to_DIMACS_Sixteen,DIMACS_reader_Sixteen

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



def dpll(solver: Dict, clauses: List[Set], symbols: Set[str], heuristics: str = 'rand'):
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
                    if(solver[easy_case] == True):
                        clauses.remove(clause)
                    else:
                        clause.remove(easy_case)

                easy_case_neg = '-'+easy_case        
                if(easy_case_neg in clause):
                    if(solver[easy_case] == False):
                        clauses.remove(clause)
                    else:
                        clause.remove(easy_case_neg)    
                if len(clause) == 0:
                    return solver, False

        easy_cases = check_easy_cases(clauses, symbols, solver)    

    #split
    if(len(symbols) != 0): 
        # symbol = symbols.copy().pop()
        # clauses_pos = clauses + [{symbol}]
        # clauses_neg = clauses + [{'-'+symbol}]
        clauses_fst, clauses_snd = split(clauses, symbols, heuristics)
    else:    
        assert len(clauses) == 0
        clauses_fst = clauses.copy()
        clauses_snd = clauses.copy()

    solver_fst, res_fst = dpll(solver, clauses_fst, symbols)
    if(res_fst == True):
        return solver_fst, True
    else:
        return dpll(solver, clauses_snd, symbols)
    

if __name__ == '__main__':
    # symbols, clauses = DIMACS_reader_Sixteen(f"16by16_cnf/16by16_1.cnf")
    # print('307' in symbols)
    # solver, if_solved = dpll({}, clauses, symbols, 'jw_os')

    cnf_files = os.listdir("9by9_cnf")
    for i in cnf_files:
        symbols, clauses = DIMACS_reader(f"9by9_cnf/{i}")
        print(len(symbols), len(clauses))
        solver, if_solved = dpll({}, clauses, symbols, 'mom')
        print(if_solved)
        # print(solver)
        dimacs_content = tt_to_dimacs(solver,if_solved)
        save_dimacs(dimacs_content, f'{i}_solution')
        

