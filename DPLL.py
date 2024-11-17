<<<<<<< Updated upstream
from typing import List, Dict, Set
from heuristics import jw_os, jw_ts, mom

def check_easy_cases(clauses, symbols, solver):
    #a list of symbols
    easy_cases = []
    for clause in clauses:
        #unit clause: {literal}
        if len(clause) == 1:
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


def split(clauses: List[Set], symbols: List[str], heuristics:str = 'rand'):
    print("aaaaaaaa")
    if(heuristics == 'jw_os'):
        symbol, is_true = jw_os(clauses, symbols)
    elif(heuristics == 'jw_ts'):
        symbol, is_true = jw_ts(clauses, symbols)
        print(symbol, is_true)
    elif(heuristics == 'mom'):    
        # symbol, is_true = mom(clauses, symbols, 2)
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



def dpll(solver: Dict, clauses: List[Set], symbols: List[str], heuristics: str = 'rand'):
    if len(clauses) == 0:
        return solver, True
    
    if len(clauses[0]) == 0:
        return solver, False
    
    #a list of symbols
    easy_cases = check_easy_cases(clauses, symbols, solver)  

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

    print("cccccc")
    #split
    if(len(symbols) != 0): 
        # symbol = symbols.copy().pop()
        # clauses_pos = clauses + [{symbol}]
        # clauses_neg = clauses + [{'-'+symbol}]
        print("bbbbbbbbb")
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
=======
>>>>>>> Stashed changes
