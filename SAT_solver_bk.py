from typing import List, Dict, Set

def check_easy_cases(clauses, symbols, solver):
    #a list of symbols
    easy_cases = []
    for clause in clauses:
        #unit clause: {literal}
        if len(clause) == 1:
            literal = clause.copy().pop()
            # if('132' == literal or '-132' == literal):
            #     print(f"easy_case_neg: 11: {literal}")      
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


def dpll(solver: Dict, clauses: List[Set], symbols: List[str]):
    # print(clauses)
    if len(clauses) == 0:
        return solver, True
    
    if len(clauses[0]) == 0:
        return solver, False
    
    #a list of symbols
    easy_cases = check_easy_cases(clauses, symbols, solver)  

    # if('132' in easy_cases):
    #     print(f"easy_case_neg: clauses 34: {clauses}")      

    while(len(easy_cases) != 0):
        #simplify
        for easy_case in easy_cases:
            #clauses.copy(): keep the old clauses for lopp while we need to remove literal from clauses
            for clause in clauses.copy():
                #check if this clause contains the easy_case literal
                # if(easy_case == '132' and clause == {'-132', '-133'}):
                #     print(f"easy_case_neg: clause 39: {clause}")
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
                        # if(easy_case_neg == '-132' and clause == {'-132', '-133'}):
                        #     print(f"easy_case_neg: clause before: {clause}")
                        clause.remove(easy_case_neg)    
                        # if(easy_case_neg == '-132' and clause == {'-132', '-133'}):
                        #     print(f"easy_case_neg: clause after: {clause}")
                if len(clause) == 0:
                    return solver, False

        easy_cases = check_easy_cases(clauses, symbols, solver)    
        # if('132' in easy_cases):
        #     print(f"easy_case_neg: clauses 34: {clauses}")   

    # print(f"solver: {solver}")
    if(len(symbols) != 0): 
        symbol = symbols.copy().pop()

        clauses_pos = clauses + [{symbol}]
        clauses_neg = clauses + [{'-'+symbol}]
    else:    
        assert len(clauses) == 0
        clauses_pos = clauses.copy()
        clauses_neg = clauses.copy()

    solver_pos, res_pos = dpll(solver, clauses_pos, symbols)
    if(res_pos == True):
        return solver_pos, True
    else:
        return dpll(solver, clauses_neg, symbols)