from typing import Dict, List, Set, Tuple
from DIMACS_parser import tt_to_dimacs, save_dimacs, DIMACS_reader,to_DIMACS


def check_unit_clauses(clauses: List[Set[str]], symbols: List[str], solver: Dict[str, bool]):
    """
    Check for unit clauses and update solver, symbols, and clauses accordingly.
    """
    easy_cases = []
    remaining_clauses = []

    for clause in clauses:
        if len(clause) == 1:
            literal = next(iter(clause))
            symbol = literal.lstrip('-')
            value = literal[0] != '-'

            if symbol not in solver:
                solver[symbol] = value
                symbols.remove(symbol)
                easy_cases.append(symbol)
        else:
            remaining_clauses.append(clause)

    return easy_cases, remaining_clauses, symbols


def simplify_clauses(clauses: List[Set[str]], easy_cases: List[str], solver: Dict[str, bool]):
    """
    Simplify clauses based on easy cases.
    """
    simplified_clauses = []

    for clause in clauses:
        satisfied = False
        new_clause = set()

        for literal in clause:
            symbol = literal.lstrip('-')
            if symbol in solver:
                value = solver[symbol]
                if (literal[0] == '-' and not value) or (literal[0] != '-' and value):
                    satisfied = True
                    break
            else:
                new_clause.add(literal)

        if not satisfied:
            if len(new_clause) == 0:
                return []  # Clause is empty
            simplified_clauses.append(new_clause)

    return simplified_clauses


def dpll(solver: Dict[str, bool], clauses: List[Set[str]], symbols: List[str]):
    """
    DPLL algorithm implementation.
    """
    if not clauses:
        return solver, True

    if any(len(clause) == 0 for clause in clauses):
        return solver, False

    easy_cases, clauses, symbols = check_unit_clauses(clauses, symbols, solver)
    while easy_cases:
        clauses = simplify_clauses(clauses, easy_cases, solver)
        if not clauses:
            return solver, True
        if any(len(clause) == 0 for clause in clauses):
            return solver, False
        easy_cases, clauses, symbols = check_unit_clauses(clauses, symbols, solver)

    # Choose a symbol and branch
    symbol = symbols.pop()

    # Try assigning True to the symbol
    solver_pos = solver.copy()
    solver_pos[symbol] = True
    result_solver, result = dpll(solver_pos, clauses + [{symbol}], symbols.copy())
    if result:
        return result_solver, True

    # Try assigning False to the symbol
    solver_neg = solver.copy()
    solver_neg[symbol] = False
    return dpll(solver_neg, clauses + [{'-' + symbol}], symbols.copy())


if __name__ == '__main__':
    '''
#     sudoku_clauses_4 = [{'143'}, {'234'}, {'241'}, {'311'}, {'324'}, {'413'}, {'111', '113', '114', '112'},
#                         {'-111', '-112'}, {'-111', '-113'}, {'-111', '-114'}, {'-113', '-112'}, {'-114', '-112'},
#                         {'-114', '-113'}, {'123', '121', '122', '124'}, {'-122', '-121'}, {'-121', '-123'},
#                         {'-124', '-121'}, {'-122', '-123'}, {'-122', '-124'}, {'-124', '-123'},
#                         {'132', '134', '131', '133'}, {'-131', '-132'}, {'-131', '-133'}, {'-131', '-134'},
#                         {'-132', '-133'}, {'-132', '-134'}, {'-134', '-133'}, {'144', '142', '143', '141'},
#                         {'-142', '-141'}, {'-143', '-141'}, {'-144', '-141'}, {'-143', '-142'}, {'-144', '-142'},
#                         {'-143', '-144'}, {'212', '211', '213', '214'}, {'-212', '-211'}, {'-213', '-211'},
#                         {'-214', '-211'}, {'-213', '-212'}, {'-212', '-214'}, {'-213', '-214'},
#                         {'223', '224', '221', '222'}, {'-222', '-221'}, {'-223', '-221'}, {'-224', '-221'},
#                         {'-223', '-222'}, {'-224', '-222'}, {'-223', '-224'}, {'232', '233', '231', '234'},
#                         {'-232', '-231'}, {'-231', '-233'}, {'-234', '-231'}, {'-232', '-233'}, {'-234', '-232'},
#                         {'-234', '-233'}, {'242', '241', '244', '243'}, {'-241', '-242'}, {'-241', '-243'},
#                         {'-241', '-244'}, {'-243', '-242'}, {'-244', '-242'}, {'-243', '-244'},
#                         {'314', '311', '312', '313'}, {'-312', '-311'}, {'-311', '-313'}, {'-314', '-311'},
#                         {'-312', '-313'}, {'-312', '-314'}, {'-314', '-313'}, {'322', '321', '323', '324'},
#                         {'-322', '-321'}, {'-323', '-321'}, {'-324', '-321'}, {'-322', '-323'}, {'-322', '-324'},
#                         {'-324', '-323'}, {'334', '333', '331', '332'}, {'-331', '-332'}, {'-331', '-333'},
#                         {'-331', '-334'}, {'-333', '-332'}, {'-334', '-332'}, {'-333', '-334'},
#                         {'343', '344', '341', '342'}, {'-342', '-341'}, {'-343', '-341'}, {'-341', '-344'},
#                         {'-343', '-342'}, {'-342', '-344'}, {'-343', '-344'}, {'412', '413', '411', '414'},
#                         {'-411', '-412'}, {'-411', '-413'}, {'-411', '-414'}, {'-412', '-413'}, {'-412', '-414'},
#                         {'-413', '-414'}, {'422', '423', '424', '421'}, {'-422', '-421'}, {'-423', '-421'},
#                         {'-424', '-421'}, {'-423', '-422'}, {'-422', '-424'}, {'-423', '-424'},
#                         {'434', '433', '431', '432'}, {'-431', '-432'}, {'-431', '-433'}, {'-431', '-434'},
#                         {'-433', '-432'}, {'-432', '-434'}, {'-433', '-434'}, {'441', '443', '444', '442'},
#                         {'-441', '-442'}, {'-441', '-443'}, {'-441', '-444'}, {'-442', '-443'}, {'-442', '-444'},
#                         {'-443', '-444'}, {'111', '121', '221', '211'}, {'-111', '-121'}, {'-111', '-211'},
#                         {'-111', '-221'}, {'-121', '-211'}, {'-121', '-221'}, {'-221', '-211'},
#                         {'212', '122', '222', '112'}, {'-122', '-112'}, {'-212', '-112'}, {'-222', '-112'},
#                         {'-122', '-212'}, {'-122', '-222'}, {'-212', '-222'}, {'123', '113', '223', '213'},
#                         {'-113', '-123'}, {'-113', '-213'}, {'-223', '-113'}, {'-213', '-123'}, {'-223', '-123'},
#                         {'-223', '-213'}, {'224', '114', '214', '124'}, {'-114', '-124'}, {'-114', '-214'},
#                         {'-114', '-224'}, {'-124', '-214'}, {'-224', '-124'}, {'-224', '-214'},
#                         {'241', '131', '231', '141'}, {'-131', '-141'}, {'-131', '-231'}, {'-131', '-241'},
#                         {'-231', '-141'}, {'-241', '-141'}, {'-241', '-231'}, {'232', '142', '242', '132'},
#                         {'-132', '-142'}, {'-132', '-232'}, {'-132', '-242'}, {'-232', '-142'}, {'-242', '-142'},
#                         {'-232', '-242'}, {'243', '233', '143', '133'}, {'-143', '-133'}, {'-133', '-233'},
#                         {'-243', '-133'}, {'-143', '-233'}, {'-143', '-243'}, {'-243', '-233'},
#                         {'144', '134', '244', '234'}, {'-134', '-144'}, {'-234', '-134'}, {'-134', '-244'},
#                         {'-234', '-144'}, {'-244', '-144'}, {'-234', '-244'}, {'321', '411', '311', '421'},
#                         {'-321', '-311'}, {'-411', '-311'}, {'-421', '-311'}, {'-411', '-321'}, {'-321', '-421'},
#                         {'-411', '-421'}, {'322', '422', '312', '412'}, {'-312', '-322'}, {'-312', '-412'},
#                         {'-312', '-422'}, {'-322', '-412'}, {'-322', '-422'}, {'-412', '-422'},
#                         {'423', '413', '323', '313'}, {'-323', '-313'}, {'-413', '-313'}, {'-423', '-313'},
#                         {'-323', '-413'}, {'-323', '-423'}, {'-423', '-413'}, {'424', '314', '324', '414'},
#                         {'-324', '-314'}, {'-314', '-414'}, {'-314', '-424'}, {'-324', '-414'}, {'-324', '-424'},
#                         {'-424', '-414'}, {'441', '431', '331', '341'}, {'-341', '-331'}, {'-431', '-331'},
#                         {'-441', '-331'}, {'-431', '-341'}, {'-441', '-341'}, {'-441', '-431'},
#                         {'442', '432', '332', '342'}, {'-342', '-332'}, {'-432', '-332'}, {'-442', '-332'},
#                         {'-342', '-432'}, {'-442', '-342'}, {'-442', '-432'}, {'443', '343', '433', '333'},
#                         {'-343', '-333'}, {'-433', '-333'}, {'-333', '-443'}, {'-343', '-433'}, {'-343', '-443'},
#                         {'-433', '-443'}, {'434', '334', '344', '444'}, {'-334', '-344'}, {'-434', '-334'},
#                         {'-334', '-444'}, {'-434', '-344'}, {'-344', '-444'}, {'-434', '-444'},
#                         {'111', '121', '131', '141'}, {'-111', '-121'}, {'-111', '-131'}, {'-111', '-141'},
#                         {'-131', '-121'}, {'-121', '-141'}, {'-131', '-141'}, {'132', '142', '122', '112'},
#                         {'-122', '-112'}, {'-132', '-112'}, {'-142', '-112'}, {'-122', '-132'}, {'-122', '-142'},
#                         {'-132', '-142'}, {'123', '113', '143', '133'}, {'-113', '-123'}, {'-113', '-133'},
#                         {'-143', '-113'}, {'-133', '-123'}, {'-143', '-123'}, {'-143', '-133'},
#                         {'144', '134', '114', '124'}, {'-114', '-124'}, {'-114', '-134'}, {'-114', '-144'},
#                         {'-124', '-134'}, {'-124', '-144'}, {'-134', '-144'}, {'211', '221', '231', '241'},
#                         {'-221', '-211'}, {'-231', '-211'}, {'-241', '-211'}, {'-231', '-221'}, {'-241', '-221'},
#                         {'-241', '-231'}, {'212', '242', '222', '232'}, {'-212', '-222'}, {'-232', '-212'},
#                         {'-212', '-242'}, {'-232', '-222'}, {'-222', '-242'}, {'-232', '-242'},
#                         {'223', '233', '213', '243'}, {'-223', '-213'}, {'-213', '-233'}, {'-213', '-243'},
#                         {'-223', '-233'}, {'-223', '-243'}, {'-243', '-233'}, {'224', '244', '234', '214'},
#                         {'-224', '-214'}, {'-234', '-214'}, {'-244', '-214'}, {'-234', '-224'}, {'-224', '-244'},
#                         {'-234', '-244'}, {'321', '331', '311', '341'}, {'-321', '-311'}, {'-331', '-311'},
#                         {'-341', '-311'}, {'-321', '-331'}, {'-341', '-321'}, {'-341', '-331'},
#                         {'322', '332', '312', '342'}, {'-312', '-322'}, {'-312', '-332'}, {'-312', '-342'},
#                         {'-322', '-332'}, {'-322', '-342'}, {'-342', '-332'}, {'343', '333', '323', '313'},
#                         {'-323', '-313'}, {'-333', '-313'}, {'-343', '-313'}, {'-323', '-333'}, {'-343', '-323'},
#                         {'-343', '-333'}, {'334', '344', '314', '324'}, {'-324', '-314'}, {'-314', '-334'},
#                         {'-314', '-344'}, {'-324', '-334'}, {'-324', '-344'}, {'-334', '-344'},
#                         {'441', '431', '411', '421'}, {'-411', '-421'}, {'-431', '-411'}, {'-441', '-411'},
#                         {'-431', '-421'}, {'-441', '-421'}, {'-441', '-431'}, {'412', '442', '422', '432'},
#                         {'-412', '-422'}, {'-412', '-432'}, {'-442', '-412'}, {'-432', '-422'}, {'-442', '-422'},
#                         {'-442', '-432'}, {'433', '443', '423', '413'}, {'-423', '-413'}, {'-433', '-413'},
#                         {'-443', '-413'}, {'-433', '-423'}, {'-443', '-423'}, {'-433', '-443'},
#                         {'434', '424', '444', '414'}, {'-424', '-414'}, {'-434', '-414'}, {'-444', '-414'},
#                         {'-434', '-424'}, {'-424', '-444'}, {'-434', '-444'}, {'111', '211', '411', '311'},
#                         {'-111', '-211'}, {'-111', '-311'}, {'-111', '-411'}, {'-211', '-311'}, {'-411', '-211'},
#                         {'-411', '-311'}, {'212', '112', '312', '412'}, {'-212', '-112'}, {'-312', '-112'},
#                         {'-412', '-112'}, {'-312', '-212'}, {'-412', '-212'}, {'-312', '-412'},
#                         {'113', '413', '313', '213'}, {'-113', '-213'}, {'-113', '-313'}, {'-113', '-413'},
#                         {'-213', '-313'}, {'-213', '-413'}, {'-413', '-313'}, {'314', '114', '214', '414'},
#                         {'-114', '-214'}, {'-114', '-314'}, {'-114', '-414'}, {'-314', '-214'}, {'-214', '-414'},
#                         {'-314', '-414'}, {'321', '121', '221', '421'}, {'-121', '-221'}, {'-321', '-121'},
#                         {'-121', '-421'}, {'-321', '-221'}, {'-421', '-221'}, {'-321', '-421'},
#                         {'322', '422', '122', '222'}, {'-122', '-222'}, {'-322', '-122'}, {'-122', '-422'},
#                         {'-322', '-222'}, {'-222', '-422'}, {'-322', '-422'}, {'123', '423', '323', '223'},
#                         {'-223', '-123'}, {'-323', '-123'}, {'-423', '-123'}, {'-223', '-323'}, {'-223', '-423'},
#                         {'-323', '-423'}, {'224', '424', '324', '124'}, {'-224', '-124'}, {'-324', '-124'},
#                         {'-124', '-424'}, {'-324', '-224'}, {'-224', '-424'}, {'-324', '-424'},
#                         {'431', '331', '131', '231'}, {'-131', '-231'}, {'-131', '-331'}, {'-131', '-431'},
#                         {'-331', '-231'}, {'-431', '-231'}, {'-431', '-331'}, {'232', '132', '432', '332'},
#                         {'-132', '-232'}, {'-132', '-332'}, {'-132', '-432'}, {'-232', '-332'}, {'-232', '-432'},
#                         {'-432', '-332'}, {'433', '233', '333', '133'}, {'-133', '-233'}, {'-333', '-133'},
#                         {'-433', '-133'}, {'-333', '-233'}, {'-433', '-233'}, {'-433', '-333'},
#                         {'334', '134', '434', '234'}, {'-234', '-134'}, {'-334', '-134'}, {'-434', '-134'},
#                         {'-234', '-334'}, {'-234', '-434'}, {'-434', '-334'}, {'241', '441', '341', '141'},
#                         {'-241', '-141'}, {'-341', '-141'}, {'-441', '-141'}, {'-241', '-341'}, {'-441', '-241'},
#                         {'-441', '-341'}, {'242', '142', '442', '342'}, {'-242', '-142'}, {'-342', '-142'},
#                         {'-442', '-142'}, {'-342', '-242'}, {'-442', '-242'}, {'-442', '-342'},
#                         {'243', '343', '443', '143'}, {'-143', '-243'}, {'-143', '-343'}, {'-143', '-443'},
#                         {'-343', '-243'}, {'-443', '-243'}, {'-343', '-443'}, {'144', '344', '244', '444'},
#                         {'-244', '-144'}, {'-344', '-144'}, {'-144', '-444'}, {'-344', '-244'}, {'-244', '-444'},
#                         {'-344', '-444'}]
#     sudoku_symbols_4 = {'241', '112', '341', '121', '332', '134', '124', '231', '114', '343', '334', '431', '244',
#                         '212', '232', '144', '333', '443', '324', '423', '414', '224', '442', '311', '344', '213',
#                         '113', '233', '411', '313', '422', '221', '434', '413', '141', '323', '331', '122', '242',
#                         '424', '132', '214', '111', '312', '433', '222', '314', '432', '211', '123', '142', '234',
#                         '321', '342', '131', '441', '143', '322', '243', '133', '421', '444', '412', '223'}
#     solver, if_solved = dpll({}, sudoku_clauses_4, sudoku_symbols_4)
'''


    # symbols, clauses = DIMACS_reader("testsets/4x4.txt")
    # solver, if_solved = dpll({}, clauses, symbols)
    # print(if_solved)
    # print(solver)
    # dimacs_content = tt_to_dimacs(solver)
    # save_dimacs(dimacs_content,'test')


    # 9*9
    to_DIMACS(9,"1000 sudokus.txt","sudoku-rules-9x9.txt")
    symbolss, clausess = DIMACS_reader("9by9.cnf")
    print(len(symbolss),len(clausess))
    solver, if_solved = dpll({}, clausess, symbolss)


    ## 4*4
    # to_DIMACS(4,"1000 sudokus.txt","sudoku-rules-9x9.txt")
    # symbols, clauses = DIMACS_reader("4by4.cnf")
    # print(len(symbols),len(clauses))
    # solver, if_solved = dpll({}, clauses, symbols)



    print(if_solved)
    print(solver)
    dimacs_content = tt_to_dimacs(solver)
    save_dimacs(dimacs_content,'testout')


