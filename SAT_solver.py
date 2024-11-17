import os
from typing import Dict, List, Set, Tuple
from DIMACS_parser import tt_to_dimacs, save_dimacs, DIMACS_reader,to_DIMACS,to_DIMACS_Sixteen,DIMACS_reader_Sixteen
from DPLL import dpll

if __name__ == '__main__':

    # symbols, clauses = DIMACS_reader("testsets/4x4.txt")
    # solver, if_solved = dpll({}, clauses, symbols)
    # print(if_solved)
    # print(solver)
    # dimacs_content = tt_to_dimacs(solver)
    # save_dimacs(dimacs_content,'test')


    # 9*9
    # to_DIMACS(9,"testsets/1000 sudokus.txt","sudoku-rules-9x9.txt")

    # cnf_files = os.listdir("9by9_cnf")
    # for i in cnf_files:
    #     symbols, clauses = DIMACS_reader(f"9by9_cnf/{i}")
    #     print(len(symbols), len(clauses))
    #     solver, if_solved = dpll({}, clauses, symbols, 'mom')
    #     # print(if_solved)
    #     # print(solver)
    #     dimacs_content = tt_to_dimacs(solver)
    #     save_dimacs(dimacs_content, f'{i}_solution')


    ## 4*4
    # to_DIMACS(4,"4x4.txt","sudoku-rules-4x4.txt")

    # cnf_files = os.listdir("4by4_cnf")
    # for i in cnf_files:
    #     symbols, clauses = DIMACS_reader(f"4by4_cnf/{i}")
    #     # print(len(symbols),len(clauses))
    #     solver, if_solved = dpll({}, clauses, symbols, 'jw_ts')
    #     # print(if_solved)
    #     # print(solver)
    #     dimacs_content = tt_to_dimacs(solver)
    #     save_dimacs(dimacs_content, f'{i}_solution')


    16*16
    # to_DIMACS_Sixteen("16x16.txt","sudoku-rules-16x16.txt")
    symbols, clauses = DIMACS_reader_Sixteen(f"16by16_cnf/16by16_1.cnf")
    print(len(symbols),len(clauses))






