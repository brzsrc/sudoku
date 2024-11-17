from typing import Dict, List, Set, Tuple
import numpy as np
import os
# from DPLL import dpll

def to_DIMACS(size,state_file,rule_file):
    # the rows num of out_file of 4*4 is 455 = 1 + 6 + 7*(4*4*4)
    # the rows num of out_file of 9*9 is 12010 = 1 + 21 + 37*(9*9*4)

    #for states
    cnf_list = []
    with open(state_file,"r") as s:
        a = s.readlines()
        for line in a:
            state = [list(line[size*i:size*(i+1)]) for i in range(size)]
            board = np.array([["_" if j == "." else j  for j in row] for row in state])
            state_ind = np.argwhere(board != "_")
            sate_list = []
            for i in state_ind:
                #take each given number
                num = board[i[0],i[1]]
                # cnf form
                cnf = f"{i[0]+1}{i[1]+1}{num} 0"
                sate_list.append(cnf)
            cnf_list.append(sate_list)

    #for rules
    with open(rule_file, "r") as r:
        lines = r.readlines()
        exist_rule = []
        repetition_rule = []
        for i in lines:
            a = i.split(" ")

            if "p" in a:
                pass
            elif len(a) == size + 1:
                exist_rule.append(i)
            else:
                repetition_rule.append(i)

    for i in range(len(cnf_list)):
        with open(f"{size}by{size}_cnf/{size}by{size}_{i+1}.cnf","w") as t:
            t.write(f"p cnf {size}{size}{size} {len(cnf_list) + len(exist_rule) + len(repetition_rule)}\n")
            for i in cnf_list[i]:
                t.write(f"{i} \n")
            for i in exist_rule:
                t.write(f"{i}")
            for i in repetition_rule:
                t.write(f"{i}")

def DIMACS_reader(file):
    #within 10*10 puzzle
    with open(file,"r") as f:
        lines = f.readlines()
        para = lines[0].split(" ")[2]
        symbols = set()
        for i in range(100,1000):
            if "0" not in str(i) and str(i)[0] <= para[0] and str(i)[1] <= para[1] and str(i)[2] <= para[2] :
                symbols.add(str(i))

        clauses = []
        for line in lines[1:]:
            clause = line[:-1].split(" ")
            end_point = clause.index("0")
            clause = clause[:end_point]
            clauses.append(set(clause))

        return symbols, clauses


def tt_to_dimacs(truth_table: Dict[str, bool]):
    sorted(truth_table)
    # Find the highest variable number to determine the number of variables
    num_var = max(int(var) for var in truth_table.keys())
    num_clauses = len(truth_table)
    clauses = []

    for key, value in truth_table.items():
        if value:
            clauses.append(f"{key} 0")
        else:
            clauses.append(f"-{key} 0")

    # Create the DIMACS CNF header
    header = f"p cnf {num_var} {num_clauses}"

    # Combine header and clauses
    dimacs_content = [header] + clauses

    return "\n".join(dimacs_content)


def save_dimacs(content: str, filename: str):
    """
    Save a string content into a DIMACS file.
    """
    filepath = f"./outputs/{filename}.output"
    with open(filepath, 'w') as file:
        file.write(content)

    print(f"DIMACS CNF file '{filename}.output' generated successfully.")




# to_DIMACS(9,"1000 sudokus.txt","sudoku-rules-9x9.txt")
# symbolss, clausess = DIMACS_reader("9by9.cnf")
# symbols, clauses = DIMACS_reader("4by4.cnf")
# print(len(symbols),len(clauses))
# print(len(symbolss),len(clausess))
# solver, if_solved = dpll({}, clausess, symbolss)
#
# print(if_solved)
# print(solver)
# dimacs_content = tt_to_dimacs(solver)
# save_dimacs(dimacs_content,'test_out')