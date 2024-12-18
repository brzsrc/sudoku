from typing import Dict, List, Set, Tuple
import numpy as np
import os
from pathlib import Path

def to_DIMACS_Sixteen(state_file,rule_file):
    # the rows num of out_file of 4*4 is 455 = 1 + 6 + 7*(4*4*4)
    # the rows num of out_file of 9*9 is 12010 = 1 + 21 + 37*(9*9*4)

    #for states
    cnf_list = []
    with open(state_file,"r") as s:
        a = s.readlines()
        for line in a:
            state = [list(line[16*i:16*(i+1)]) for i in range(16)]
            board = np.array([["_" if j == "." else j for j in row] for row in state])
            state_ind = np.argwhere(board != "_")
            sate_list = []
            value_dic = {"A":"10","B":"11","C":"12","D":"13","E":"14","F":"15","G":"16"}
            for i in state_ind:
                #take each given number
                num = board[i[0],i[1]]
                try:
                    num = int(num)
                    cnf = f"{17 * 17 * (i[0] + 1) + 17 * (i[1] + 1) + num} 0"
                except:
                    num = int(value_dic[num])
                    cnf = f"{17 * 17 * (i[0] + 1) + 17 * (i[1] + 1) + num} 0"
                # cnf form

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
            elif len(a) == 16 + 1:
                exist_rule.append(i)
            else:
                repetition_rule.append(i)
    try:
        os.makedirs(f"{16}by{16}_cnf")
    except:
        pass
    for i in range(len(cnf_list)):
        with open(f"{16}by{16}_cnf2/{16}by{16}_{i+1}.cnf","w") as t:
            t.write(f"p cnf {16}{16}{16} {len(cnf_list) + len(exist_rule) + len(repetition_rule)}\n")
            for i in cnf_list[i]:
                t.write(f"{i} \n")
            for i in exist_rule:
                t.write(f"{i}")
            for i in repetition_rule:
                t.write(f"{i}")


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
    os.makedirs(f"{size}by{size}_cnf")
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

        symbols = set()
        clauses = []
        begin_line = 0

        for i, line in enumerate(lines):
            fst_char = line.split(" ")[0]
            if fst_char == 'p':
                begin_line = i+1

        for line in lines[begin_line:]:
            clause = line.split(" ")
            if '0\n' in clause:
                end_point = clause.index("0\n")
            else:
                end_point = clause.index("0")
            clause = clause[:end_point]
            clauses.append(set(clause))
        return symbols, clauses


def tt_to_dimacs(truth_table: Dict[str, bool], if_solved:bool):
    sorted(truth_table)
    if not truth_table:
        return ""
    if not if_solved:
        return ""
    # Find the highest variable number to determine the number of variables
    num_var = max(int(var) for var in truth_table.keys())
    num_clauses = len(truth_table)
    clauses = []

    for key, value in truth_table.items():
        if value:
            clauses.append(f"{key} 0")

    # Create the DIMACS CNF header
    header = f"p cnf {num_var} {num_clauses}"

    # Combine header and clauses
    dimacs_content = [header] + clauses

    return "\n".join(dimacs_content)


def save_dimacs(content: str, filename: str):
    """
    Save a string content into a DIMACS file.
    """

    filename = filename.removesuffix(".cnf")
    filepath = f"./outputs/{filename}.output"
    filepath_parent = Path(filepath).resolve().parent
    if not os.path.exists(f"{filepath_parent}"):
        os.makedirs(f"{filepath_parent}")
    with open(filepath, 'w') as file:
        file.write(content)

    print(f"DIMACS CNF file '{filename}.output' generated successfully.")




# to_DIMACS(9,"1000 sudokus.txt","sudoku-rules-9x9.txt")
# symbolss, clausess = DIMACS_reader("9by9.cnf")
# symbols, clauses = DIMACS_reader("4by4.cnf")
# print(len(symbols),len(clauses))
# print(len(symbolss),len(clausess))
#solver, if_solved = dpll({}, clausess, symbolss)
#
# print(if_solved)
# print(solver)
#dimacs_content = tt_to_dimacs(solver)
# save_dimacs(dimacs_content,'test_out')
