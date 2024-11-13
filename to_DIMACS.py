import os
import numpy as np


# 4 x 4
def fbf_parser(state_file,rule_file):
    #for states
    with open(state_file,"r") as s:
        a = s.readlines()
        # for line in a:
        state = [list(a[0][4*i:4*(i+1)]) for i in range(4)]
        board = np.array([["_" if j == "." else j  for j in row] for row in state ])
        state_ind = np.argwhere(board != "_")
        cnf_list = []
        for i in state_ind:
            #take each given number
            num = board[i[0],i[1]]
            # cnf form
            cnf = f"{i[0]+1}{i[1]+1}{num} 0"
            cnf_list.append(cnf)
        print(cnf_list)

    #for rules
    with open(rule_file, "r") as r:
        lines = r.readlines()
        exist_rule = []
        repetition_rule = []
        for i in lines:
            a = i.split(" ")

            if "p" in a:
                pass
            elif len(a) == 5:
                exist_rule.append(i)
            else:
                repetition_rule.append(i)

    with open("test_4by4", "w") as t:
        t.write("p cnf 444 448\n")
        for i in cnf_list:
            t.write(f"{i} \n")
        for i in exist_rule:
            t.write(f"{i}")
        for i in repetition_rule:
            t.write(f"{i}")

fbf_parser("4x4.txt","sudoku-rules-4x4.txt")