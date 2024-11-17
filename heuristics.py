from typing import List, Dict, Set
import math
import numpy as np

# get the sum over all clauses where l is in, 
# l here is all pos and neg symbols in list symbols
def get_sum_list(clauses: List[Set], symbols: List[str]):
    j_list_pos = []
    j_list_neg = []
    for symbol in symbols:
        j_sum_pos = 0
        j_sum_neg = 0
        for clause in clauses:
            if symbol in clause:
                j_sum_pos += math.pow(2, -len(clause))
            if '-'+symbol in clause:
                j_sum_neg += math.pow(2, -len(clause))
        j_list_pos = j_list_pos + [j_sum_pos]
        j_list_neg = j_list_neg + [j_sum_neg]
    return j_list_pos, j_list_neg


def jw_os(clauses: List[Set], symbols: List[str]):
    j_list_pos, j_list_neg = get_sum_list(clauses, symbols)
    pos_max_idx = np.argmax(np.array(j_list_pos))
    neg_max_idx = np.argmax(np.array(j_list_neg))
    if(j_list_pos[pos_max_idx] >= j_list_neg[neg_max_idx]):
        return symbols[pos_max_idx], True
    else:
        return symbols[neg_max_idx], False
    

def jw_ts(clauses: List[Set], symbols: List[str]):
    j_list_pos, j_list_neg = get_sum_list(clauses, symbols)
    j_list_sum = [sum(x) for x in zip(j_list_pos, j_list_neg)]
    max_idx = np.argmax(np.array(j_list_sum))

    if(j_list_pos[max_idx] >= j_list_neg[max_idx]):
        return symbols[max_idx], True
    else:
        return symbols[max_idx], False    
    
def mom(clauses: List[Set], symbols: List[str], k:int):
    pass

