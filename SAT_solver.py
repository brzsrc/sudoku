import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple
from DIMACS_parser import tt_to_dimacs, save_dimacs, DIMACS_reader,to_DIMACS,to_DIMACS_Sixteen,DIMACS_reader_Sixteen
from DPLL import dpll
from js import Heuristic, Rand, JWOneSide, JWTwoSide, MOM, SodokuSolver


def entrypoint():
    '''
    -S1: rand
    -S2: jw_os
    -S3: jw_ts
    -S4: mom
    '''
    sn_map = {'-S1': Rand(), '-S2': JWOneSide(), '-S3': JWTwoSide(), '-S4': MOM()}

    print(sys.argv)
    assert len(sys.argv) == 3
    _, sn, filename = sys.argv
    path = Path(filename).with_suffix('.cnf')

    assert sn in sn_map
    assert path.exists()
    return path, sn_map.get(sn)



if __name__ == '__main__':
    path, heuristic = entrypoint()
    filename = str(path)
    solver = SodokuSolver(filename, heuristic)
    solvable = solver.solve()
    dimacs_content = tt_to_dimacs(solver.solution, solvable)
    save_dimacs(dimacs_content, filename)







