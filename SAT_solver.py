import sys
from pathlib import Path
from DIMACS_parser import tt_to_dimacs, save_dimacs
from dpll import Rand, JWOneSide, JWTwoSide, MOM, SodokuSolver, DLCS, DLIS, FirstPos, Max


def entrypoint():
    '''
    -S1: rand
    -S2: jw_os
    -S3: jw_ts
    -S4: mom
    '''
    sn_map = {'-S1': Rand(), '-S2': JWOneSide(), '-S3': JWTwoSide(), '-S4': MOM(), '-S5': DLCS(), '-S6': DLIS(), '-S7': FirstPos(), '-S8': Max()}

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







