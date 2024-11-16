from typing import Dict, List, Set, Tuple

def DIMACS_reader(file):
    #within 10*10 puzzle
    with open(file,"r") as f:
        lines = f.readlines()
        para = lines[0].split(" ")[2]
        print(lines[0].split(" ")[2])
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