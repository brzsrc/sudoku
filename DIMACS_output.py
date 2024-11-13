from typing import Dict, List, Set, Tuple
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