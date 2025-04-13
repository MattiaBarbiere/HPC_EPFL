import os
import random
import tqdm
import math

random.seed(42)

path = os.path.dirname(os.path.abspath(__file__))

# Access the original matrix file and read the number of non-zero entries
with open(f'{path}/lap2D_5pt_n2000.mtx', 'r') as f:
    lines_original = f.readlines()
    # The first line contains the number of non-zero entries
    N = int(lines_original[2].split()[0])
    NNZ_original = int(lines_original[2].split()[2])

# The maximum number of processors is 100
NUM_NNZ_PER_PROCESSOR = NNZ_original // 100
print(f"NUM_NNZ_PER_PROCESSOR = {NUM_NNZ_PER_PROCESSOR}")

# Compute the average number of non-zero entries per row
values_original = []
for line in tqdm.tqdm(lines_original):
    if line.startswith('%') or len(line.strip().split()) != 3:
        continue
    i, j, val = line.strip().split()
    values_original.append((int(i), int(j), float(val)))

nnz_per_row = [0] * (N)
for i, j, val in values_original:
    nnz_per_row[i - 1] += 1
    nnz_per_row[j - 1] += 1

# Compute mean and variance of the number of non-zero entries per row
from numpy import mean, std
mean_number_nnz_per_row = mean(nnz_per_row)
print(f"Mean: {mean_number_nnz_per_row}")
print(f"Std: {std(nnz_per_row)}")

# NOTE: The number of non-zero entries is almost uniformly distributed. Thus to reduce the matrix 
#      I will only take the top left corner of the matrix.

effective_nnz_per_processor = {}

def construct_matrix(p):
    """
    Constructs a matrix for p processors.
    Each processor will have approximately NUM_NNZ_PER_PROCESSOR non-zero entries.
    """
    # Calculate the number of non-zero entries for this processor
    nnz = NUM_NNZ_PER_PROCESSOR * p

    num_rows_new_matrix = 2 * math.ceil(nnz / mean_number_nnz_per_row)

    values = []
    for i, j, val in tqdm.tqdm(values_original):
        if i <= num_rows_new_matrix and j <= num_rows_new_matrix:
            values.append((i, j, val))

    # Get the largest index
    max_index = max(max(i, j) for i, j, _ in values)

    # Print work per processor
    effective_nnz_per_processor[p] = len(values)/p

    # Start the file
    filename = f'{path}/weak_matrix_{p}.mtx'

    with open(filename, 'w') as f:
        # Write the header
        f.write(f"%%MatrixMarket matrix coordinate real symmetric\n")
        f.write(f"% Generated 10-Apr-2025\n")
        f.write(f"{max_index} {max_index} {len(values)}\n")
        # Write the matrix entries
        for i, j, val in tqdm.tqdm(values):
            f.write(f"{i} {j} {val}\n")

for p in [1, 2, 4, 8, 10, 15, 20, 25, 30, 40, 50, 75, 100]:
    construct_matrix(p)
    print(f"Matrix for {p} processors constructed.")



# Some statistics about the number of non-zero entries per processor
print(f"Average number of non-zero entries per processor: {sum(effective_nnz_per_processor.values())/len(effective_nnz_per_processor)}")
print(f"Maximum number of non-zero entries per processor: {max(effective_nnz_per_processor.values())}")
print(f"Minimum number of non-zero entries per processor: {min(effective_nnz_per_processor.values())}")
print(f"Standard deviation of non-zero entries per processor: {std(list(effective_nnz_per_processor.values()))}")