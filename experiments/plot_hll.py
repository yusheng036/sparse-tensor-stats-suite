from matrix import (
    diagonal_matrix,
    tridiagonal_matrix,
    banded_matrix,
    lower_triangular_matrix,
    upper_triangular_matrix,
    horizontal_striped_matrix,
    vertical_striped_matrix,
    mod32_matrix,
    reorder_by_mod,
    reorder_horizontal_stripes,
    reorder_vertical_stripes,
)

from hll import hll_cohen_estimator
from cohen import cohen_estimator
from mnc import mnc


from scipy import sparse

def matrix_generation (matrix_types: str, n=256):
    if matrix_types == "diagonal":
        return diagonal_matrix(n)
    if matrix_types == "tridiagonal":
        return tridiagonal_matrix(n)
    if matrix_types == "banded":
        return banded_matrix(n, 5)
    if matrix_types == "lower_triangular":
        return lower_triangular_matrix(n)
    if matrix_types == "upper_triangular":
        return upper_triangular_matrix(n)
    if matrix_types == "horizontal_strip":
        return horizontal_striped_matrix(256)
    if matrix_types == "vertical_strip":
        return vertical_striped_matrix(256)
    if matrix_types == "mod32":
        return mod32_matrix(n)
    if matrix_types == "reorder_mod32":
        return reorder_by_mod(mod32_matrix(n))
    if matrix_types == "reorder_horizontal_strip":
        return reorder_horizontal_stripes(horizontal_striped_matrix(n))
    if matrix_types == "reorder_vertical_strip":
        return reorder_vertical_stripes(vertical_striped_matrix(n))

def ground_truth(A: sparse.spmatrix, B:sparse.spmatrix):
    return ((A.astype(bool) @ B.astype(bool)) != 0).nnz

def benchmark (matrix_types, n=256):

    rows = []

    for i in matrix_types:
        A = matrix_generation(i, n)
        B = matrix_generation(i, n)
        true_nnz = ground_truth(A, B)

        estimators = ["cohen", "hll", "mnc"]

        for j in estimators:
            if j == "cohen":
                m = cohen_estimator(A, B)
            elif j == "hll":
                m = hll_cohen_estimator(A, B)
            else:
                m = mnc(A, B)

            error = abs(m - true_nnz) / true_nnz
            ratio = m /true_nnz

            rows.append({
                "matrix_type": i,
                "estimator": j,
                "estimated_nnz": m,
                "true_nnz": true_nnz,
                "rel_error": error,
                "ratio": ratio,
            })
    return rows

def print_benchmark_table(rows):
    headers = ("Matrix Type", "Estimator", "Relative Error", "nnz/true_nnz")

    w1 = max(len(headers[0]), max(len(r["matrix_type"]) for r in rows))
    w2 = max(len(headers[1]), max(len(r["estimator"]) for r in rows))
    w3 = len(headers[2])
    w4 = len(headers[3])

    total_width = w1 + w2 + w3 + w4 + 9

    print("=" * total_width)
    print(
        f"{headers[0]:<{w1}} | "
        f"{headers[1]:<{w2}} | "
        f"{headers[2]:<{w3}} | "
        f"{headers[3]:<{w4}}"
    )
    print("-" * total_width)

    prev_matrix = None
    for r in rows:
        if prev_matrix is not None and r["matrix_type"] != prev_matrix:
            print("-" * total_width)

        print(
            f"{r['matrix_type']:<{w1}} | "
            f"{r['estimator']:<{w2}} | "
            f"{r['rel_error']:<{w3}.6f} | "
            f"{r['ratio']:<{w4}.6f}"
        )
        prev_matrix = r["matrix_type"]

    print("=" * total_width)

matrix_types = [
    "diagonal",
    "tridiagonal",
    "banded",
    "lower_triangular",
    "upper_triangular",
    "horizontal_strip",
    "vertical_strip",
    "mod32",
]

rows = benchmark(matrix_types, n=256)
print_benchmark_table(rows)