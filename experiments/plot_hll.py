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

from src import (
    cohen_estimator,
    hll_cohen_estimator,
    MNC,
)

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
        return reorder_by_mod(n)
    if matrix_types == "reorder_horizontal_strip":
        return reorder_horizontal_stripes(n)
    if matrix_types == "reorder_vertical_strip":
        return reorder_vertical_stripes(n)

def ground_truth(A: sparse.spmatrix, B:sparse.spmatrix):
    return ((A.astype(bool) @ B.astype(bool)) != 0).nnz

def benchmark (matrix_types, n=256):

    rows = []
    for i in matrix_types:
        A = matrix_generation(i, n)
        B = matrix_generation(i, n)
        true_nnz = ground_truth(A, B)
    ...
