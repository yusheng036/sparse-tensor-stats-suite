import pytest
import numpy as np
import sparse
from scipy import sparse as sp
from src.hll import (
    hll_estimator_C_ik,
    hll_estimator_C_il,
    hll_estimator_C_ijlm,
    hll_estimator_C_il_2,
    hll_estimator_C_ijm,
    hll_estimator,
)
from src.suitesparse_util import (
    rand_csr,
    rand_coo_3d,
    rand_coo_2d,
    ground_truth_2d,
    ground_truth_coo,
    assert_estimate,
)

@pytest.mark.parametrize("I,J,K", [(30, 40, 25), (10, 12, 8)])
@pytest.mark.parametrize("density", [0.01, 0.05])
@pytest.mark.parametrize("error", [0.1, 0.05])
def test_hll_C_ik(I, J, K, density, error):
    A = rand_csr(I, J, density, seed=1)
    B = rand_csr(J, K, density, seed=2)

    true_nnz = ground_truth_2d(A, B)
    est = hll_estimator_C_ik(A, B, error=error)

    assert_estimate(est, true_nnz, I * K, error)


@pytest.mark.parametrize("I,J,K,L", [(10, 8, 6, 10), (5, 4, 3, 5)])
@pytest.mark.parametrize("density", [0.05, 0.1])
@pytest.mark.parametrize("error", [0.1, 0.05])
def test_hll_C_il(I, J, K, L, density, error):
    A = rand_coo_3d((I, J, K), density, seed=3)
    B = rand_coo_3d((J, K, L), density, seed=4)

    true_nnz = ground_truth_coo(A, B, "ijk,jkl->il")
    est = hll_estimator_C_il(A, B, error=error)

    assert_estimate(est, true_nnz, I * L, error)


@pytest.mark.parametrize("I,J,K,L", [(10, 8, 6, 10), (5, 4, 3, 5)])
@pytest.mark.parametrize("density", [0.05, 0.1])
@pytest.mark.parametrize("error", [0.1, 0.05])
def test_hll_C_il_2(I, J, K, L, density, error):
    A = rand_coo_3d((I, J, K), density, seed=5)
    B = rand_coo_3d((J, K, L), density, seed=6)

    true_nnz = ground_truth_coo(A, B, "ijk,jkl->il")
    est = hll_estimator_C_il_2(A, B, error=error)

    assert_estimate(est, true_nnz, I * L, error)


@pytest.mark.parametrize("I,J,K,L,M", [(6, 5, 4, 6, 5), (4, 3, 3, 4, 3)])
@pytest.mark.parametrize("density", [0.05, 0.1])
@pytest.mark.parametrize("error", [0.1, 0.05])
def test_hll_C_ijlm(I, J, K, L, M, density, error):
    A = rand_coo_3d((I, J, K), density, seed=7)
    B = rand_coo_3d((K, L, M), density, seed=8)

    true_nnz = ground_truth_coo(A, B, "ijk,klm->ijlm")
    est = hll_estimator_C_ijlm(A, B, error=error)

    assert_estimate(est, true_nnz, I * J * L * M, error)


@pytest.mark.parametrize("I,J,K,M", [(10, 8, 6, 10), (5, 4, 3, 5)])
@pytest.mark.parametrize("density", [0.05, 0.1])
@pytest.mark.parametrize("error", [0.1, 0.05])
def test_hll_C_ijm(I, J, K, M, density, error):
    A = rand_coo_3d((I, J, K), density, seed=9)
    B = rand_coo_2d((K, M), density, seed=10)

    true_nnz = ground_truth_coo(A, B, "ijk,km->ijm")
    est = hll_estimator_C_ijm(A, B, error=error)

    assert_estimate(est, true_nnz, I * J * M, error)



@pytest.mark.parametrize("I,J,K,L", [(10, 8, 6, 10), (5, 4, 3, 5)])
@pytest.mark.parametrize("density", [0.05, 0.1])
@pytest.mark.parametrize("error", [0.1, 0.05])
def test_hll_general_C_il(I, J, K, L, density, error):
    A = rand_coo_3d((I, J, K), density, seed=11)
    B = rand_coo_3d((J, K, L), density, seed=12)

    true_nnz = ground_truth_coo(A, B, "ijk,jkl->il")
    est = hll_estimator(
        A, B,
        A_reduced=[1, 2], B_reduced=[0, 1],
        free_A=[0], free_B=[2],
        error=error
    )

    assert_estimate(est, true_nnz, I * L, error)


@pytest.mark.parametrize("I,J,K,L,M", [(6, 5, 4, 6, 5), (4, 3, 3, 4, 3)])
@pytest.mark.parametrize("density", [0.05, 0.1])
@pytest.mark.parametrize("error", [0.1, 0.05])
def test_hll_general_C_ijlm(I, J, K, L, M, density, error):
    A = rand_coo_3d((I, J, K), density, seed=13)
    B = rand_coo_3d((K, L, M), density, seed=14)

    true_nnz = ground_truth_coo(A, B, "ijk,klm->ijlm")
    est = hll_estimator(
        A, B,
        A_reduced=[2], B_reduced=[0],
        free_A=[0, 1], free_B=[1, 2],
        error=error
    )

    assert_estimate(est, true_nnz, I * J * L * M, error)


@pytest.mark.parametrize("I,J,K,M", [(10, 8, 6, 10), (5, 4, 3, 5)])
@pytest.mark.parametrize("density", [0.05, 0.1])
@pytest.mark.parametrize("error", [0.1, 0.05])
def test_hll_general_C_ijm(I, J, K, M, density, error):
    A = rand_coo_3d((I, J, K), density, seed=15)
    B = rand_coo_2d((K, M), density, seed=16)

    true_nnz = ground_truth_coo(A, B, "ijk,km->ijm")
    est = hll_estimator(
        A, B,
        A_reduced=[2], B_reduced=[0],
        free_A=[0, 1], free_B=[1],
        error=error
    )

    assert_estimate(est, true_nnz, I * J * M, error)