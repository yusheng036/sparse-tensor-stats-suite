import hyperloglog
from scipy import sparse
from collections import defaultdict

def hll_estimator_C_ik(A: sparse.spmatrix, B: sparse.spmatrix, error=0.05) -> float:
    """
    nnz of C_ik = sum_{j} A_ij B_jk
    """
    m, n =  A.shape
    _, l = B.shape

    A = A.astype(bool, copy=False).tocsc()
    B = B.astype(bool, copy=False).tocsc()

    S_j = []
    for j in range(n):
        h = hyperloglog.HyperLogLog(error)
        rows = A.indices[A.indptr[j]: A.indptr[j + 1]]
        for i in rows:
            h.add(int(i))
        S_j.append(h)

    nnz = 0.0
    for k in range(l):
        h = hyperloglog.HyperLogLog(error)
        rows = B.indices[B.indptr[k]: B.indptr[k + 1]]
        for j in rows:
            h.update(S_j[j])
        nnz += len(h)

    return nnz


def hll_estimator_C_il(A: sparse.COO, B: sparse.COO, error=0.05) -> float:
    """
    nnz of C_il = sum_{j,k} A_ijk B_jkl
    """

    S = defaultdict(lambda: hyperloglog.HyperLogLog(error))
    for (i, j, k) in zip(A.coords[0], A.coords[1], A.coords[2]):
        key = (int(j), int(k))
        S[key].add(int(i))

    b_nnz = defaultdict(list)
    for (j, k, l) in zip(B.coords[0], B.coords[1], B.coords[2]):
        if (int(j), int(k)) in S:
            b_nnz[int(l)].append((int(j), int(k)))

    nnz = 0.0
    for _, n in b_nnz.items():
        h = hyperloglog.HyperLogLog(error)
        for (j, k) in n:
            h.update(S[(j, k)])
        nnz += len(h)
    return nnz


def hll_estimator_C_ijlm(A: sparse.COO, B: sparse.COO, error=0.05) -> float:
    """
    nnz of C_ijlm = sum_{k} A_ijk B_klm
    """

    S = defaultdict(lambda: hyperloglog.HyperLogLog(error))
    for (i, j, k) in zip(A.coords[0], A.coords[1], A.coords[2]):
        S[int(k)].add(str((int(i), int(j))))

    b_nnz = defaultdict(list)
    for (k, l, m) in zip(B.coords[0], B.coords[1], B.coords[2]):
        if int(k) in S:
            b_nnz[(int(l), int(m))].append(int(k))

    nnz = 0.0
    for _, k in b_nnz.items():
        h = hyperloglog.HyperLogLog(error)
        for k_i in k:
            h.update(S[k_i])
        nnz += len(h)
    return nnz



def hll_estimator_C_il_2(A: sparse.COO, B: sparse.COO, error=0.05) -> float:
    """
    nnz of C_il = sum_{k} A_ijk B_jkl
    """

    S = defaultdict(lambda: hyperloglog.HyperLogLog(error))
    for (i, j, k) in zip(A.coords[0], A.coords[1], A.coords[2]):
        S[int(j), int(k)].add(int(i))

    b_nnz = defaultdict(list)
    for (j, k, l) in zip(B.coords[0], B.coords[1], B.coords[2]):
        if (int(j), int(k)) in S:
            b_nnz[int(l)].append((int(j), int(k)))

    nnz = 0.0
    for _, n in b_nnz.items():
        h = hyperloglog.HyperLogLog(error)
        for (j, k) in n:
            h.update(S[(j, k)])
        nnz += len(h)
    return nnz


def hll_estimator_C_ijm(A: sparse.COO, B: sparse.COO, error=0.05) -> float:
    """
    nnz of C_ijm = sum_k A_ijk B_km
    """

    S = defaultdict(lambda: hyperloglog.HyperLogLog(error))
    for (k, m) in zip(B.coords[0], B.coords[1]):
        S[int(k)].add(int(m))

    b_nnz = defaultdict(list)
    for (i, j, k) in zip(A.coords[0], A.coords[1], A.coords[2]):
        if int(k) in S:
            b_nnz[(int(i), int(j))].append(int(k))

    nnz = 0.0
    for _, k in b_nnz.items():
        h = hyperloglog.HyperLogLog(error)
        for k_i in k:
            h.update(S[k_i])
        nnz += len(h)
    return nnz


def hll_estimator(
    A: sparse.COO,
    B: sparse.COO,
    A_reduced: list[int],
    B_reduced: list[int],
    free_A: list[int],
    free_B: list[int],
    error: float = 0.05
) -> float:

    S = defaultdict(lambda: hyperloglog.HyperLogLog(error))
    for i in zip(*[A.coords[d] for d in range(A.ndim)]):
        sketch = tuple(int(i[d]) for d in A_reduced)
        free = tuple(int(i[d]) for d in free_A)
        S[sketch].add(str(free) if len(free) > 1 else int(free[0]))

    b_nnz = defaultdict(list)
    for j in zip(*[B.coords[d] for d in range(B.ndim)]):
        sketch = tuple(int(j[d]) for d in B_reduced)
        free = tuple(int(j[d]) for d in free_B)
        if sketch in S:
            b_nnz[free].append(sketch)

    nnz = 0.0
    for _, sketch in b_nnz.items():
        h = hyperloglog.HyperLogLog(error)
        for k in sketch:
            h.update(S[k])
        nnz += len(h)
    return nnz