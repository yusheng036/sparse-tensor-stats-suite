from __future__ import annotations

import io
import tarfile
import urllib.request
from pathlib import Path
from typing import Union

import numpy as np
from scipy import sparse as sp
import sparse
from scipy.io import mmread
from urllib.parse import quote

def suitesparse_tar_url(group: str, name: str) -> str:
    return f"https://sparse.tamu.edu/MM/{quote(group, safe="")}/{quote(name, safe="")}.tar.gz"

def download_cached(url: str, cache_dir: str = ".cache/suitesparse") -> bytes:
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / url.split("/")[-1]
    if not path.exists():
        with urllib.request.urlopen(url) as r:
            path.write_bytes(r.read())
    return path.read_bytes()

def load_first_mtx_from_tar(tar_gz: Union[bytes, Path, str]) -> sparse.csr_matrix:
    if isinstance(tar_gz, (Path, str)):
        tf = tarfile.open(tar_gz, mode="r:gz")
    elif isinstance(tar_gz, (bytes, bytearray, memoryview)):
        tf = tarfile.open(fileobj=io.BytesIO(tar_gz), mode="r:gz")
    else:
        raise TypeError(f"Unsupported tar_gz type: {type(tar_gz)}")

    with tf:
        mtx_members = [m for m in tf.getmembers() if m.name.endswith(".mtx")]
        if not mtx_members:
            raise RuntimeError("No .mtx file found in tarball")

        f = tf.extractfile(mtx_members[0])
        assert f is not None

        A = mmread(f)
        A = sparse.csr_matrix(A) if not sparse.issparse(A) else A.tocsr()

    if A.nnz:
        A.data = np.ones_like(A.data, dtype=np.int8)
    return A

def load_suitesparse_matrix(group, name):
    npz_path = Path(f".cache/npz/{group}_{name}.npz")
    return sp.load_npz(npz_path).tocsr()

def ground_truth(A: sparse.spmatrix, B: sparse.spmatrix) -> int:
    A2, B2 = A.copy(), B.copy()
    if A2.nnz:
        A2.data[:] = 1
    if B2.nnz:
        B2.data[:] = 1
    C = (A2 @ B2).tocsr()
    if C.nnz:
        C.data[:] = 1
        C.eliminate_zeros()
    return C.nnz

def ground_truth_2d(A, B):
    C = (A.astype(bool) @ B.astype(bool)).astype(bool)
    return int(C.nnz)

def rand_csr(m, n, density, seed) -> sp.csr_matrix:
    rng = np.random.default_rng(seed)
    return sp.random(
        m, n, density=density, format="csr",
        random_state=rng,
        data_rvs=lambda k: np.ones(k, dtype=np.int8),
    )

def rand_coo_2d(shape, density, seed=0):
    rng = np.random.default_rng(seed)
    total = int(np.prod(shape) * density)
    coords = [rng.integers(0, s, total) for s in shape]
    data = np.ones(total)
    return sparse.COO(coords, data, shape=shape)

def rand_coo_3d(shape, density, seed=0):
    rng = np.random.default_rng(seed)
    total = int(np.prod(shape) * density)
    coords = [rng.integers(0, s, total) for s in shape]
    data = np.ones(total)
    return sparse.COO(coords, data, shape=shape)

def ground_truth_coo(A, B, contraction):
    A_dense = A.todense()
    B_dense = B.todense()
    C_dense = np.einsum(contraction, A_dense, B_dense).astype(bool)
    return int(np.count_nonzero(C_dense))

def assert_estimate(est, true_nnz, max_nnz, error):
    assert est >= 0.0
    assert est <= max_nnz * 1.5 + 1e-9
    if true_nnz == 0:
        assert est <= 1e-9
    else:
        rel_err = abs(est - true_nnz) / true_nnz
        assert rel_err <= max(0.35, error * 10)

