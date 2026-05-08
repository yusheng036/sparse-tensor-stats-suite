# Sparse Tensor Stats Suite

A research toolkit for estimating the number of nonzero entries (nnz) in the output of sparse matrix/tensor multiplications — without computing the product directly. It implements and compares several estimation algorithms against ground truth on both synthetic and real-world matrices from the SuiteSparse Matrix Collection.

## Algorithms

| Module | Algorithm | Description |
|---|---|---|
| `src/hll.py` | HyperLogLog (HLL) | Set-sketch estimator for 2D and 3D sparse tensor contractions |
| `src/cohen.py` | Cohen's min-hash estimator | Exponential random variable sketching for 2D matmul nnz |
| `src/lrb.py` | Localized Region Bound (LRB) | Upper bound on output nnz by partitioning the shared dimension into regions |
| `src/mnc.py` | MNC estimator | Structural histogram estimator exploiting row/column degree distributions |

## Project Structure

```
src/
  hll.py              # HyperLogLog-based nnz estimators (2D and 3D contractions)
  cohen.py            # Cohen min-hash estimator
  lrb.py              # Localized Region Bound estimator
  mnc.py              # MNC structural estimator
  suitesparse_util.py # Downloading/loading SuiteSparse matrices, ground truth helpers

experiments/
  matrix.py           # Synthetic sparse matrix generators and sparsity pattern plots
  plot_hll.py         # HLL accuracy experiments
  plot_lrb.py         # LRB accuracy experiments

tests/
  test_hll.py
  test_cohen.py
  test_lrb.py
  test_mnc.py
  test_suitesparse.py
```

## Installation

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install hyperloglog sparse
```

## Running Tests

```bash
pytest
```

## Usage

### HyperLogLog estimator (2D matmul)

```python
from scipy import sparse
from src.hll import hll_estimator_C_ik

A = sparse.random(1000, 500, density=0.01, format="csr")
B = sparse.random(500, 800, density=0.01, format="csr")

estimated_nnz = hll_estimator_C_ik(A, B, error=0.05)
```

### Cohen estimator

```python
from src.cohen import cohen_estimator

estimated_nnz = cohen_estimator(A, B, r=64)
```

### Localized Region Bound

```python
from src.lrb import lrb_matmul_stats

upper_bound = lrb_matmul_stats(A, B, regions=32)
```

### MNC estimator

```python
from src.mnc import mnc

estimated_nnz = mnc(A, B)
```

### Loading SuiteSparse matrices

```python
from src.suitesparse_util import download_cached, suitesparse_tar_url, load_first_mtx_from_tar

url = suitesparse_tar_url("SNAP", "web-Google")
data = download_cached(url)           # cached to .cache/suitesparse/
A = load_first_mtx_from_tar(data)
```

## Dependencies

- `numpy`
- `scipy`
- `sparse` (PyData Sparse, for COO tensor support)
- `hyperloglog` (for HLL estimators)
- `matplotlib` (for experiment plots)
