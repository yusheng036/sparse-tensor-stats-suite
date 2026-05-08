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
  hll.py              # HyperLogLog-based nnz estimators
  cohen.py            # Cohen min-hash estimator
  lrb.py              # Localized Region Bound estimator
  mnc.py              # MNC structural estimator
  suitesparse_util.py # Downloading/loading SuiteSparse matrices

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

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pip install hyperloglog sparse
```

## Dependencies

- `numpy`
- `scipy`
- `sparse`
- `hyperloglog`
- `matplotlib`
