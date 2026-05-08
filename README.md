# Sparse Tensor Stats Suite

A research toolkit for estimating the number of nonzero entries (nnz) in the output of sparse matrix/tensor multiplications without producing the intermediates. It implements and compares several estimation algorithms against ground truth on both synthetic and real-world matrices from the SuiteSparse Matrix Collection.

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
  hll.py
  cohen.py
  lrb.py
  mnc.py
  suitesparse_util.py

experiments/
  matrix.py
  plot_hll.py
  plot_lrb.py

tests/
  test_hll.py
  test_cohen.py
  test_lrb.py
  test_mnc.py
  test_suitesparse.py
```

## Dependencies

- `numpy`
- `scipy`
- `sparse`
- `hyperloglog`
- `matplotlib`
