# PyConGrad
A somewhat optimized generic batch conjugate gradient algorithm that works with PyTorch, NumPy, CuPy, and whatever other backend you like as long as you code it up.

Heavily inspired by [sbarratt/torch_cg](https://github.com/sbarratt/torch_cg) and uses similar function signatures.

## Installation and Usage

```bash
$ pip install congrad # If you already have either NumPy, PyTorch, CuPy, or your backend of choice installed
$ pip install congrad[numpy] # To enforce NumPy dependency
$ pip install congrad[torch] # To enforce PyTorch dependency
$ pip install congrad[cupy] # To enforce CuPy dependency
```

```python
from congrad.numpy import cg_batch # Or congrad.torch or congrad.cupy

X = np.random.rand(100, 100)
A = X @ X.T + 0.001 * np.eye(100)
b = np.random.rand(100, 10) # PyConGrad expects matrix equations, so the rightmost dimension is for batching.

def A_batch(x):
    return np.matmul(A, x)

solution, solve_info = cg_batch(A_batch, b)
```

For more information, documentation, and detailed instructions, see the `examples` folder.  In particular, notebook 2 shows you how to use a different batch dimension if you want to batch over vectors instead of matrices.
