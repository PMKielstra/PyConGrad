import cupy

from backend import Backend
from cg_batch import cg_batch_generic

class _CupyBackend(Backend):
    def norm(X):
        return cupy.linalg.norm(X, axis=-1)
    
    def dot(X, Y):
        return cupy.squeeze(cupy.matmul(X[:, :, None], y[:, None, :]), axis=-1)

    def all_true(X):
        return X.all()

    def max_vector_scalar(X, y):
        return cupy.maximum(X, y)
    
cg_batch = cg_batch_generic(_CupyBackend)