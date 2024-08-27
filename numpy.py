import numpy

from backend import Backend
from cg_batch import cg_batch_generic

class _NumpyBackend(Backend):
    def norm(X):
        return numpy.linalg.norm(X, axis=-1)
    
    def dot(X, Y):
        return numpy.squeeze(numpy.matmul(X[:, :, None], y[:, None, :]), axis=-1)

    def all_true(X):
        return X.all()

    def max_vector_scalar(X, y):
        return numpy.maximum(X, y)
    
cg_batch = cg_batch_generic(_NumpyBackend)