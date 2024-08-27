from .backend import Backend
from .cg_batch import cg_batch_generic

def make_backend(np):
    """Make a backend class for either NumPy or CuPy."""
    class NumPyCuPyBackend(Backend):
        def norm(X):
            return np.linalg.norm(X, axis=-1)
        
        def dot(X, Y):
            return np.squeeze(np.matmul(X[:, None, :], Y[:, :, None]), axis=-1)
    
        def all_true(X):
            return X.all()
    
        def max_vector_scalar(X, y):
            return np.maximum(X, y)

        def presentable_norm(residual):
            return np.max(residual).item()
            
    return NumPyCuPyBackend
