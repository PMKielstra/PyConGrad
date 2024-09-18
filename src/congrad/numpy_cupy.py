from .backend import Backend
from .cg_batch import cg_batch_generic

def make_backend(np):
    """Make a backend class for either NumPy or CuPy."""
    class NumPyCuPyBackend(Backend):
        def norm(X):
            return np.linalg.norm(X, axis=-2, keepdims=True)
        
        def dot(X, Y):
            XX = np.expand_dims(np.swapaxes(X, -1, -2), -2)
            YY = np.expand_dims(np.swapaxes(Y, -1, -2), -1)
            dot_prod = np.matmul(XX, YY)
            return np.swapaxes(np.squeeze(dot_prod, axis=-1), -1, -2)
                
        def all_true(X):
            return X.all()
    
        def max_vector_scalar(X, y):
            return np.maximum(X, y)

        def presentable_norm(residual):
            return np.max(residual).item()
            
    return NumPyCuPyBackend
