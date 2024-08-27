import torch

from backend import Backend
from cg_batch import cg_batch_generic

class _TorchBackend(Backend):
    def norm(X):
        return torch.linalg.vector_norm(X, dim=-1)
    
    def dot(X, Y):
        return torch.einsum("bi,bi->b", X, Y).unsqueeze(1)

    def all_true(X):
        return X.all()

    def max_vector_scalar(X, y):
        if not type(X) == type(y):
            y = torch.tensor(y).type(type(X))
        return torch.maximum(X, y)
    
cg_batch = cg_batch_generic(_TorchBackend)