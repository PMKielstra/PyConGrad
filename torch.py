import torch

from backend import Backend
from cg_batch import cg_batch_generic

class _TorchBackend(Backend):
    def norm(X):
        return torch.norm(X, dim=-1)
    
    def dot(X, Y):
        return torch.bmm(X[:, :, None], Y[:, None, :]).squeeze(2)

    def all_true(X):
        return X.all()

    def max_vector_scalar(X, y):
        if not type(X) == type(y):
            y = torch.tensor(y).type(X)
        return torch.maximum(X, y)
    
cg_batch = cg_batch_generic(_TorchBackend)