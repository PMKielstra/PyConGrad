try:
    import torch
    
    from .backend import Backend
    from .cg_batch import cg_batch_generic
    
    class TorchBackend(Backend):
        def norm(X):
            return torch.linalg.vector_norm(X, dim=-2, keepdim=True)
        
        def dot(X, Y):
            return torch.matmul(X.transpose(-1, -2).unsqueeze(-2),
                                Y.transpose(-1, -2).unsqueeze(-1)) \
            .squeeze(-1).transpose(-1, -2)
    
        def all_true(X):
            return X.all()
    
        def max_vector_scalar(X, y):
            if not type(X) == type(y):
                y = torch.tensor(y).type(type(X))
            return torch.maximum(X, y)

        def presentable_norm(residual):
            return torch.max(residual).item()
        
    cg_batch = cg_batch_generic(TorchBackend)
    
except ImportError:
    cg_batch = None
    from .backend_import_warn import backend_import_warn
    backend_import_warn("torch")