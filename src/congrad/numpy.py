try:
    import numpy
    
    from .cg_batch import cg_batch_generic
    from .numpy_cupy import make_backend
    
    cg_batch = cg_batch_generic(make_backend(numpy))
    
except ImportError:
    cg_batch = None
    from .backend_import_warn import backend_import_warn
    backend_import_warn("numpy")