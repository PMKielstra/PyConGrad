from warnings import warn

def backend_import_warn(package):
    warn(f"Cannot find module {package}; cg_batch will be None.")