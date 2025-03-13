###############################################
### Helper function to set default value if
### user-specified value is not present
def set_default(name, default_value, pms):

    if (hasattr(pms, name)): return getattr(pms, name)
    else: return default_value
