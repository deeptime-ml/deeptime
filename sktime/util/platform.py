
def module_available(modname: str) -> bool:
    r"""Checks whether a module is installed and available for import by the current interpreter.

    Parameters
    ----------
    modname : str
        Name of the module

    Returns
    -------
    available: bool
        Whether the module is available.
    """
    try:
        __import__(modname)
        return True
    except ImportError:
        return False
