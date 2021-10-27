
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


def handle_progress_bar(progress):
    r"""Takes a (potential) progress bar, if None, just returns an iterable that does nothing but return everything.

    Parameters
    ----------
    progress : progress bar or None, optional
        The progressbar

    Returns
    -------
    progress_bar : iterable
        A progress bar (or no/identity progress bar if input was None).
    """
    if progress is None:
        class progress:
            def __init__(self, x=None, **_):
                self._x = x

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                return False

            def __iter__(self):
                for x in self._x:
                    yield x

            def update(self): pass

    return progress
