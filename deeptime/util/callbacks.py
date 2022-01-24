from .platform import handle_progress_bar


def supports_progress_interface(bar):
    r""" Method to check if a progress bar supports the deeptime interface, meaning that it
    has `update`, `close`, and `set_description` methods as well as a `total` attribute.

    Parameters
    ----------
    bar : object, optional
        The progress bar implementation to check, can be None.

    Returns
    -------
    supports : bool
        Whether the progress bar is supported.

    See Also
    --------
    ProgressCallback
    """
    has_methods = all(callable(getattr(bar, method, None)) for method in supports_progress_interface.required_methods)
    return has_methods


supports_progress_interface.required_methods = ['update', 'close', 'set_description']


class ProgressCallback:
    r"""Base callback function for the c++ bindings to indicate progress by incrementing a progress bar.

    Parameters
    ----------
    progress : object
       Tested for a tqdm progress bar. Should implement `update()`, `set_description()`, and `close()`. Should
       also possess a `total` constructor keyword argument.
    total : int
       Number of iterations to completion.
    description : string
       text to display in front of the progress bar.

    See Also
    --------
    supports_progress_interface
    """

    def __init__(self, progress, description=None, total=None):
        self.progress_bar = handle_progress_bar(progress)(total=total)
        assert supports_progress_interface(self.progress_bar), \
            f"Progress bar did not satisfy interface! It should at least have " \
            f"the method(s) {supports_progress_interface.required_methods}."
        if description is not None:
            self.progress_bar.set_description(description)

    def __call__(self, *args, **kw):
        self.progress_bar.update()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress_bar.close()
