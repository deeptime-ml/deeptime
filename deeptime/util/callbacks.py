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
    has_attributes = all(hasattr(bar, attribute) for attribute in supports_progress_interface.required_attributes)
    return has_methods and has_attributes


supports_progress_interface.required_methods = ['update', 'close', 'set_description']
supports_progress_interface.required_attributes = ['n']


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
        self.total = total
        self.set_description(description)

        assert supports_progress_interface(self.progress_bar), \
            f"Progress bar did not satisfy interface! It should at least have " \
            f"the method(s) {supports_progress_interface.required_methods} and " \
            f"the attribute(s) {supports_progress_interface.required_attributes}."

    def __call__(self, inc=1, *args, **kw):
        self.progress_bar.update(inc)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self.progress_bar.total = self.progress_bar.n  # force finish
        self.progress_bar.close()

    def set_description(self, value):
        self.progress_bar.set_description(value)


class IterationErrorProgressCallback(ProgressCallback):
    r"""Callback function for the c++ bindings to indicate progress by incrementing a progress bar and showing the
    iteration error on each iteration.

    Parameters
    ----------
    progress : object
       Tested for a tqdm progress bar. Should implement `update()`, `set_description()`, and `close()`. Should
       also possess a `total` constructor keyword argument.
    total : int
       Number of iterations to completion.
    description : string
       text to display in front of the progress bar.

    Notes
    -----
    To display the iteration error, the error needs to be passed to `__call__()` as keyword argument `error`.

    See Also
    --------
    supports_progress_interface, ProgressCallback
    """

    def __init__(self, progress, description=None, total=None):
        super().__init__(progress, description, total)
        self.description = description

    def __call__(self, inc=1, *args, **kw):
        super().__call__(inc)
        if 'error' in kw:
            super().set_description("{} - [inc: {:.1e}]".format(self.description, kw.get('error')))
