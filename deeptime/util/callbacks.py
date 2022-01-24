import copy
from .platform import handle_progress_bar


class Callback:
    """Base callback function for the c++ bindings to indicate progress by incrementing a progress bar.

       Parameters
       ----------
       progress_bar : object
           Tested for a tqdm progress bar. Should implement update() and close() and have .total and .desc properties.
       n_iter : int
           Number of iterations to completion.
       display_text : string
           text to display in front of the progress bar.
       """

    def __init__(self, progress, n_iter=None, display_text=None):
        self.progress_bar = handle_progress_bar(progress)()
        if display_text is not None:
            self.progress_bar.desc = display_text
        if n_iter is not None:
            self.progress_bar.total = n_iter

    def __call__(self, *args, **kw):
        self.progress_bar.update()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.progress_bar.close()
