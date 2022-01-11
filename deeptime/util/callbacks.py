import copy

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
    def __init__(self, progress_bar, n_iter, display_text):

        self.progress_bar = None
        if progress_bar is not None:
            self.progress_bar = copy.copy(progress_bar)
            self.progress_bar.desc = display_text
            self.progress_bar.total = n_iter

    def __call__(self):
        """Callback method for the c++ bindings."""
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress_bar is not None:
            self.progress_bar.close()
