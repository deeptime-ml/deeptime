import copy

class Callback:
    """Base callback function for the c++ bindings to indicate progress by incrementing a progress bar."""
    def __init__(self, progress_bar, n_iter, display_text):
        """Initialize the progress bar.

        Parameters
        ----------
        progress_bar : object
            Tested for a tqdm progress bar. Should implement update() and close() and have .total and .desc properties.
        n_iter : int
            Number of iterations to completion.
        display_text : string
            text to display in front of the progress bar.
        """
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


class TRAMCallback(Callback):
    """Callback for the TRAM estimate process. Increments a progress bar and optionally saves iteration increments and
    log likelihoods to a list."""
    def __init__(self, progress_bar, n_iter, log_likelihoods_list=None, increments=None,
                 save_convergence_info=False):
        super().__init__(progress_bar, n_iter, "Running TRAM estimate")
        self.log_likelihoods = log_likelihoods_list
        self.increments = increments
        self.save_convergence_info = save_convergence_info
        self.last_increment = 0

    def __call__(self, increment, log_likelihood):
        super().__call__()

        if self.save_convergence_info:
            if self.log_likelihoods is not None:
                self.log_likelihoods.append(log_likelihood)
            if self.increments is not None:
                self.increments.append(increment)

        self.last_increment = increment
