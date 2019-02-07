import abc

class Estimator(object):

    @abc.abstractmethod
    def fit(self, data):
        pass

    @property
    @abc.abstractmethod
    def model(self):
        pass

    def copy_current_model(self):
        """
        Copy current model

        Returns
        -------
        Copy of current model
        """
        import copy
        return copy.deepcopy(self.model)
