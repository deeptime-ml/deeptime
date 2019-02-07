import abc

class Estimator(object):

    def __init__(self, model=None):
        self._model = model if model is not None else self._create_model()

    @abc.abstractmethod
    def fit(self, data):
        pass

    @property
    def model(self):
        return self._model

    @abc.abstractmethod
    def _create_model(self):
        pass

    def copy_current_model(self):
        import copy
        return copy.deepcopy(self.model)
