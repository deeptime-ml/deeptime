import abc


class Model(object):

    def copy(self):
        import copy
        return copy.deepcopy(self)


class Estimator(object):

    def __init__(self, model=None):
        self._model = model if model is not None else self._create_model()

    @abc.abstractmethod
    def fit(self, data):
        pass

    def fetch_model(self) -> Model:
        return self._model

    @abc.abstractmethod
    def _create_model(self):
        pass


class Transformer(object):

    @abc.abstractmethod
    def transform(self, data):
        pass
