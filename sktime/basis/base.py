class Observable(object):

    def _evaluate(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return self._evaluate(x)
