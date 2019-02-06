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







trajs = ["xyz.xyz", "abc.xyz"]

t = TICA()

for traj in trajs:
    for data in read_this(traj).lag(10):
        t.partial_fit(data)

estimated_model = t.copy_current_model()
estimated_model.lagtime
