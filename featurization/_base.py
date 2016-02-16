'''
Created on 15.02.2016

@author: marscher
'''
from pyemma.util.annotators import deprecated


class Feature(object):

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    @deprecated
    def map(self, traj):
        r"""Deprecated: use transform(traj)

        """
        return self.transform(traj)
