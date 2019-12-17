.. _ref-coordinates:

Coordinates package (pyemma.coordinates)
========================================
The *coordinates* package contains tools to select features from MD-trajectories.
It also assigns them to a discrete state space, which will be later used in Markov
modeling.

It supports reading from MD-trajectories, comma separated value ASCII files and 
NumPy arrays. The discretized trajectories are being stored as NumPy arrays of
integers.

.. automodule:: pyemma.coordinates

.. toctree::
   :maxdepth: 1


.. include impl detail here hidden, to get generate docs for these
   as sometimes api doc links to them.

Implementation
--------------
.. toctree::
   :maxdepth: 1

   coordinates.clustering
   coordinates.data
   coordinates.transform
   coordinates.pipelines
