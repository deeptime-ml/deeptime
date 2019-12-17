
=======================================
Scikit-Time
=======================================

Scikit-Time is a Python library for the analysis for time series.
Currently it contains algorithms for the estimation, validation and analysis Markov models.

* Empiric covariance estimation for large (main memory exceeding) data-sets (online algorithm).
* Time-lagged independent component analysis (TICA).
* Variational approach for Markov processes (VAMP).
* Clustering methods
* Markov state model (MSM) estimation and validation and Bayesian estimation of MSMs.
* Computing Metastable states and structures with Perron-cluster cluster analysis (PCCA).
* Systematic coarse-graining of MSMs to transition models with few states.
* Hidden Markov Models (HMM) and Bayesian estimation for HMMs.
* Extensive analysis options for MSMs and HMMs, e.g. calculation of committors, mean first passage times,
  transition rates, experimental expectation values and time-correlation functions, etc.
* Transition Path Theory (TPT).

Technical features:

* Code is implemented in Python3 and C/C++.
* Runs on Linux (64 bit) and MacOS (64 bit).
* Modular and flexible object structure, consisting of data Transformers, model Estimators and Models.
* Basic compatibility with `scikit-learn <http://scikit-learn.org/>`_. More complete compatibility will follow.


Installation
============

.. toctree::
   :maxdepth: 2

   INSTALL
   Configuration

Documentation
=============

.. toctree::
   :maxdepth: 2

   api/index

Tutorials
=========

.. toctree::
   :maxdepth: 2

   tutorial

Development
===========
.. toctree::
   :maxdepth: 2

   CHANGELOG
   DEVELOPMENT

Legal Notices
=============
.. toctree::
   :maxdepth: 1

   legal

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
