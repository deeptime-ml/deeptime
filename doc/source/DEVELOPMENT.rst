=================
Developer's Guide
=================

.. toctree::
   :maxdepth: 2

Contributing
============

Basic Idea
----------
We use the "devel" branch to develop pyEMMA. When "devel" has reached
a mature state in terms of current functionality and stability, we
merge "devel" into the master branch. This happens at the time
when a release is made.

In order to develop certain features you should not work on "devel"
directly either, but rather branch it to a new personal feature branch
(here you can do whatever you want). Then, after testing your feature, you can offer the changes
to be merged on to the devel branch by doing a *pull request* (see below).
If accepted, that branch will be merged into devel, and unless overridden
by other changes your feature will make it eventually to master and the
next release.

Why
---
* Always have a tested and stable master branch.
* Avoid interfering with other developers until changes are merged.

How
---
One of the package maintainers merges the development branch(es) periodically.
All you need to do is to make your changes in the feature branch (see below
for details), and then offer a pull request. When doing so, a bunch of
automatic code tests will be run to test for direct or indirect bugs that
have been introduced by the change. This is done by a continuous integration (CI)
software like Jenkins http://jenkins-ci.org or Travis-CI http://travis-ci.org,
the first one is open source and the second one is free for open source projects
only. Again, you do not have to do anything here, as this happens automatically
after a pull request. You will see the output of these tests in the pull request
page on github.


Commit messages
---------------

Use commit messages in the style "[$package]: change" whenever the changes belong
to one package or module. You can suppress "pyemma" (that is trivial) and
"api" (which doesn't show up in the import).

E.g.: ::

      [msm.analysis]: implemented sparse pcca

That way other developers and the package managers immediately know which modules have
been changed and can watch out for possible cross-effects. Also this makes commits look uniform.

If you have a complex commit affecting several modules or packages, break it down
into little pieces with easily understandable commit messages. This allows us to
go back to intermediate stages of your work if something fails at the end.


Testing
-------
We use Pythons unittest module to write test cases for all algorithm.

To run all tests invoke: ::

    python setup.py test

or directly invoke nosetests in pyemma working copy: ::

    nosetests $PYEMMA_DIR

It is encouraged to run all tests (if you are changing core features),
you can also run individual tests by directly invoking them with the python interpreter.

Documentation
-------------
Every function, class, and module that you write must be documented. Please check out

    https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt

how to do this. In short, after every function (class, module) header, there should
be a docstring enclosed by """ ... """, containing a short and long description what
the function does, a clear description of the input parameters, and the return value,
and any relevant cross-references or citations. You can include Latex-style math
in the docstring.

For a deeper understanding and reference please have a look at the Sphinx documentation

    http://sphinx-doc.org/

In particular, the API functions (those publicly visible to the external user) should
be well documented.

To build the documentation you need the dependencies from the file 
requirements-build-docs.txt which you can install via pip::

   pip install -r requirements-build-docs.txt
    
afterwards you are ready to build the documentation::

   cd doc
   make html

The HTML docs can then be found in the doc/build/html directory.


Workflow
========
A developer creates a feature branch "feature" and commits his or her work to
this branch. When he or she is done with his work (have written at least a
working test case for it), he or she pushes this feature branch to his or her fork
and creates a pull request.  The pull request can then be reviewed and
merged upstream.

0. Get up to date - pull the latest changes from devel

::
   
      # first get the latest changes
      git pull 

1. Compile extension modules (also works with conda distributions)

::

      python setup.py develop

In contrast to *install*, which copies the development version into your
package directory, the develop flag results in simply putting a link from
your package directory into your development directory. That way local changes
to python files are immediately active when you import the package. You only
need to re-execute the above command, when a C extension was changed.

2. Create a new feature branch by copying from the devel branch and switch to it:

::
   
      # switch to development branch
      git checkout devel 
      # create new branch and switch to it
      git checkout -b feature 

3. Work on your feature branch. Here you can roam freely.

4. Write unit test and TEST IT (see above)! :-)

::

      touch fancy_feat_test.py
      # test the unit
      python fancy_feat_test.py
      # run the whole test-suite 
      # (to ensure that your newfeature has no side-effects)
      cd $PYEMMA_DIR
      python setup.py test
      

5. Commit your changes

::

      git commit fancy_feat.py fancy_feat_test.py \
          -m "Implementation and unit test for fancy feature"

repeat 3.-5. as often as necessary to accomplish your task. Remember to split your changes into small
commits.

6. Make changes available by pushing your commits to the server and creating a pull request

::

      # push your branch to your fork on github
      git push myfork feature
      
On github create a pull request from myfork/feature to origin/devel,
see https://help.github.com/articles/using-pull-requests


Conclusions
-----------

* Feature branches allow you to work without interfering with others.
* The devel branch contains all tested implemented features.
* The devel branch is used to test for cross-effects between features.
* Work with pull request to ensure your changes are being tested automatically
  and can be reviewed.
* The master branch contains all tested features and represents the
  set of features that are suitable for public usage.
  

Publish a new release
=====================

1. Merge current devel branch into master

::

   git checkout master; git merge devel

2. Make a new tag 'vmajor.minor.patch'. major means major release (major new functionalities),
   minor means minor changes and new functionalities, patch means no new functionality but just
   bugfixes or improvement to the docs.

::

   git tag -m "release description" v1.1

3. IMPORTANT: first push, then push --tags

::

   git push; git push --tags

4. Update conda recipes and perform binstar pushing (partially automatized)


