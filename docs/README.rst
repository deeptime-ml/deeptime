Building the documentation
==========================

In order to build the documentation you need to install the requirements
via the following command::

	pip install -r requirements-build-doc.txt


This will mainly drag in sphinx, matplotlib, jupyter notebooks and some small utilities.


After you have successfully installed these dependencies, you can invoke::

	make ipython-rst

to execute and convert the notebooks found in ../pyemma-ipython

After this step you are ready to convert the generated RST files to HTML via::

	make html


Note that for a quick build you can safely omit the ipython-rst step.


