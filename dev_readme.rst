-------------------------
How to make a new release
-------------------------

Checklist
---------

* Do all tests pass?
* Are the changes tested in the upstream/dependent software?


Steps
-----
1. Create a new tag with git on your local working copy::

   git tag v1.1 -m "Release message"

2. Push the tag to origin::

   git push --tags

3. Create a source distribution to upload on PyPI::
   
   python setup.py sdist
 
   This will create a $name-$version.tar.gz file in the dist directory.

4. Upload on PyPI with twine (you need to have an account on PyPI for that)::
   
   pip install twine
   twine upload dist/$name-$version.tar.gz --user $user --password $pass

5. Update the conda recipe for Omnia channel
  a. Create a fork of https://github.com/omnia-md/conda-recipes
  b. Edit the recipe (meta.yml) to match the new dependencies and version numbers.
  c. Create a pull request on https://github.com/omnia-md/conda-recipes/pulls to
     have your changes tested and merged.
  d. Sit back and wait for the packages being build.
