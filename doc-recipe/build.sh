rm pyemma -r

pyemma_version=`python -c "import pyemma as e; print(e.version)"` 
export BUILD_DIR=${PREFIX}/v${pyemma_version}

cd doc
pip install -r requirements-build-doc.txt


make ipython-rst
make html

# remove the deps from $PREFIX so we have only the docs left.
pip uninstall -y -r requirements-build-doc.txt
