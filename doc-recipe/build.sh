#!/usr/bin/env bash
# remove the source, since we depend on the built (conda) version
rm pyemma -r

pyemma_version=`python -c "import pyemma as e; print(e.version)"`
export BUILD_DIR=${PREFIX}/v${pyemma_version}

# disable progress bars
export PYEMMA_CFG_DIR=`mktemp -d`
python -c "import pyemma; pyemma.config.show_progress_bars=False; pyemma.config.save()";

# install requirements, which are not available in conda
cd doc
pip install -r requirements-build-doc.txt

# if we have the fu-berlin file system, we copy the unpublished data (bpti)
if [[ -d /group/ag_cmb/pyemma_performance/unpublished ]]; then
    cp /group/ag_cmb/pyemma_performance/unpublished ./pyemma-ipython -vuR
fi

jupyter nbextension install --py --sys-prefix widgetsnbextension
jupyter nbextension enable  --py --sys-prefix widgetsnbextension

make clean
make ipython-rst
make html

# we only want to have the html contents
mv $BUILD_DIR/html/* $BUILD_DIR
rm -rf $BUILD_DIR/doctrees

# remove the deps from $PREFIX so we have only the docs left.
pip uninstall -y -r requirements-build-doc.txt

