#!/bin/bash

# The pull request number if the current job is a pull request, “false” if it’s not a pull request.
if [[ ! "$TRAVIS_PULL_REQUEST" == "false" ]]; then
    echo "This is a pull request. No deployment will be done."; exit 0
fi

# For builds not triggered by a pull request this is the name of the branch currently being built;
# whereas for builds triggered by a pull request this is the name of the branch targeted by the pull request (in many cases this will be master).
if [ "$TRAVIS_BRANCH" != "devel" ]; then
    echo "No deployment on BRANCH='$TRAVIS_BRANCH'"; exit 0
fi


# Deploy to binstar
conda install --yes anaconda-client jinja2
pushd .
cd $HOME/miniconda/conda-bld
FILES=*/${PACKAGENAME}-dev-*.tar.bz2
for filename in $FILES; do
    anaconda -t $BINSTAR_TOKEN upload --force -u ${ORGNAME} -p ${PACKAGENAME}-dev ${filename}
done
popd

# call cleanup only for py35, numpy111
if [[ "$CONDA_PY" == "3.5" && "$CONDA_NPY" == "111" && "$TRAVIS_OS_NAME" == "linux" ]]; then
    python devtools/ci/travis/dev_pkgs_del_old.py
else
   echo "only executing cleanup script for py35 && npy111 && linux"
fi

