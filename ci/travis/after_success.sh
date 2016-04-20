#!/bin/bash
if [ "$TRAVIS_PULL_REQUEST" = true ]; then
    echo "This is a pull request. No deployment will be done."; exit 0
fi


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

# call cleanup
python dev_pkgs_del_old.py
