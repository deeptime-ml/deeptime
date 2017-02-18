#!/bin/bash
if [[ $OSTYPE == darwin* ]]; then
     export CFLAGS="-headerpad_max_install_names"
     export CXXFLAGS=$CFLAGS
fi
$PYTHON setup.py install --single-version-externally-managed --record record.txt
