#!/bin/bash
find . -maxdepth 1 -name \*.ipynb -exec sh -c \
  'for i do  jupyter nbconvert --ClearOutputPreprocessor.enabled=True --clear-output --inplace $i ; done' sh {} \;
