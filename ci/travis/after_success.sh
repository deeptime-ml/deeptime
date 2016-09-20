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

