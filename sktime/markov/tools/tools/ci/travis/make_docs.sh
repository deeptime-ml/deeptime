#!/bin/bash
function install_deps {
	wget https://github.com/jgm/pandoc/releases/download/1.13.2/pandoc-1.13.2-1-amd64.deb \
		-O pandoc.deb
	dpkg -x pandoc.deb $HOME

	export PATH=$PATH:$HOME/usr/bin
	# try to execute pandoc
	pandoc --version
	
	conda install -q --yes $doc_deps
	pip install -r requirements-build-doc.txt wheel
}

function build_doc {
	pushd doc; make ipython-rst html
	# workaround for docs dir => move doc to build/docs afterwards
	# travis (currently )expects docs in build/docs (should contain index.html?) 
	mv build/html ../build/docs
	popd
}

function deploy_doc {
	echo "[distutils]
index-servers = pypi

[pypi]
username:marscher
password:${pypi_pass}" > ~/.pypirc

	python setup.py upload_docs
}

# build docs only for python 2.7 and for normal commits (not pull requests) 
if [[ $TRAVIS_PYTHON_VERSION = "2.7" ]] && [[ "${TRAVIS_PULL_REQUEST}" = "false" ]]; then
	install_deps
	build_doc
	deploy_doc
fi
