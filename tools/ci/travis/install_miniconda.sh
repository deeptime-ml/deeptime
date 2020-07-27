#!/bin/bash

# make TARGET overrideable with env
: ${TARGET:=$HOME/miniconda}

function install_miniconda {
	echo "installing miniconda to $TARGET"
	wget http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O mc.sh -o /dev/null
	bash mc.sh -b -f -p $TARGET
	# taken from conda-smithy
	conda config --remove channels defaults
	conda config --add channels defaults
	conda config --add channels conda-forge
	conda config --set show_channel_urls true
}

install_miniconda
export PATH=$TARGET/bin:$PATH
