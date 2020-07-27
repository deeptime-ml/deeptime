#!/usr/bin/env python

# This file is part of MSMTools.
#
# Copyright (c) 2015, 2014 Computational Molecular Biology Group, Freie Universitaet Berlin (GER)
#
# MSMTools is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

""" MSMTools

MSMTools contains an API to estimate and analyze Markov state models.
"""
DOCLINES = __doc__.split("\n")

import sys
import versioneer
import warnings

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Environment :: Console
Environment :: MacOS X
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Natural Language :: English
Operating System :: MacOS :: MacOS X
Operating System :: POSIX
Programming Language :: Python :: 2.7
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Bio-Informatics
Topic :: Scientific/Engineering :: Chemistry
Topic :: Scientific/Engineering :: Mathematics
Topic :: Scientific/Engineering :: Physics

"""


def get_cmdclass():
    from numpy.distutils.command.build_ext import build_ext

    class BuildExt(build_ext):
        def build_extensions(self):
            # setup OpenMP support
            from setup_util import detect_openmp
            openmp_enabled, additional_libs = detect_openmp(self.compiler)
            if openmp_enabled:
                warnings.warn('enabled openmp')
                if sys.platform == 'darwin':
                    omp_compiler_args = ['-fopenmp=libiomp5']
                else:
                    omp_compiler_args = ['-fopenmp']
                omp_libraries = ['-l%s' % l for l in additional_libs]
                omp_defines = [('USE_OPENMP', None)]
                for ext in self.extensions:
                    ext.extra_compile_args += omp_compiler_args
                    ext.extra_link_args += omp_libraries
                    ext.define_macros += omp_defines

            build_ext.build_extensions(self)

    cmd = versioneer.get_cmdclass()
    cmd['build_ext'] = BuildExt
    return cmd


metadata = dict(
    name='msmtools',
    maintainer='Martin K. Scherer',
    maintainer_email='m.scherer@fu-berlin.de',
    author='Benjamin Trendelkamp-Schroer',
    author_email='benjamin.trendelkamp-schroer@fu-berlin.de',
    url='http://github.com/markovmodel/msmtools',
    license='LGPLv3+',
    description=DOCLINES[0],
    long_description=open('README.rst').read(),
    version=versioneer.get_version(),
    platforms=["Windows", "Linux", "Solaris", "Mac OS-X", "Unix"],
    classifiers=[c for c in CLASSIFIERS.split('\n') if c],
    keywords='Markov State Model Algorithms',
    # runtime dependencies
    install_requires=['numpy>=1.6.0',
                      'scipy>=0.11',
                      'decorator',
                      ],
    setup_requires=['numpy', 'cython'],
    zip_safe=False,
    cmdclass=get_cmdclass(),
)


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, '', top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       # quiet=True,
                       )
    config.add_subpackage('msmtools')
    return config


# not installing?
if not(len(sys.argv) == 1 or (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
                                                  sys.argv[1] in ('--help-commands',
                                                                  '--version',
                                                                  'clean')))):
    metadata['configuration'] = configuration

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**metadata)
