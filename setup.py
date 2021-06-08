import sys
from pathlib import Path

import setuptools

from numpy.distutils.command.build_ext import build_ext

import versioneer
import pybind11

CCODE_TEMPLATE = """{includes}
int main(void) {{
    {code}
    return 0;
}}"""


def _gen_ccode(includes, code):
    if not isinstance(includes, (list, tuple)):
        includes = [includes]
    return CCODE_TEMPLATE.format(includes="\n".join(["#include <{}>".format(inc) for inc in includes]), code=code)


def supports_omp(cc):
    import os
    import tempfile
    import shutil
    from copy import deepcopy
    from distutils.errors import CompileError, LinkError

    cc = deepcopy(cc)  # avoid side-effects
    if sys.platform == 'darwin':
        cc.add_library('iomp5')
    elif sys.platform.startswith('linux'):
        cc.add_library('gomp')

    tmpdir = None
    try:
        tmpdir = tempfile.mkdtemp()
        tmpfile = tempfile.mkstemp(suffix=".c", dir=tmpdir)[1]
        with open(tmpfile, 'w') as f:
            f.write(_gen_ccode("omp.h", "omp_get_num_threads();"))
        obj = cc.compile([os.path.abspath(tmpfile)], output_dir=tmpdir)
        cc.link_executable(obj, output_progname=os.path.join(tmpdir, 'a.out'))
    except (CompileError, LinkError):
        return False
    finally:
        # cleanup
        if tmpdir is not None:
            shutil.rmtree(tmpdir, ignore_errors=True)
    return True


class Build(build_ext):

    def build_extensions(self):
        extra_compile_args = []
        extra_link_args = []
        define_macros = []

        from numpy import get_include as _np_inc
        np_inc = _np_inc()
        pybind_inc = Path(pybind11.get_include())
        common_inc = Path('deeptime') / 'src' / 'include'

        if self.compiler.compiler_type == 'msvc':
            cxx_flags = ['/EHsc', '/std:c++latest', '/DVERSION_INFO=\\"%s\\"' % self.distribution.get_version()]
            extra_link_args.append('/machine:X64')
        else:
            cxx_flags = ['-std=c++17']
            extra_compile_args += ['-pthread']
            extra_link_args = ['-lpthread']
        has_openmp = supports_omp(self.compiler)
        if has_openmp:
            extra_compile_args += ['-fopenmp' if sys.platform != 'darwin' else '-fopenmp=libiomp5']
            if sys.platform.startswith('linux'):
                extra_link_args += ['-lgomp']
            elif sys.platform == 'darwin':
                extra_link_args += ['-liomp5']
            else:
                raise ValueError("Should not happen.")
            define_macros += [('USE_OPENMP', None)]

        for ext in self.extensions:
            ext.include_dirs.append(np_inc)
            ext.include_dirs.append(pybind_inc.resolve())
            ext.include_dirs.append(common_inc.resolve())
            if ext.language == 'c++':
                ext.extra_compile_args += cxx_flags
                ext.extra_compile_args += extra_compile_args
                ext.extra_link_args += extra_link_args
                ext.define_macros += define_macros

        super(Build, self).build_extensions()


cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = Build

metadata = \
    dict(
        name='deeptime',
        version=versioneer.get_version(),
        author='Moritz Hoffmann',
        author_email='moritz.hoffmann@fu-berlin.de',
        url='http://github.com/deeptime-ml/deeptime',
        description='Python library for analysis of time series data including dimensionality reduction, '
                    'clustering, and Markov model estimation.',
        long_description='Deeptime is a Python library for analysis for time series data. '
                         'In particular, methods for dimensionality reduction, clustering, and Markov '
                         'model estimation are implemented. It is available for Python 3.6+.',
        cmdclass=cmdclass,
        zip_safe=False,
        setup_requires=['cython'],
        install_requires=['numpy', 'scipy', 'scikit-learn', 'threadpoolctl'],
        extras_require={
            'deep-learning': ['pytorch'],
            'plotting': ['matplotlib', 'networkx']
        },
        package_data={
            'deeptime.data': ['data/*.npz'],
            'deeptime.src.include': ['*.h'],
            'deeptime.clustering.include': ['*.h'],
            'deeptime.markov._bindings.include': ['*.h'],
            'deeptime.markov.hmm._bindings.include': ['*.h'],
            'deeptime.data.include': ['*.h']
        },
        include_package_data=True,
        python_requires='>= 3.6',
    )


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration(None, parent_package, top_path)
    config.set_options(ignore_setup_xxx_py=True,
                       assume_default_configuration=True,
                       delegate_options_to_subpackages=True,
                       # quiet=True,
                       )
    config.add_subpackage('deeptime')
    return config


if __name__ == '__main__':
    import os

    from numpy.distutils.core import setup
    metadata['configuration'] = configuration
    setup(**metadata)
