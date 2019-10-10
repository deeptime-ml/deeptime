import sys

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

import versioneer

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

    def build_extension(self, ext):
        from numpy import get_include as _np_inc
        np_inc = _np_inc()
        pybind_inc = 'lib/pybind11/include'

        ext.include_dirs.append(np_inc)
        ext.include_dirs.append(pybind_inc)

        if supports_omp(self.compiler):
            ext.extra_compile_args += ['-fopenmp' if sys.platform != 'darwin' else '-fopenmp=libiomp5']
            if sys.platform.startswith('linux'):
                ext.extra_link_args += ['-lgomp']
            elif sys.platform == 'darwin':
                ext.extra_link_args += ['-liomp5']
            else:
                raise ValueError("Hmm.")
            ext.define_macros += [('USE_OPENMP', None)]

        super(Build, self).build_extension(ext)

cmdclass = versioneer.get_cmdclass()
cmdclass['build_ext'] = Build

metadata = \
    dict(
        name='scikit-time',
        version=versioneer.get_version(),
        author='cmb',
        author_email='nope',
        description='scikit-time project',
        long_description='',
        ext_modules=[
            Extension('sktime.covariance.util.covar_c._covartools', sources=[
                'sktime/covariance/util/covar_c/covartools.cpp',
            ], language='c++'),
            Extension('sktime.numeric.eig_qr', sources=[
                'sktime/numeric/eig_qr.pyx'],
                      language_level=3),
            Extension('sktime.clustering._clustering_bindings', sources=[
                'sktime/clustering/src/clustering_module.cpp'
            ], include_dirs=['sktime/clustering/include'],
                      language='c++', extra_compile_args=['-std=c++17']),
            Extension('sktime.markovprocess._markovprocess_bindings', sources=[
                'sktime/markovprocess/src/markovprocess_module.cpp'
            ], include_dirs=['sktime/markovprocess/include'],
                      language='c++', extra_compile_args=['-std=c++17']),
        ],
        cmdclass=cmdclass,
        zip_safe=False,
        install_requires=['numpy'],
        packages=find_packages(),
    )

# workaround for https://reviews.llvm.org/D8467, see https://github.com/pybind/pybind11/issues/1818
if sys.platform == 'darwin':
   for e in metadata['ext_modules']:
       e.extra_compile_args.append('-fsized-deallocation')

if __name__ == '__main__':
    setup(**metadata)
