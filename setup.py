from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class Build(build_ext):

    def build_extension(self, ext):
        from numpy import get_include as _np_inc
        np_inc = _np_inc()
        pybind_inc = 'lib/pybind11/include'

        ext.include_dirs.append(np_inc)
        ext.include_dirs.append(pybind_inc)
        super(Build, self).build_extension(ext)


if __name__ == '__main__':
    setup(
        name='scikit-time',
        version='0.0.1',
        author='cmb',
        author_email='nope',
        description='scikit-time project',
        long_description='',
        ext_modules=[
            Extension('sktime.covariance.util.covar_c', sources=[
                'sktime/covariance/util/covar_c/covartools.cpp',
            ], language='c++'),
            Extension('sktime.numeric.eig_qr', sources=[
                'sktime/numeric/eig_qr.pyx'],
                      language_level=3),
            Extension('sktime.clustering._bindings', sources=[
                'sktime/clustering/src/clustering_module.cpp'
            ], include_dirs=['sktime/clustering/include'],
                      language='c++', extra_compile_args=['-std=c++17']),
        ],
        cmdclass=dict(build_ext=Build),
        zip_safe=False,
        install_requires=['numpy'],
        # TODO: pep517
        #setup_requires=['cython']
    )
