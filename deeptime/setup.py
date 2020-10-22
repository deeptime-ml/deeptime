def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('deeptime', parent_package, top_path)

    config.add_subpackage('clustering')
    config.add_subpackage('covariance')
    config.add_subpackage('data')
    config.add_subpackage('decomposition')
    config.add_subpackage('markov')
    config.add_subpackage('numeric')
    config.add_subpackage('kernels')
    config.add_subpackage('basis')
    config.add_subpackage('util')
    config.add_subpackage('sindy')

    from Cython.Build import cythonize
    config.ext_modules = cythonize(
        config.ext_modules,
        compiler_directives={'language_level': '3'})

    return config
