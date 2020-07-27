
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('msmtools', parent_package, top_path)

    config.add_subpackage('analysis')
    config.add_subpackage('analysis.dense')
    config.add_subpackage('analysis.sparse')

    config.add_subpackage('dtraj')
    config.add_subpackage('estimation')

    config.add_subpackage('flux')
    config.add_subpackage('flux.dense')
    config.add_subpackage('flux.sparse')

    config.add_subpackage('generation')
    config.add_subpackage('util')

    from Cython.Build import cythonize
    config.ext_modules = cythonize(
        config.ext_modules,
        compiler_directives={'language_level': '3'})

    return config


if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
