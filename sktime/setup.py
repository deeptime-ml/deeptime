
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import setuptools
    import os

    config = Configuration('sktime', parent_package, top_path)
    location = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    for pkg in setuptools.find_packages(location):
        config.add_subpackage(pkg)

    #config.add_data_dir('data')

    from Cython.Build import cythonize
    config.ext_modules = cythonize(
        config.ext_modules,
        compiler_directives={'language_level': '3'})

    return config
