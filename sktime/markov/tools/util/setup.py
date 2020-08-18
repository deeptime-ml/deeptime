from pathlib import Path


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('util', parent_package, top_path)

    config.add_extension('kahandot',
                         sources=['kahandot/kahandot_module.cpp'],
                         include_dirs=[Path(top_path) / 'sktime' / 'src' / 'include'],
                         language='c++')
    return config
