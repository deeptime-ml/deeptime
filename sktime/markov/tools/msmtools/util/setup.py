import numpy as np


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('util', parent_package, top_path)

    config.add_extension('kahandot',
                         sources=['kahandot_src/_kahandot.c', 'kahandot_src/kahandot.pyx'],
                         include_dirs=['kahandot_src/', np.get_include()])
    return config
