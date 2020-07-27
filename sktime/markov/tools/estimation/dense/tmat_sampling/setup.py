import numpy as np


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('tmat_sampling', parent_package, top_path)

    config.add_library('rnglib', sources=['src/rnglib/rnglib.c', 'src/rnglib/ranlib.c'])

    config.add_extension('sampler_rev',
                         sources=['sampler_rev.pyx',
                                  'src/sample_rev.c'],
                         include_dirs=['src/', np.get_include()],
                         libraries=['rnglib'])

    config.add_extension('sampler_revpi',
                         sources=['sampler_revpi.pyx',
                                  'src/sample_revpi.c'],
                         include_dirs=['src/', np.get_include()],
                         libraries=['rnglib'])

    return config
