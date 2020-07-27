import numpy as np


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('mle', parent_package, top_path)

    config.add_extension('mle_trev',
                         sources=['mle_trev.pyx',
                                  'src/_mle_trev.c',
                                  ],
                         include_dirs=['src/', np.get_include()]
                         )

    config.add_extension('mle_trev_given_pi',
                         sources=['mle_trev_given_pi.pyx',
                                  'src/_mle_trev_given_pi.c'],
                         include_dirs=['src/', np.get_include()])

    return config
