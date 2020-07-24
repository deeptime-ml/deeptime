import numpy as np
import os


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('mle', parent_package, top_path)
    common_include_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                      '..', '..', '..', 'util', 'include'))
    assert os.path.exists(common_include_dir)
    config.add_extension('mle_trev',
                         sources=['mle_trev.pyx',
                                  'src/_mle_trev.c'],
                         include_dirs=['src/', np.get_include(), common_include_dir])

    config.add_extension('mle_trev_given_pi',
                         sources=['mle_trev_given_pi.pyx',
                                  'src/_mle_trev_given_pi.c'],
                         include_dirs=['src/', np.get_include(), common_include_dir])

    config.add_extension('newton.objective_sparse',
                         sources=['newton/objective_sparse.pyx'],
                         include_dirs=[np.get_include()])

    config.add_subpackage('newton')

    return config
