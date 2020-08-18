import numpy as np


def configuration(parent_package='', top_path=None):
    config = np.distutils.misc_util.Configuration('mle', parent_package, top_path)
    config.add_extension('newton.objective_sparse',
                         sources=['newton/objective_sparse.pyx'],
                         include_dirs=[np.get_include()])
    config.add_subpackage('newton')

    return config
