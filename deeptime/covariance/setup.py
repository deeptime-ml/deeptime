def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('covariance', parent_package, top_path)
    config.add_subpackage('util')
    config.add_subpackage('util.covar_c')

    return config
