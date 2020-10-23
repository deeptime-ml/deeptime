def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('tmat_sampling', parent_package, top_path)
    return config
