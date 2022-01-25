def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('covar_c', parent_package, top_path)
    config.add_extension('_covartools', sources=['covartools.hpp', 'covartools.cpp'], language='c++')
    return config
