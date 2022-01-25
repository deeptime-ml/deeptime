def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('sparse', parent_package, top_path)
    config.add_extension('_mle_sparse_bindings', sources=['_bindings/src/mle_sparse_module.cpp'], language='c++')
    return config
