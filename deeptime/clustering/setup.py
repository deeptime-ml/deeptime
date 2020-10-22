def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('clustering', parent_package, top_path)

    config.add_extension('_clustering_bindings',
                         sources=['src/clustering_module.cpp'],
                         include_dirs=['include'],
                         language='c++',
                         )
    return config
