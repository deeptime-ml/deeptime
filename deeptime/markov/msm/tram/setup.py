def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('tram', parent_package, top_path)

    config.add_extension('_tram_bindings',
                     sources=['_bindings/src/tram_module.cpp'],
                     include_dirs=['_bindings/include', '../../../numeric/_bindings/include'],
                     language='c++',
                     )

    return config
