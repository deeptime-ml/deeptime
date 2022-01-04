def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('numeric', parent_package, top_path)

    config.add_extension('eig_qr',
                         sources=['deeptime/numeric/eig_qr.pyx'],
                         )
    config.add_extension('_numeric_bindings',
                         sources=['_bindings/src/numeric_module.cpp'],
                         include_dirs=['_bindings/include'],
                         language='c++')

    return config
