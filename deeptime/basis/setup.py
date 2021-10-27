def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('basis', parent_package, top_path)

    config.add_extension('_basis_bindings',
                         sources=['src/basis_bindings.cpp'],
                         include_dirs=[],
                         language='c++',
                         )
    return config
