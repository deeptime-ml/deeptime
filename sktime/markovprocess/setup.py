
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('markovprocess', parent_package, top_path)

    config.add_extension('_markovprocess_bindings',
                         sources=['src/markovprocess_module.cpp'],
                         include_dirs=['include'],
                         language='c++',
                         )

    config.add_subpackage('bhmm')
    config.add_subpackage('generation')

    return config
