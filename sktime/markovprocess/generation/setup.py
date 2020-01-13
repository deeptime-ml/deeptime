
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('generation', parent_package, top_path)

    config.add_extension('_markovprocess_generation_bindings',
                         sources=['bindings.cpp'],
                         language='c++',
                         )

    return config
