
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('markovprocess', parent_package, top_path)

    config.add_extension('_markovprocess_bindings',
                         sources=['_bindings/src/markovprocess_module.cpp'],
                         include_dirs=['_bindings/include'],
                         language='c++',
                         )

    config.add_subpackage('msm')
    config.add_subpackage('hmm')
    config.add_subpackage('bhmm')

    return config
