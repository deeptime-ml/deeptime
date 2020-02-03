
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('markovprocess', parent_package, top_path)

    config.add_extension('_bindings',
                         sources=['_binding/src/markovprocess_module.cpp'],
                         include_dirs=['_binding/include'],
                         language='c++',
                         )

    config.add_subpackage('msm')
    config.add_subpackage('bhmm')
    config.add_subpackage('hmm')

    return config
