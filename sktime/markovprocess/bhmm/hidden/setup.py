def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('hidden', parent_package=parent_package, top_path=top_path)
    config.add_extension('_bhmm_hidden_bindings', sources=['src/bhmm_hidden_module.cpp'],
                         language='c++', extra_compile_args=['-O3']
                         )

    return config
