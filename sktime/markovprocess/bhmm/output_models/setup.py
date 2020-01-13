def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('output_models', parent_package, top_path)
    config.add_extension('_bhmm_output_models', sources=['src/bhmm_output_models_module.cpp'],
                         language='c++', extra_compile_args=['-O3']
                         )

    return config
