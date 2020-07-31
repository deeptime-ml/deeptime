from pathlib import Path


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('dense', parent_package, top_path)

    config.add_subpackage('tmat_sampling')

    config = Configuration('mle', parent_package, top_path)
    config.add_extension('_mle_bindings',
                         sources=['_bindings/src/mle_module.cpp'],
                         include_dirs=['_bindings/include', Path(top_path) / 'sktime' / 'src' / 'include'],
                         language='c++',
                         extra_compile_args=['-fvisibility=hidden'])

    return config
