from pathlib import Path

from numpy.distutils.misc_util import Configuration


def configuration(parent_package='', top_path=None):
    config = Configuration('mle', parent_package, top_path)

    config.add_extension('newton.objective_sparse_ops',
                         sources=['newton/objective_sparse_ops.cpp'],
                         include_dirs=[Path(top_path) / 'deeptime' / 'src' / 'include'],
                         language='c++',
                         extra_compile_args=['-fvisibility=hidden'])

    config.add_subpackage('newton')
    return config
