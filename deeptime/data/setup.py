from pybind11.setup_helpers import Pybind11Extension


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('data', parent_package, top_path)
    config.add_data_files(
        'data/double_well_discrete.npz',
    )
    ext = Pybind11Extension('_data_bindings',
                            sources=config.paths(['src/data_module.cpp']),
                            include_dirs=config.paths(['include']),
                            cxx_std=17)
    config.ext_modules.append(ext)
    return config
