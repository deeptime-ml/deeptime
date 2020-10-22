def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('init', parent_package, top_path)

    config.add_subpackage("discrete")
    config.add_subpackage("gaussian")

    return config
