
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    config = Configuration('estimation', parent_package, top_path)
    config.add_subpackage('dense')
    config.add_subpackage('sparse')

    return config
