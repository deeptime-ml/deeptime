def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('tools', parent_package, top_path)

    config.add_subpackage('analysis')
    config.add_subpackage('analysis.dense')

    config.add_subpackage('estimation')

    config.add_subpackage('flux')

    config.add_extension('kahandot',
                         sources=['kahandot/kahandot_module.cpp'],
                         language='c++')

    return config
