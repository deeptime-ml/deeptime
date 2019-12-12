
def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration

    config = Configuration('bhmm', parent_package, top_path)

    config.add_subpackage('estimators')
    config.add_subpackage('hidden')
    config.add_subpackage('hmm')
    config.add_subpackage('init')
    config.add_subpackage('output_models')
    config.add_subpackage('output_models.impl_c')
    config.add_subpackage('util')

    return config
