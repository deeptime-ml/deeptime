import pytest
from numpy.testing import assert_equal, assert_

pytest.importorskip("torch")
import torch

from deeptime.util.torch import CheckpointManager, Stats


@pytest.mark.parametrize("n_ckpts", [1, 2, 5])
@pytest.mark.parametrize("mode", ['min', 'max'])
def test_checkpoint_manager(tmp_path, n_ckpts, mode):
    output_dir = tmp_path / 'mgr'
    mgr = CheckpointManager(output_dir, keep_n_checkpoints=n_ckpts, best_metric_mode=mode)
    assert_equal(mgr.keep_n_checkpoints, n_ckpts)
    assert_equal(mgr.best_metric_mode, mode)
    assert_equal(mgr.best_metric_value, None)

    for step in range(50):
        mgr.step(step, metric_value=step, models={})

    best = torch.load(output_dir / 'best.ckpt')
    assert_equal(mgr.best_metric_value, best['step'])
    if mode == 'min':
        assert_equal(mgr.best_metric_value, 0)
    else:
        assert_equal(mgr.best_metric_value, 49)

    for i in range(50 - n_ckpts):
        assert_(not (output_dir / f'checkpoint_{i}.ckpt').exists())
    for i in range(50 - n_ckpts, 50):
        assert_((output_dir / f'checkpoint_{i}.ckpt').exists())


def test_stats():
    stats = Stats("mygroup", items=['score1', 'score2'])
    assert_equal(stats.group, 'mygroup')
    assert_equal(stats.items, ['score1', 'score2'])

    stats.add([torch.tensor(1.), torch.tensor(1.)])
    stats.add([torch.tensor(2.), torch.tensor(2.)])
    stats.add([torch.tensor(3.), torch.tensor(3.)])
    assert_equal(len(stats.samples), 3)
    for i in range(3):
        assert_equal(stats.samples[i], [i+1, i+1])

    names = []
    values = []
    steps = []
    walltimes = []

    class WriterMock(object):

        def add_scalar(self, name, value, global_step, walltime):
            names.append(name)
            values.append(value)
            steps.append(global_step)
            walltimes.append(walltime)

    writer = WriterMock()
    stats.write(writer, global_step=1, walltime=10, clear=True)

    assert_equal(names[0], 'mygroup/score1')
    assert_equal(names[1], 'mygroup/score2')
    assert_equal(values, [2., 2.])
    assert_equal(steps, [1, 1])
    assert_equal(walltimes, [10, 10])
