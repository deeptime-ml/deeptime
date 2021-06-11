from deeptime.util.platform import module_available
if not module_available("torch"):
    raise ValueError("Importing this module is only possible with a working installation of PyTorch.")
del module_available

import numpy as np

from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn


def map_data(data, device=None, dtype=np.float32) -> List[torch.Tensor]:
    r""" Maps data (or list/tuple of data) to torch tensors of a specific dtype and device.

    Parameters
    ----------
    data : ndarray or tensor or list thereof
        The input data.
    device : Device or None
        The device.
    dtype : dtype, default=None
        The dtype.

    Yields
    ------
    mapped_data : list of torch Tensor
        The mapped data.
    """
    with torch.no_grad():
        if not isinstance(data, (list, tuple)):
            data = [data]
        for x in data:
            if isinstance(x, torch.Tensor):
                x = x.to(device=device)
            else:
                x = torch.from_numpy(np.asarray(x, dtype=dtype).copy()).to(device=device)
            yield x


class MLP(nn.Module):
    r""" A multilayer perceptron which can, e.g., be used as a neural network lobe for VAMPNets.

    Parameters
    ----------
    units : list of int
        The units of the fully connected layers.
    nonlinearity : callable, default=torch.nn.ELU
        A callable (like a constructor) which yields an instance of a particular activation function.
    initial_batchnorm : bool, default=True
        Whether to use batch normalization before the data enters the rest of the network.
    output_nonlinearity : callable, default=None
        The output activation/nonlinearity. If the data decomposes into states, it can make sense to use
        an output activation like softmax which produces a probability distribution over said states.
        The callable should take no arguments and produce an object of type :code:`torch.nn.Module`.
    """

    def __init__(self, units: List[int], nonlinearity=nn.ELU, initial_batchnorm: bool = False,
                 output_nonlinearity=None):
        super().__init__()
        if len(units) > 1:
            layers = []
            if initial_batchnorm:
                layers.append(nn.BatchNorm1d(units[0]))
            for fan_in, fan_out in zip(units[:-2], units[1:-1]):
                layers.append(nn.Linear(fan_in, fan_out))
                layers.append(nonlinearity())
            layers.append(nn.Linear(units[-2], units[-1]))
            if output_nonlinearity is not None:
                layers.append(output_nonlinearity())
            self._sequential = nn.Sequential(*layers)
        else:
            self._sequential = nn.Identity()

    def forward(self, inputs):
        return self._sequential(inputs)


class CheckpointManager:
    r""" A checkpoint manager for pytorch models. It can keep track of a metric, save the best model according to
    that metric, and prune too old checkpoints. """

    def __init__(self, output_dir, keep_n_checkpoints: int = 5, best_metric_mode='max'):
        r""" Creates a new checkpoint manager.

        Parameters
        ----------
        output_dir : path_like
            The output directory under which checkpoints are stored. The (according to metric) best checkpoint gets
            assigned the filename "best.ckpt", the others are enumerated as "checkpoint_{step}.ckpt".
        keep_n_checkpoints : int, default=5
            The number of sequential checkpoints to keep. The manager will prune older checkpoints based
            on filename.
        best_metric_mode : str, default='max'
            Whether the smallest or the largest metric is the best. Defaults to 'max', i.e., the largest.
        """
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._keep_n_checkpoints = keep_n_checkpoints
        self._best_metric_value = None
        self._best_metric_mode = best_metric_mode

    @property
    def keep_n_checkpoints(self) -> int:
        r""" The number of checkpoints to keep.

        :getter: Yields the number of checkpoints to keep.
        :setter: Sets the number of checkpoints to keep. Pruning can be called manually or upon next :meth:`step`.
        :type: int
        """
        return self._keep_n_checkpoints

    @keep_n_checkpoints.setter
    def keep_n_checkpoints(self, value: int):
        self._keep_n_checkpoints = value

    @property
    def best_metric_value(self) -> Optional[float]:
        r""" The current best metric value. Can initially be None.

        :getter: The best value.
        :type: float or None
        """
        return self._best_metric_value

    @property
    def best_metric_mode(self) -> str:
        r""" The mode under which the metric is taken into account. Can either be 'max', i.e., larger metric values are
        better, or 'min' for the opposite behavior.

        :getter: Yields the mode.
        :setter: Sets the mode.
        :type: str, one of ('max', 'min')
        """
        return self._best_metric_mode

    @best_metric_mode.setter
    def best_metric_mode(self, value: str):
        assert value in ('min', 'max'), f"Unknown mode {value}, supported are min and max."
        self._best_metric_mode = value

    def step(self, step: int, metric_value: float, models: Dict[str, torch.nn.Module]):
        r""" Records a step in the checkpoint manager. This automatically prunes old checkpoints according to
        :meth:`max_n_checkpoints`.

        Parameters
        ----------
        step : int
            The training step. Should be monotonically increasing during training.
        metric_value : float
            Value of the metric.
        models : dict from model name to torch module
            Dictionary of modules which should be stored in the checkpoint.
        """
        self._make_checkpoint(step, models, self._output_dir / f"checkpoint_{step}.ckpt")
        self.prune_checkpoints()
        if metric_value is not None:
            op = min if self.best_metric_mode == 'min' else max
            if self.best_metric_value is None or (metric_value == op(self.best_metric_value, metric_value)):
                self._best_metric_value = metric_value
                self._make_checkpoint(step, models, self._output_dir / f"best.ckpt")

    @staticmethod
    def _make_checkpoint(step, models: Dict[str, torch.nn.Module], outfile: Path):
        r""" Makes the actual checkpoint. """
        save_dict = {k: v.state_dict() for k, v in models.items()}
        save_dict['step'] = step
        torch.save(save_dict, outfile)
        return outfile

    def prune_checkpoints(self):
        r""" Prunes old checkpoints. """
        checkpoints = list(self._output_dir.glob("*.ckpt"))
        steps = []
        for ckpt in checkpoints:
            fname = str(Path(ckpt).name)
            filtered = "".join(list(c for c in filter(str.isdigit, fname)))
            if len(filtered) > 0:
                steps.append(int(filtered))
        while len(steps) > self.keep_n_checkpoints:
            oldest = min(steps)
            oldest_path = self._output_dir.joinpath(
                "checkpoint_{}.ckpt".format(oldest)
            )
            oldest_path.unlink()
            steps.remove(oldest)


class Stats:
    r""" Object that collects training statistics in a certain group. This can be used to track, e.g., loss values
    and validation values.

    >>> stats = Stats(group='train', items=['loss1', 'loss2'])

    Adding some artificial data (first list element belongs to loss1, second to loss2).
    This could be recording the loss during an epoch of training.
    >>> stats.add([torch.tensor(1.), torch.tensor(1.)])
    >>> stats.add([torch.tensor(2.), torch.tensor(2.)])
    >>> stats.add([torch.tensor(3.), torch.tensor(3.)])

    After the epoch, the mean value of the collected stats can be written using a tensorboard summary writer:

    .. code-block:: python
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(log_dir="path/to/train/logdir")
        stats.write(writer, global_step=step)  # this also clears the stats for the next round of collecting data.
    """

    def __init__(self, group: str, items: List[str]):
        r""" Instantiates a new stats object.

        Parameters
        ----------
        group : str
            The group this stats object belongs to.
        items : list of str
            List of strings
        """
        self._stats = []
        self._group = group
        self._items = items

    def add(self, data: List[torch.Tensor]):
        r""" Adds data to the statistics.

        Parameters
        ----------
        data : list of torch tensors
            Adds a list of tensors. Must be of same length as the number of items that are tracked in this object.
        """
        if len(data) != len(self._items):
            raise ValueError("Incompatible stats")
        self._stats.append(torch.stack([x.detach() for x in data]))

    @property
    def items(self) -> List[str]:
        r""" The items that are tracked by this object.

        :getter: Yields the items.
        :type: list of str
        """
        return self._items

    @property
    def group(self) -> str:
        r""" The group that this object belongs to (e.g., validation or train).

        :getter: Yields the group.
        :type: str
        """
        return self._group

    @property
    def samples(self):
        r""" Property to access the currently stored statistics.

        :getter: Gets the statistics.
        :type: (n_items, n_data) ndarray
        """
        return torch.stack(self._stats).cpu().numpy()

    def write(self, writer, global_step: int = None,
              walltime: float = None, clear: bool = True):
        r"""Writes the current statistics using a tensorboard SummaryWriter or an :class:`OutputHandler`.

        Parameters
        ----------
        writer : SummaryWriter
            A tensorboard summary writer used to write statistics.
        global_step : int, optional, default=None
            Optionally the global step value to record.
        walltime: float, optional, default=None
            Optionally the walltime to record.
        clear : bool, default=True
            Whether to clear the statistics, see also :meth:`clear`.
        """
        stats = torch.stack(self._stats)
        mean_stats = stats.mean(dim=0)
        for ix, item in enumerate(self._items):
            name = self._group + "/" + item
            value = torch.mean(mean_stats[ix]).cpu().numpy()
            writer.add_scalar(name, value, global_step=global_step, walltime=walltime)
        if clear:
            self.clear()

    def clear(self):
        r""" Empties the statistics. This is default behavior if statistics are written to a summary file, but
        sometimes it can be useful to track statistics for some more time and eventually clear it manually. """
        self._stats.clear()
