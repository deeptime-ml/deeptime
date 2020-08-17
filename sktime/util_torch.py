from .util import module_available

if not module_available("torch"):
    raise ValueError("Importing this module is only possible with a working installation of PyTorch.")
del module_available

from pathlib import Path
from typing import List

import numpy as np

import torch


class OutputHandler(object):
    def __init__(self, output_dir, keep_n_checkpoints, validate: bool = False, test: bool = False):
        from torch.utils.tensorboard import SummaryWriter

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoint_dir = self._init_dir("checkpoints")
        self.samples_dir = self._init_dir("samples")
        self.logs_dir = self._init_dir("logs")

        self.logs_writer = SummaryWriter(log_dir=self.logs_dir / "train")
        if validate:
            self.logs_writer_val = SummaryWriter(log_dir=self.logs_dir / "val")
        else:
            self.logs_writer_val = None
        if test:
            self.logs_writer_test = SummaryWriter(log_dir=self.logs_dir / "test")
        else:
            self.logs_writer_test = None

        self.current_writer = self.logs_writer
        self.keep_n_checkpoints = keep_n_checkpoints

    def train(self):
        self.current_writer = self.logs_writer

    def eval(self):
        self.current_writer = self.logs_writer_val

    def test(self):
        self.current_writer = self.logs_writer_test

    def _init_dir(self, directory: str):
        p = self.output_dir / directory
        p.mkdir(parents=True, exist_ok=True)
        return p

    def make_checkpoint(self, step, dictionary):
        outfile = self.checkpoint_dir / f"checkpoint_{step}.ckpt"
        torch.save(dictionary, outfile)
        return outfile

    def prune_checkpoints(self):
        checkpoints = list(self.checkpoint_dir.glob("*.ckpt"))
        steps = []
        for ckpt in checkpoints:
            fname = str(Path(ckpt).name)
            steps.append(int("".join(list(c for c in filter(str.isdigit, fname)))))
        while len(steps) > self.keep_n_checkpoints:
            oldest = min(steps)
            oldest_path = self.checkpoint_dir.joinpath(
                "checkpoint_{}.ckpt".format(oldest)
            )
            oldest_path.unlink()
            steps.remove(oldest)

    def latest_checkpoint(self):
        checkpoints = list(self.checkpoint_dir.glob("*.ckpt"))
        if len(checkpoints) > 0:
            steps = []
            for ckpt in checkpoints:
                fname = str(Path(ckpt).name)
                steps.append(int("".join(list(c for c in filter(str.isdigit, fname)))))
            latest = max(steps)
            latest_ckpt = self.checkpoint_dir / f"checkpoint_{latest}.ckpt"
            return latest_ckpt
        else:
            return None

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        self.current_writer.add_scalar(tag, scalar_value, global_step=global_step, walltime=walltime)


class Stats(object):
    r""" Object that collects training statistics in a certain group. """

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
        return np.array(self._stats)

    def write(self, writer: OutputHandler, global_step: int = None,
              walltime: float = None, clear: bool = True):
        r"""Writes the current statistics using a tensorboard SummaryWriter or an :class:`OutputHandler`.

        Parameters
        ----------
        writer : SummaryWriter or OutputHandler
            A tensorboard summary writer or :class:`OutputHandler` which is used to write statistics.
        global_step : int, optional, default=None
            Optionally the global step value to record.
        walltime: float, optional, default=None
            Optionally the walltime to record.
        clear : bool, default=True
            Whether to clear the statistics, see also :meth:`clear`.
        """
        stats = torch.stack(self._stats)
        for ix, item in enumerate(self._items):
            name = self._group + "/" + item
            value = torch.mean(stats[..., ix]).cpu().numpy()
            writer.add_scalar(name, value, global_step=global_step, walltime=walltime)
        if clear:
            self.clear()

    def print(self, step, max_steps=None, clear=True):
        r""" Prints the collected statistics and potentially clears it.

        Parameters
        ----------
        step : int
            The step, purely cosmetical.
        max_steps : int, optional, default=None
            Optionally maximum number of steps, purely cosmetical.
        clear: bool, default=True
            Whether to clear the statistics after printing.
        """
        stats = torch.stack(self._stats)
        if max_steps is None:
            print(f"Step {step}:")
        else:
            print(f"Step {step}/{max_steps}:")
        for ix, item in enumerate(self._items):
            name = self._group + "/" + item
            value = torch.mean(stats[..., ix]).cpu().numpy()
            print(f"\t{name}: {value}")
        if clear:
            self.clear()

    def clear(self):
        r""" Empties the statistics. This is default behavior if statistics are written to a summary file, but
        sometimes it can be useful to track statistics for some more time and eventually clear it manually. """
        self._stats.clear()
