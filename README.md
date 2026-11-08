# deeptime

[![Python versions](https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fpypi.org%2Fpypi%2Fdeeptime%2Fjson&query=%24.urls%5B0%3A24%3A4%5D.python_version&label=python)](https://pypi.org/project/deeptime) [![Documentation](https://img.shields.io/badge/docs-deeptime--ml.github.io-blue)](https://deeptime-ml.github.io/) [![License: LGPL v3](https://img.shields.io/badge/License-LGPL%20v3-blue.svg)](https://www.gnu.org/licenses/lgpl-3.0) [![Tests](https://github.com/deeptime-ml/deeptime/actions/workflows/tests.yml/badge.svg)](https://github.com/deeptime-ml/deeptime/actions/workflows/tests.yml) [![codecov](https://codecov.io/gh/deeptime-ml/deeptime/branch/main/graph/badge.svg?token=MgQZqDM4sK)](https://codecov.io/gh/deeptime-ml/deeptime) [![DOI](https://img.shields.io/badge/DOI-10.1088%2F2632--2153%2Fac3de0-blue)](https://doi.org/10.1088/2632-2153/ac3de0)

Deeptime is a general purpose Python library offering various tools to estimate dynamical models 
based on time-series data including conventional linear learning methods, such as Markov State 
Models (MSMs), Hidden Markov Models (HMMs) and Koopman models, as well as kernel and 
deep learning approaches such as VAMPnets and deep MSMs. The library is largely compatible 
with scikit-learn, having a range of Estimator classes for these different models, but in 
contrast to scikit-learn also provides Model classes, e.g., in the case of an MSM, 
which provide a multitude of analysis methods to compute interesting thermodynamic, kinetic 
and dynamical quantities, such as free energies, relaxation times and transition paths.

## Main components

|  <!-- -->  |  <!-- -->  |  <!-- -->  |
|  :---:	|  :---:  |  :---:  |
|  Dimension reduction  |  Deep dimension reduction  |  SINDy  |
|  [![Dimension reduction](https://user-images.githubusercontent.com/1685266/208686380-087687e0-4dfa-4d27-a2a0-957c33566276.png)](https://deeptime-ml.github.io/latest/index_dimreduction.html) |  [![Deep dimension reduction](https://user-images.githubusercontent.com/1685266/208686212-f84f0a5b-a014-49d1-a469-dfa8a661d555.png)](https://deeptime-ml.github.io/latest/index_deepdimreduction.html)  |  [![SINDy](https://user-images.githubusercontent.com/1685266/208684380-d0234430-50fb-4a62-8d97-73ce1ebf2832.png)](https://deeptime-ml.github.io/latest/notebooks/sindy.html)  |
|  Markov state models  |  Hidden Markov models  |  Datasets  |
|  [![MSMs](https://user-images.githubusercontent.com/1685266/208686588-2e8b960b-06b0-4633-93a6-5df1e5b63209.png)](https://deeptime-ml.github.io/latest/index_msm.html) |  [![HMMs](https://user-images.githubusercontent.com/1685266/208683917-ef7acb41-062c-4503-b48d-dc7718779d9a.png)](https://deeptime-ml.github.io/latest/notebooks/hmm.html)  |  [![Datasets](https://user-images.githubusercontent.com/1685266/208684805-45c82242-6a8c-43f1-88b8-add1af4e7438.png)](https://deeptime-ml.github.io/latest/index_datasets.html)  |

## Installation

| [![conda-forge](https://img.shields.io/conda/v/conda-forge/deeptime?color=brightgreen&label=conda-forge)](https://github.com/conda-forge/deeptime-feedstock) | [![PyPI](https://img.shields.io/pypi/v/deeptime?color=brightgreen)](https://pypi.org/project/deeptime) |
|:-:|:-:|
| `conda install -c conda-forge deeptime` | `pip install deeptime` |

Both channels ship compiled binaries for Linux, macOS (x86_64 and arm64), and Windows.
Optional extras: `pip install "deeptime[deep-learning]"` pulls in PyTorch for VAMPnets,
time-lagged autoencoders and deep MSMs; `pip install "deeptime[plotting]"` adds matplotlib and
networkx for `deeptime.plots`.

## Quickstart

A Markov state model from a two-dimensional toy time series with two metastable states:
project onto the slow coordinate with TICA, discretize with k-means, count transitions and
estimate the MSM.

```python
from deeptime.data import ellipsoids
from deeptime.decomposition import TICA
from deeptime.clustering import KMeans
from deeptime.markov import TransitionCountEstimator
from deeptime.markov.msm import MaximumLikelihoodMSM

traj = ellipsoids(seed=17).observations(n_steps=20000)  # ndarray of shape (20000, 2)

tica = TICA(lagtime=10, dim=1).fit_fetch(traj)  # fit_fetch() fits and returns the model
proj = tica.transform(traj)  # slow coordinate, shape (20000, 1)
dtraj = KMeans(n_clusters=20).fit_transform(proj)  # discrete trajectory of cluster indices

counts = TransitionCountEstimator(lagtime=10, count_mode='sliding').fit_fetch(dtraj)
msm = MaximumLikelihoodMSM(reversible=True).fit_fetch(counts)

print(msm.timescales(k=2))  # ≈ [16.5  3.1] steps; the process generating the data has 16.2
```

The returned `MarkovStateModel` also provides the stationary distribution, PCCA+ metastable
sets, mean first passage times, reactive fluxes and more. The
[MSM tutorial](https://deeptime-ml.github.io/latest/notebooks/mlmsm.html) continues from here
with implied timescales and a Chapman–Kolmogorov test to validate the model.

## More examples

### VAMPnets

Learn a nonlinear two-state decomposition with a
[VAMPnet](https://deeptime-ml.github.io/latest/notebooks/vampnets.html), a neural network
trained to maximize the VAMP score (requires PyTorch):

```python
import torch
from deeptime.data import sqrt_model
from deeptime.decomposition import VAMP
from deeptime.decomposition.deep import VAMPNet
from deeptime.util.data import TrajectoryDataset

_, traj = sqrt_model(n_samples=10000, seed=17)  # two metastable states that are not linearly separable

dataset = TrajectoryDataset(lagtime=1, trajectory=traj.astype('float32'))
loader = torch.utils.data.DataLoader(dataset, batch_size=512, shuffle=True)
lobe = torch.nn.Sequential(torch.nn.Linear(2, 32), torch.nn.ELU(), torch.nn.Linear(32, 2), torch.nn.Softmax(dim=1))
vampnet = VAMPNet(lobe=lobe, learning_rate=1e-3).fit(loader, n_epochs=20).fetch_model()

print(VAMP(lagtime=1, observable_transform=vampnet).fit_fetch(traj).score(2))  # ≈ 1.77
print(VAMP(lagtime=1, dim=1).fit_fetch(traj).score(2))  # ≈ 1.28, linear VAMP cannot resolve the two states
```

The trained `vampnet` transforms frames to soft state memberships and plugs into the rest of
the library as an `observable_transform`, as shown, or as input to clustering and MSM
estimation.

### Hidden Markov models

Segment a noisy one-dimensional signal into hidden states with a
[hidden Markov model](https://deeptime-ml.github.io/latest/notebooks/hmm.html) with
Gaussian output:

```python
from deeptime.data import ellipsoids
from deeptime.markov.hmm import MaximumLikelihoodHMM, init

signal = ellipsoids(seed=17).observations(n_steps=20000)[:, 1]  # two overlapping Gaussians

initial_hmm = init.gaussian.from_data(signal, n_hidden_states=2, reversible=True)
hmm = MaximumLikelihoodHMM(initial_hmm, lagtime=10).fit_fetch(signal)

print(hmm.transition_model.timescales())  # ≈ [16.0]
print(hmm.hidden_state_trajectories[0][:10])  # most likely hidden state per frame
```

Longer worked examples, including an MSM of alanine dipeptide from molecular dynamics data
and coherent sets in the Bickley jet, are collected in the
[examples section](https://deeptime-ml.github.io/latest/index_nbexamples.html) of the
documentation.

## Citing

If you use deeptime in your work, please cite

```bibtex
@article{hoffmann2021deeptime,
  title = {Deeptime: a {P}ython library for machine learning dynamical models from time series data},
  author = {Hoffmann, Moritz and Scherer, Martin Konrad and Hempel, Tim and Mardt, Andreas and de Silva, Brian and Husic, Brooke Elena and Klus, Stefan and Wu, Hao and Kutz, J. Nathan and Brunton, Steven and No{\'e}, Frank},
  journal = {Machine Learning: Science and Technology},
  volume = {3},
  number = {1},
  pages = {015009},
  year = {2021},
  publisher = {IOP Publishing},
  doi = {10.1088/2632-2153/ac3de0}
}
```

## Building from source

Please see the [developer guide](https://deeptime-ml.github.io/latest/index_dev.html).

## Contributing

Bug reports, feature requests and pull requests are welcome on
[GitHub](https://github.com/deeptime-ml/deeptime). The
[developer guide](https://deeptime-ml.github.io/latest/index_dev.html) describes the
estimator/model interfaces, how to build from source and how to run the tests.

Deeptime is distributed under the [LGPL-3.0](https://github.com/deeptime-ml/deeptime/blob/main/LICENSE.txt) license.
