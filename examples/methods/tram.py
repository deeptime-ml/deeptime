from deeptime.markov.msm import TRAM
import numpy as np

tram = TRAM()

tram.fit_fetch(np.ones(1))
