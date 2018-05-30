# Compatibility Python 2/3
from __future__ import division, print_function, absolute_import
from builtins import range
# ----------------------------------------------------------------------------------------------------------------------

import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt
from opto.opto.classes.AcquisitionFunction import AcquisitionFunction
from scipy.stats import norm


class EI(AcquisitionFunction):
    """
    Expected Improvement
    """

    def __init__(self, model, logs, parameters={}):
        super(EI, self).__init__(model=model, parameters=parameters)
        self.target = parameters.get('target', None)
        # The default bias is according to [Lizotte]
        self.bias = parameters.get('bias', 0.01)
        if (self.target is None) and (logs is not None):
            # we store the best observation sof ar
            self.target = logs.get_best_objectives() - self.bias

    def set_target(self, target):
        self.target = target

    def evaluate(self, parameters):
        mean, var = self._m.predict(parameters.T)
        assert np.all(var >= 0), 'Variance can never be negative!'

        s = np.sqrt(var)
        s[s == 0] = np.spacing(1)  # To avoid numerical instabilities
        z = (self.target - mean) / s
        f = s * (z * norm.cdf(z) + norm.pdf(z))  # Derivation from [Jones 1998]
        # f = (eta - m - self.par) * norm.cdf(z) + s * norm.pdf(z) # alternative
        f = -f  # Invert f, because the optimizer currently only minimizes.

        # check for NaN and Inf errors
        assert np.all(np.invert(np.isnan(f)))
        assert np.all(np.invert(np.isinf(f)))
        return f

    # TODO: can we make __init__ use this code to avoid duplication?
    def update(self, model, logs, parameters={}):
        self._m = model  # Model
        self.target = parameters.get('target', None)
        # The default bias is according to [Lizotte]
        self.bias = parameters.get('bias', 0.01)
        if self.target is None:
            # we store the best observation so far
            self.target = logs.get_best_objectives() - self.bias
