#
# Root of the warwickmodel module.
# Provides access to all shared functionality.
#
# This file is part of WARWICKMODEL
# (https://github.com/I-Bouros/warwick-covid-transmission.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""warwickmodel is a Epidemiology Modelling library based on ``epimodels``
(https://github.com/I-Bouros/multi-epi-model-cross-analysis.git).
It contains functionality for creating regional modelling of the number of
cases of infections, hospitalisations and deaths for different age-groups,
vaccination statuses and regions during an outbreak of the SARS-Cov-2 virus.

The submodule warwickmodel.inference provides functionality for running
parameter inference on all our models using both optimisation and
sampling methods, using the PINTS and epimodels python modules.
"""

# Import version info
from .version_info import VERSION_INT, VERSION  # noqa

# Import inference submodule
from .inference import (  # noqa
    WarwickLancLogLik,
    WarwickLancLogPrior,
    WarwickLancSEIRInfer
)

# Import models
from .model import WarwickLancSEIRModel  # noqa

# Import model parameter controller classes
from._parameters import (  # noqa
    ICs,
    RegParameters,
    Transmission,
    DiseaseParameters,
    SimParameters,
    VaccineParameters,
    SocDistParameters,
    ParametersController
)
