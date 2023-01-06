#
# Root of the warwickmodel module.
# Provides access to all shared functionality.
#
# This file is part of WARWICKMODEL
# (https://github.com/I-Bouros/warwick-covid-transmission.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""epimodels is a Epidemiology Modelling library.
It contains functionality for creating regional, contact population matrices
as well as modelling of the number of cases of infections by compartment
during an outbreak of the SARS-Cov-2 virus.

The submodule epimodels.inference provides functionality for running parameter
inference on all our models using both optimisation and sampling methods, using
the PINTS and epimodels python modules.
"""

# Import version info
from .version_info import VERSION_INT, VERSION  # noqa

# Import inference submodule
from . import inference  # noqa

# Import models
from .model import WarwickLancSEIRModel  # noqa

# Import model parameter controller classes
from._parameters import (  # noqa
    ICs,
    RegParameters,
    DiseaseParameters,
    SimParameters,
    ParametersController
)
