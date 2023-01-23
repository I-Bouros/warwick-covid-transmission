#
# Parameter Classes
#
# This file is part of WARWICKMODEL
# (https://github.com/I-Bouros/warwick-covid-transmission.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for the Parameter classes for the epidemiological
model include in the `warwickmodel` Python module.

The Parameter classes store the model parameters as class features and with the
object then fed into the model when :meth:`simulate` is called on the model
class.

"""

import numpy as np
from iteration_utilities import deepflatten

import warwickmodel as wm

#
# ICs Class
#


class ICs(object):
    """ICs:
    Base class for the ICs of the model: a deterministic SEIR used
    by the Universities of Warwick and Lancaster to model the Covid-19
    epidemic and the effects of vaccines and waning immunity on the
    epidemic trajectory in different countries.

    Parameters
    ----------
    susceptibles_IC : list of lists
        Initial number of susceptibles classifed by age (column name) and
        region (row name) for different vaccination statuses (unvaccinated,
        fully-vaccinated, boosted, partially-waned, fully-waned,
        previous-variant immunity) (column name II).
    exposed1_IC : list of lists
        Initial number of exposed of the first type classifed by age
        (column name) and region (row name) for different vaccination
        statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
        fully-waned, previous-variant immunity) (column name II).
    exposed2_IC : list of lists
        Initial number of exposed of the second type classifed by age
        (column name) and region (row name) for different vaccination
        statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
        fully-waned, previous-variant immunity) (column name II).
    exposed3_IC : list of lists
        Initial number of exposed of the third type classifed by age
        (column name) and region (row name) for different vaccination
        statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
        fully-waned, previous-variant immunity) (column name II).
    exposed4_IC : list of lists
        Initial number of exposed of the forth type classifed by age
        (column name) and region (row name) for different vaccination
        statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
        fully-waned, previous-variant immunity) (column name II).
    exposed5_IC : list of lists
        Initial number of exposed of the fifth type classifed by age
        (column name) and region (row name) for different vaccination
        statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
        fully-waned, previous-variant immunity) (column name II).
    infectives_sym_IC :list of lists
        Initial number of symptomatic infectives classifed by age
        (column name) and region (row name) for different vaccination
        statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
        fully-waned, previous-variant immunity) (column name II).
    infectives_asym_IC : list of lists
        Initial number of asymptomatic infectives classifed by age
        (column name) and region (row name) for different vaccination
        statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
        fully-waned, previous-variant immunity) (column name II).
    recovered_IC : list of lists
        Initial number of recovered classifed by age (column name) and
        region (row name).

    """
    def __init__(self, model, susceptibles_IC, exposed1_IC, exposed2_IC,
                 exposed3_IC, exposed4_IC, exposed5_IC,
                 infectives_sym_IC, infectives_asym_IC, recovered_IC):
        super(ICs, self).__init__()

        # Set model
        if not isinstance(model, wm.WarwickLancSEIRModel):
            raise TypeError(
                'The model must be a Warwick-Lancaster SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(
            susceptibles_IC, exposed1_IC, exposed2_IC, exposed3_IC,
            exposed4_IC, exposed5_IC, infectives_sym_IC, infectives_asym_IC,
            recovered_IC)

        # Set ICs parameters
        self.susceptibles = susceptibles_IC
        self.exposed1 = exposed1_IC
        self.exposed2 = exposed2_IC
        self.exposed3 = exposed3_IC
        self.exposed4 = exposed4_IC
        self.exposed5 = exposed5_IC
        self.infectives_sym = infectives_sym_IC
        self.infectives_asym = infectives_asym_IC
        self.recovered = recovered_IC

    def _check_parameters_input(
            self, susceptibles_IC, exposed1_IC, exposed2_IC, exposed3_IC,
            exposed4_IC, exposed5_IC, infectives_sym_IC, infectives_asym_IC,
            recovered_IC):
        """
        Check correct format of ICs input.

        Parameters
        ----------
        susceptibles_IC : list of lists
            Initial number of susceptibles classifed by age (column name) and
            region (row name) for different vaccination statuses (unvaccinated,
            fully-vaccinated, boosted, partially-waned, fully-waned,
            previous-variant immunity) (column name II).
        exposed1_IC : list of lists
            Initial number of exposed of the first type classifed by age
            (column name) and region (row name) for different vaccination
            statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity) (column name II).
        exposed2_IC : list of lists
            Initial number of exposed of the second type classifed by age
            (column name) and region (row name) for different vaccination
            statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity) (column name II).
        exposed3_IC : list of lists
            Initial number of exposed of the third type classifed by age
            (column name) and region (row name) for different vaccination
            statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity) (column name II).
        exposed4_IC : list of lists
            Initial number of exposed of the forth type classifed by age
            (column name) and region (row name) for different vaccination
            statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity) (column name II).
        exposed5_IC : list of lists
            Initial number of exposed of the fifth type classifed by age
            (column name) and region (row name) for different vaccination
            statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity) (column name II).
        infectives_sym_IC :list of lists
            Initial number of symptomatic infectives classifed by age
            (column name) and region (row name) for different vaccination
            statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity) (column name II).
        infectives_asym_IC : list of lists
            Initial number of asymptomatic infectives classifed by age
            (column name) and region (row name) for different vaccination
            statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity) (column name II).
        recovered_IC : list of lists
            Initial number of recovered classifed by age (column name) and
            region (row name).

        """
        if np.asarray(susceptibles_IC).ndim != 2:
            raise ValueError('The inital numbers of susceptibles storage \
                format must be 2-dimensional.')
        if np.asarray(susceptibles_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        susceptibles.')
        if np.asarray(susceptibles_IC).shape[1] != 6 * self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        susceptibles.')
        for ic in np.asarray(susceptibles_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of susceptibles must be integer or\
                            float.')

        if np.asarray(exposed1_IC).ndim != 2:
            raise ValueError('The inital numbers of exposed of the first type \
                storage format must be 2-dimensional.')
        if np.asarray(exposed1_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the first type.')
        if np.asarray(exposed1_IC).shape[1] != 6 * self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the first type.')
        for ic in np.asarray(exposed1_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of exposed of the first type must \
                            be integer or float.')

        if np.asarray(exposed2_IC).ndim != 2:
            raise ValueError('The inital numbers of exposed of the second type\
                storage format must be 2-dimensional.')
        if np.asarray(exposed2_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the second type.')
        if np.asarray(exposed2_IC).shape[1] != 6 * self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the second type.')
        for ic in np.asarray(exposed2_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of exposed of the second type must\
                            be integer or float.')

        if np.asarray(exposed3_IC).ndim != 2:
            raise ValueError('The inital numbers of exposed of the third type\
                storage format must be 2-dimensional.')
        if np.asarray(exposed3_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the third type.')
        if np.asarray(exposed3_IC).shape[1] != 6 * self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the third type.')
        for ic in np.asarray(exposed3_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of exposed of the third type must\
                            be integer or float.')

        if np.asarray(exposed4_IC).ndim != 2:
            raise ValueError('The inital numbers of exposed of the fourth type\
                storage format must be 2-dimensional.')
        if np.asarray(exposed4_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the fourth type.')
        if np.asarray(exposed4_IC).shape[1] != 6 * self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the fourth type.')
        for ic in np.asarray(exposed4_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of exposed of the fourth type must\
                            be integer or float.')

        if np.asarray(exposed5_IC).ndim != 2:
            raise ValueError('The inital numbers of exposed of the fifth type\
                storage format must be 2-dimensional.')
        if np.asarray(exposed5_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the fifth type.')
        if np.asarray(exposed5_IC).shape[1] != 6 * self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        exposed of the fifth type.')
        for ic in np.asarray(exposed5_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of exposed of the fifth type must\
                            be integer or float.')

        if np.asarray(infectives_sym_IC).ndim != 2:
            raise ValueError('The inital numbers of symptomatic infectives \
                storage format must be 2-dimensional.')
        if np.asarray(infectives_sym_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        symptomatic infectives.')
        if np.asarray(infectives_sym_IC).shape[1] != 6 * self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        symptomatic infectives.')
        for ic in np.asarray(infectives_sym_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of symptomatic infectives must be\
                        integer or float.')

        if np.asarray(infectives_asym_IC).ndim != 2:
            raise ValueError('The inital numbers of asymptomatic infectives\
                storage format must be 2-dimensional.')
        if np.asarray(infectives_asym_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        asymptomatic infectives.')
        if np.asarray(infectives_asym_IC).shape[1] != 6 * self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        asymptomatic infectives.')
        for ic in np.asarray(infectives_asym_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of asymptomatic infectives must be\
                        integer or float.')

        if np.asarray(recovered_IC).ndim != 2:
            raise ValueError('The inital numbers of recovered storage format \
                must be 2-dimensional.')
        if np.asarray(recovered_IC).shape[0] != len(self.model.regions):
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        recovered.')
        if np.asarray(recovered_IC).shape[1] != self.model._num_ages:
            raise ValueError(
                    'Wrong number of rows for the inital numbers of \
                        recovered.')
        for ic in np.asarray(recovered_IC):
            for _ in ic:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'The inital numbers of recovered must be integer or \
                            float.')

    def total_population(self):
        """
        Returns the country-specific total number of individuals in each age
        group.

        Returns
        -------
        List of lists
            List of the ountry-specific total number of individuals in each age
            group using the initial conditions of the
            :class:`WarwickLancSEIRModel` the class relates to.

        """
        a = self.model._num_ages
        total_pop = 0

        ics_vac_stat = [self.susceptibles, self.exposed1, self.exposed2,
                        self.exposed3, self.exposed4, self.exposed5,
                        self.infectives_sym, self.infectives_asym]

        for _ in ics_vac_stat:
            total_pop += np.asarray(_)[:, :a] + np.asarray(_)[:, a:(2*a)] + \
                np.asarray(_)[:, (2*a):(3*a)] + np.asarray(_)[:, (3*a):(4*a)] \
                + np.asarray(_)[:, (4*a):(5*a)] + np.asarray(_)[:, (5*a):(6*a)]

        return total_pop + np.array(self.recovered)

    def __call__(self):
        """
        Returns the initial conditions of the :class:`WarwickLancSEIRModel` the
        class relates to.

        Returns
        -------
        List of lists
            List of the initial conditions of the :class:`WarwickLancSEIRModel`
            the class relates to.

        """
        ics = []
        a = self.model._num_ages

        # Susceptibles
        ics += [
            np.asarray(self.susceptibles)[:, :a],
            np.asarray(self.susceptibles)[:, a:(2*a)],
            np.asarray(self.susceptibles)[:, (2*a):(3*a)],
            np.asarray(self.susceptibles)[:, (3*a):(4*a)],
            np.asarray(self.susceptibles)[:, (4*a):(5*a)],
            np.asarray(self.susceptibles)[:, (5*a):(6*a)]]

        # Exposed unvaccinated
        ics += [
            np.asarray(self.exposed1)[:, :a],
            np.asarray(self.exposed2)[:, :a],
            np.asarray(self.exposed3)[:, :a],
            np.asarray(self.exposed4)[:, :a],
            np.asarray(self.exposed5)[:, :a]]

        # Exposed fully vaccinated
        ics += [
            np.asarray(self.exposed1)[:, a:(2*a)],
            np.asarray(self.exposed2)[:, a:(2*a)],
            np.asarray(self.exposed3)[:, a:(2*a)],
            np.asarray(self.exposed4)[:, a:(2*a)],
            np.asarray(self.exposed5)[:, a:(2*a)]]

        # Exposed boosted
        ics += [
            np.asarray(self.exposed1)[:, (2*a):(3*a)],
            np.asarray(self.exposed2)[:, (2*a):(3*a)],
            np.asarray(self.exposed3)[:, (2*a):(3*a)],
            np.asarray(self.exposed4)[:, (2*a):(3*a)],
            np.asarray(self.exposed5)[:, (2*a):(3*a)]]

        # Exposed partially waned
        ics += [
            np.asarray(self.exposed1)[:, (3*a):(4*a)],
            np.asarray(self.exposed2)[:, (3*a):(4*a)],
            np.asarray(self.exposed3)[:, (3*a):(4*a)],
            np.asarray(self.exposed4)[:, (3*a):(4*a)],
            np.asarray(self.exposed5)[:, (3*a):(4*a)]]

        # Exposed fully waned
        ics += [
            np.asarray(self.exposed1)[:, (4*a):(5*a)],
            np.asarray(self.exposed2)[:, (4*a):(5*a)],
            np.asarray(self.exposed3)[:, (4*a):(5*a)],
            np.asarray(self.exposed4)[:, (4*a):(5*a)],
            np.asarray(self.exposed5)[:, (4*a):(5*a)]]

        # Exposed previous variant
        ics += [
            np.asarray(self.exposed1)[:, (5*a):(6*a)],
            np.asarray(self.exposed2)[:, (5*a):(6*a)],
            np.asarray(self.exposed3)[:, (5*a):(6*a)],
            np.asarray(self.exposed4)[:, (5*a):(6*a)],
            np.asarray(self.exposed5)[:, (5*a):(6*a)]]

        # Symptomatic infected
        ics += [
            np.asarray(self.infectives_sym)[:, :a],
            np.asarray(self.infectives_sym)[:, a:(2*a)],
            np.asarray(self.infectives_sym)[:, (2*a):(3*a)],
            np.asarray(self.infectives_sym)[:, (3*a):(4*a)],
            np.asarray(self.infectives_sym)[:, (4*a):(5*a)],
            np.asarray(self.infectives_sym)[:, (5*a):(6*a)]]

        # Asymptomatic infected
        ics += [
            np.asarray(self.infectives_asym)[:, :a],
            np.asarray(self.infectives_asym)[:, a:(2*a)],
            np.asarray(self.infectives_asym)[:, (2*a):(3*a)],
            np.asarray(self.infectives_asym)[:, (3*a):(4*a)],
            np.asarray(self.infectives_asym)[:, (4*a):(5*a)],
            np.asarray(self.infectives_asym)[:, (5*a):(6*a)]]

        return ics + [self.recovered]

#
# RegParameters Class
#


class RegParameters(object):
    """RegParameters:
    Base class for the regional and time dependent parameters of the model: a
    deterministic SEIR used by the Universities of Warwick and Lancaster to
    model the Covid-19 epidemic and the effects of vaccines and waning
    immunity on the epidemic trajectory in different countries.

    Parameters
    ----------
    region_index : int
        Index of region for which we wish to simulate.

    """
    def __init__(self, model, region_index):
        super(RegParameters, self).__init__()

        # Set model
        if not isinstance(model, wm.WarwickLancSEIRModel):
            raise TypeError(
                'The model must be a Warwick-Lancaster SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(region_index)

        # Set regional and time dependent parameters
        self.region_index = region_index

    def _check_parameters_input(self, region_index):
        """
        Check correct format of the regional and time dependent parameters
        input.

        Parameters
        ----------
        region_index : int
            Index of region for which we wish to simulate.

        """
        if not isinstance(region_index, int):
            raise TypeError('Index of region to evaluate must be integer.')
        if region_index <= 0:
            raise ValueError('Index of region to evaluate must be >= 1.')
        if region_index > len(self.model.regions):
            raise ValueError('Index of region to evaluate is out of bounds.')

    def __call__(self):
        """
        Returns the regional and time dependent parameters of the
        :class:`WarwickLancSEIRModel` the class relates to.

        Returns
        -------
        List of lists
            List of the regional and time dependent parameters of the
            :class:`WarwickLancSEIRModel` the class relates to.

        """
        return self.region_index

#
# DiseaseParameters Class
#


class DiseaseParameters(object):
    """DiseaseParameters:
    Base class for the disease-specific parameters of the model: a
    deterministic SEIR used by the Universities of Warwick and Lancaster to
    model the Covid-19 epidemic and the effects of vaccines and waning
    immunity on the epidemic trajectory in different countries.

    Parameters
    ----------
    d : int or float or list
        Age-dependent probabilities of dispalying symptoms.
    tau : int or float
        Reduction in transmission for an asymptomatic infectious compared
        to the symptomatic case.
    we : list
        Rates of waning of immunity for current and an older variant.
    omega : int or float
        Change in susceptibility due to the variant.

    """
    def __init__(self, model, d, tau, we, omega):
        super(DiseaseParameters, self).__init__()

        # Set model
        if not isinstance(model, wm.WarwickLancSEIRModel):
            raise TypeError(
                'The model must be a Warwick-Lancaster SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(d, tau, we, omega)

        # Set disease-specific parameters
        self.tau = tau

        if isinstance(d, (float, int)):
            self.d = d * np.ones(self.model._num_ages)
        else:
            self.d = d

        self.we = we
        self.omega = omega

    def _check_parameters_input(self, d, tau, we, omega):
        """
        Check correct format of the disease-specific parameters input.

        Parameters
        ----------
        d : int or float or list
            Age-dependent probabilities of dispalying symptoms.
        tau : int or float
            Reduction in transmission for an asymptomatic infectious compared
            to the symptomatic case.
        we : list
            Rates of waning of immunity for current and an older variant.
        omega : int or float
            Change in susceptibility due to the variant.

        """
        if not isinstance(tau, (float, int)):
            raise TypeError('Reduction in transmission must be float or\
                integer.')
        if tau < 0:
            raise ValueError('Reduction in transmission must be => 0.')
        if tau > 1:
            raise ValueError('Reduction in transmission must be <= 1 .')

        if isinstance(d, (float, int)):
            d = [d]
        if np.asarray(d).ndim != 1:
            raise ValueError('The age-dependent probabilities of dispalying\
                symptoms storage format must be 1-dimensional.')
        if (np.asarray(d).shape[0] != self.model._num_ages) and (
                np.asarray(d).shape[0] != 1):
            raise ValueError(
                    'Wrong number of age groups for the age-dependent \
                        probabilities of dispalying symptoms.')
        for _ in d:
            if not isinstance(_, (float, int)):
                raise TypeError('The age-dependent probabilities of dispalying\
                    symptoms must be float or integer.')
            if _ < 0:
                raise ValueError('The age-dependent probabilities of \
                    dispalying symptoms must be => 0.')
            if _ > 1:
                raise ValueError('The age-dependent probabilities of \
                    dispalying symptoms must be <= 1.')

        if isinstance(we, (float, int)):
            we = [we, we, 0]  # if we only model one variant
        if np.asarray(we).ndim != 1:
            raise ValueError('The rates of waning of immunity for current\
                and an older variant storage format must be 1-dimensional.')
        if np.asarray(we).shape[0] != 3:
            raise ValueError(
                    'Wrong number of rates of waning of immunity for\
                    current and an older variant.')
        for _ in we:
            if not isinstance(_, (float, int)):
                raise TypeError('The Rates of waning of immunity for\
                    current and an older variant must be float or integer.')
            if _ < 0:
                raise ValueError('The Rates of waning of immunity for\
                    current and an older variant must be => 0.')

        if not isinstance(omega, (float, int)):
            raise TypeError('Change in susceptibility due to the variant must \
                be float or integer.')
        if omega < 0:
            raise ValueError('Change in susceptibility due to the variant must\
                be => 0.')

    def __call__(self):
        """
        Returns the disease-specific parameters of the
        :class:`WarwickLancSEIRModel` the class relates to.

        Returns
        -------
        List of lists
            List of the disease-specific parameters of the
            :class:`WarwickLancSEIRModel` the class relates to.

        """
        return [self.d, self.tau, self.we, self.omega]

#
# Transmission Class
#


class Transmission(object):
    """Transmission:
    Base class for the transmission-specific parameters of the model: a
    deterministic SEIR used by the Universities of Warwick and Lancaster to
    model the Covid-19 epidemic and the effects of vaccines and waning
    immunity on the epidemic trajectory in different countries.

    Parameters
    ----------
    beta : int or float or list
        Age-dependent susceptibility to infection.
    alpha : int or float
        Rate of progression to infection from exposed.
    gamma : int or float or list
        Age-dependent rate of recovery.

    """
    def __init__(self, model, beta, alpha, gamma):
        super(Transmission, self).__init__()

        # Set model
        if not isinstance(model, wm.WarwickLancSEIRModel):
            raise TypeError(
                'The model must be a Warwick-Lancaster SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(beta, alpha, gamma)

        # Set transmission-specific parameters
        self.alpha = alpha

        if isinstance(beta, (float, int)):
            self.beta = beta * np.ones(self.model._num_ages)
        else:
            self.beta = beta

        if isinstance(gamma, (float, int)):
            self.gamma = gamma * np.ones(self.model._num_ages)
        else:
            self.gamma = gamma

    def _check_parameters_input(self, beta, alpha, gamma):
        """
        Check correct format of the transmission-specific parameters input.

        Parameters
        ----------
        beta : int or float or list
            Age-dependent susceptibility to infection.
        alpha : int or float
            Rate of progression to infection from exposed.
        gamma : int or float or list
            Age-dependent rate of recovery.

        """
        if not isinstance(alpha, (float, int)):
            raise TypeError(
                'The rate of progression to infection from exposed must \
                    be float or integer.')
        if alpha < 0:
            raise ValueError('The rate of progression to infection \
                from exposed must be => 0.')

        if isinstance(beta, (float, int)):
            beta = [beta]
        if np.asarray(beta).ndim != 1:
            raise ValueError('The age-dependent susceptibility to infection\
                storage format must be 1-dimensional.')
        if (np.asarray(beta).shape[0] != self.model._num_ages) and (
                np.asarray(beta).shape[0] != 1):
            raise ValueError(
                    'Wrong number of age groups for the age-dependent \
                        susceptibility to infection.')
        for _ in beta:
            if not isinstance(_, (float, int)):
                raise TypeError('The age-dependent susceptibility to \
                    infection must be float or integer.')
            if _ < 0:
                raise ValueError('The age-dependent susceptibility to \
                    infection must be => 0.')

        if isinstance(gamma, (float, int)):
            gamma = [gamma]
        if np.asarray(gamma).ndim != 1:
            raise ValueError('The age-dependent recovery rate\
                storage format must be 1-dimensional.')
        if (np.asarray(gamma).shape[0] != self.model._num_ages) and (
                np.asarray(gamma).shape[0] != 1):
            raise ValueError(
                    'Wrong number of age groups for the age-dependent \
                        recovery rate.')
        for _ in gamma:
            if not isinstance(_, (float, int)):
                raise TypeError('The age-dependent srecovery rate must \
                    be float or integer.')
            if _ < 0:
                raise ValueError('The age-dependent recovery rate must \
                    be => 0.')

    def __call__(self):
        """
        Returns the transmission-specific parameters of the
        :class:`WarwickLancSEIRModel` the class relates to.

        Returns
        -------
        List of lists
            List of the transmission-specific parameters of the
            :class:`WarwickLancSEIRModel` the class relates to.

        """
        return [self.beta, self.alpha, self.gamma]

#
# SimParameters Class
#


class SimParameters(object):
    """SimParameters:
    Base class for the simulation method's parameters of the model: a
    deterministic SEIR used by the Universities of Warwick and Lancaster to
    model the Covid-19 epidemic and the effects of vaccines and waning
    immunity on the epidemic trajectory in different countries.

    Parameters
    ----------
    method : str
        The type of solver implemented by the simulator.
    times : list
        List of time points at which we wish to evaluate the ODEs
        system.
    eps : boolean
        Indicator parameter for deploying boosters to the recovered
        compartment.

    """
    def __init__(self, model, method, times, eps=False):
        super(SimParameters, self).__init__()

        # Set model
        if not isinstance(model, wm.WarwickLancSEIRModel):
            raise TypeError(
                'The model must be a Warwick-Lancaster SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(method, times, eps)

        # Set other simulation parameters
        self.method = method
        self.times = times
        self.eps = eps

    def _check_parameters_input(self, method, times, eps):
        """
        Check correct format of the simulation method's parameters input.

        Parameters
        ----------
        method: str
            The type of solver implemented by the simulator.
        times : list
            List of time points at which we wish to evaluate the ODEs
            system.
        eps : boolean
            Indicator parameter for deploying boosters to the recovered
            compartment.

        """
        # Check times format
        if not isinstance(times, list):
            raise TypeError('Time points of evaluation must be given in a list\
                format.')
        for _ in times:
            if not isinstance(_, (int, float)):
                raise TypeError('Time points of evaluation must be integer or \
                    float.')
            if _ <= 0:
                raise ValueError('Time points of evaluation must be > 0.')

        if not isinstance(method, str):
            raise TypeError('Simulation method must be a string.')
        if method not in (
                'RK45', 'RK23', 'Radau',
                'BDF', 'LSODA', 'DOP853'):
            raise ValueError('Simulation method not available.')

        if not isinstance(eps, bool):
            raise TypeError('Indicator parameter for deploying boosters to \
                the recovered compartment must be a boolean.')

    def __call__(self):
        """
        Returns the simulation method's parameters of the
        :class:`WarwickLancSEIRModel` the class relates to.

        Returns
        -------
        List
            List of the simulation method's parameters of the
            :class:`WarwickLancSEIRModel` the class relates to.

        """
        return [self.method, self.eps]

#
# SocDistParameters Class
#


class SocDistParameters(object):
    """SocDistParameters:
    Base class for the social distancing parameters of the model: a
    deterministic SEIR used by the Universities of Warwick and Lancaster to
    model the Covid-19 epidemic and the effects of vaccines and waning
    immunity on the epidemic trajectory in different countries.

    Parameters
    ----------
    phi : int or float or list
        Country-specific control factor.

    """
    def __init__(self, model, phi=1):
        super(SocDistParameters, self).__init__()

        # Set model
        if not isinstance(model, wm.WarwickLancSEIRModel):
            raise TypeError(
                'The model must be a Warwick-Lancaster SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(phi)

        # Set social distancing parameters
        if isinstance(phi, (float, int)):
            self.phi = phi * np.ones(len(self.model.regions))
        else:
            self.phi = phi

    def _check_parameters_input(self, phi):
        """
        Check correct format of the social distancing parameters input.

        Parameters
        ----------
        phi : int or float or list
            Country-specific control factor.

        """
        if isinstance(phi, (float, int)):
            phi = [phi]
        if np.asarray(phi).ndim != 1:
            raise ValueError('The country-specific control factor\
                storage format must be 1-dimensional.')
        if (np.asarray(phi).shape[0] != len(self.model.regions)) and (
                np.asarray(phi).shape[0] != 1):
            raise ValueError(
                    'Wrong number of regions for the country-specific control\
                     factor.')
        for _ in phi:
            if not isinstance(_, (float, int)):
                raise TypeError('The country-specific control factor must \
                    be float or integer.')
            if _ < 0:
                raise ValueError('The country-specific control factor \
                    must be => 0.')
            if _ > 1:
                raise ValueError('The country-specific control factor \
                    must be <= 1.')

    def __call__(self):
        """
        Returns the social distancing parameters of the
        :class:`WarwickLancSEIRModel` the class relates to.

        Returns
        -------
        List
            List of the social distancing parameters of the
            :class:`WarwickLancSEIRModel` the class relates to.

        """
        return self.phi


#
# VaccineParameters Class
#


class VaccineParameters(object):
    """VaccineParameters:
    Base class for the vaccination-specific parameters of the model: a
    deterministic SEIR used by the Universities of Warwick and Lancaster to
    model the Covid-19 epidemic and the effects of vaccines and waning
    immunity on the epidemic trajectory in different countries.

    Parameters
    ----------
    vac : int or float or list
        Country-specific vaccination rate of the susceptible population.
    vacb : int or float or list
        Country-specific booster vaccination rate.
    adult : list
        List of the proportions of each age-group that are boosted or
        vaccinated.
    nu_tra : int or float or list
        Vaccine effects on transmission for different vaccination statuses
        (unvaccinated, fully-vaccinated, boosted, partially-waned,
        fully-waned, previous-variant immunity).
    nu_symp : int or float or list
        Vaccine effects on symptom development for different vaccination
        statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
        fully-waned, previous-variant immunity).
    nu_inf : int or float or list
        Vaccine effects on infectiousness for different vaccination
        statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
        fully-waned, previous-variant immunity).
    nu_sev_h : int or float or list
        Vaccine effects on hospitalised severe outcomes for different
        vaccination statuses (unvaccinated, fully-vaccinated, boosted,
        partially-waned, fully-waned, previous-variant immunity).
    nu_sev_d : int or float or list
        Vaccine effects on dead severe outcomes for different vaccination
        statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
        fully-waned, previous-variant immunity).

    """
    def __init__(
            self, model, vac, vacb, adult, nu_tra, nu_symp, nu_inf, nu_sev_h,
            nu_sev_d):
        super(VaccineParameters, self).__init__()

        # Set model
        if not isinstance(model, wm.WarwickLancSEIRModel):
            raise TypeError(
                'The model must be a Warwick-Lancaster SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(
            vac, vacb, adult, nu_tra, nu_symp, nu_inf, nu_sev_h, nu_sev_d)

        # Set vaccination parameters
        self.adult = adult

        if isinstance(vac, (float, int)):
            self.vac = vac * np.ones(len(self.model.regions))
        else:
            self.vac = vac

        if isinstance(vacb, (float, int)):
            self.vacb = vacb * np.ones(len(self.model.regions))
        else:
            self.vacb = vacb

        if isinstance(nu_tra, (float, int)):
            self.nu_tra = nu_tra * np.ones(6)
        else:
            self.nu_tra = nu_tra

        if isinstance(nu_symp, (float, int)):
            self.nu_symp = nu_symp * np.ones(6)
        else:
            self.nu_symp = nu_symp

        if isinstance(nu_inf, (float, int)):
            self.nu_inf = nu_inf * np.ones(6)
        else:
            self.nu_inf = nu_inf

        if isinstance(nu_sev_h, (float, int)):
            self.nu_sev_h = nu_sev_h * np.ones(6)
        else:
            self.nu_sev_h = nu_sev_h

        if isinstance(nu_sev_d, (float, int)):
            self.nu_sev_d = nu_sev_d * np.ones(6)
        else:
            self.nu_sev_d = nu_sev_d

    def _check_parameters_input(
            self, vac, vacb, adult, nu_tra, nu_symp, nu_inf, nu_sev_h,
            nu_sev_d):
        """
        Check correct format of the vaccination-specific parameters input.

        Parameters
        ----------
        vac : int or float or list
            Country-specific vaccination rate of the susceptible population.
        vacb : int or float or list
            Country-specific booster vaccination rate.
        adult : list
            List of the proportions of each age-group that are boosted or
            vaccinated.
        nu_tra : int or float or list
            Vaccine effects on transmission for different vaccination statuses
            (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity).
        nu_symp : int or float or list
            Vaccine effects on symptom development for different vaccination
            statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity).
        nu_inf : int or float or list
            Vaccine effects on infectiousness for different vaccination
            statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity).
        nu_sev_h : int or float or list
            Vaccine effects on hospitalised severe outcomes for different
            vaccination statuses (unvaccinated, fully-vaccinated, boosted,
            partially-waned, fully-waned, previous-variant immunity).
        nu_sev_d : int or float or list
            Vaccine effects on dead severe outcomes for different vaccination
            statuses (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity).

        """
        if isinstance(vac, (float, int)):
            vac = [vac]
        if np.asarray(vac).ndim != 1:
            raise ValueError('The country-specific vaccination rate of the \
                susceptible population storage format must be 1-dimensional.')
        if (np.asarray(vac).shape[0] != len(self.model.regions)) and (
                np.asarray(vac).shape[0] != 1):
            raise ValueError(
                    'Wrong number of regions for the country-specific \
                     vaccination rate of the susceptible population.')
        for _ in vac:
            if not isinstance(_, (float, int)):
                raise TypeError('The country-specific vaccination rate of the \
                    susceptible population must be float or integer.')
            if _ < 0:
                raise ValueError('The country-specific vaccination rate of the\
                    susceptible population must be => 0.')

        if isinstance(vacb, (float, int)):
            vacb = [vacb]
        if np.asarray(vacb).ndim != 1:
            raise ValueError('The country-specific booster vaccination rate \
                storage format must be 1-dimensional.')
        if (np.asarray(vacb).shape[0] != len(self.model.regions)) and (
                np.asarray(vacb).shape[0] != 1):
            raise ValueError(
                    'Wrong number of regions for the country-specific \
                     booster vaccination rate.')
        for _ in vacb:
            if not isinstance(_, (float, int)):
                raise TypeError('The country-specific booster vaccination \
                    rate must be float or integer.')
            if _ < 0:
                raise ValueError('The country-specific booster vaccination \
                    rate must be => 0.')

        if np.asarray(adult).ndim != 1:
            raise ValueError('The list of the proportions of each age-group \
                that are boosted or vaccinated storage format must be \
                1-dimensional.')
        if np.asarray(adult).shape[0] != len(self.model.age_groups):
            raise ValueError(
                    'Wrong number of age-groups for the list of the \
                    proportions of each age-group that are boosted or \
                    vaccinated.')
        for _ in adult:
            if not isinstance(_, (float, int)):
                raise TypeError('The proportions of each age-group that are \
                    boosted or vaccinated must be float or integer.')
            if _ < 0:
                raise ValueError('The proportions of each age-group that are \
                    boosted or vaccinated must be => 0.')
            if _ > 1:
                raise ValueError('The proportions of each age-group that are \
                    boosted or vaccinated must be <= 1.')

        if isinstance(nu_tra, (float, int)):
            nu_tra = [nu_tra]
        if np.asarray(nu_tra).ndim != 1:
            raise ValueError('Vaccine effects on transmission for different \
                vaccination statuses storage format must be 1-dimensional.')
        if (np.asarray(nu_tra).shape[0] != 6) and (
                np.asarray(nu_tra).shape[0] != 1):
            raise ValueError(
                    'Wrong number of regions for the vaccine effects on \
                     transmission for different vaccination statuses.')
        for _ in nu_tra:
            if not isinstance(_, (float, int)):
                raise TypeError('Vaccine effects on transmission for \
                    different vaccination statuses must be float or integer.')
            if _ < 0:
                raise ValueError('Vaccine effects on transmission for \
                    different vaccination statuses must be => 0.')

        if isinstance(nu_symp, (float, int)):
            nu_symp = [nu_symp]
        if np.asarray(nu_symp).ndim != 1:
            raise ValueError('Vaccine effects on symptom development for \
                different vaccination statuses storage format must be \
                1-dimensional.')
        if (np.asarray(nu_symp).shape[0] != 6) and (
                np.asarray(nu_symp).shape[0] != 1):
            raise ValueError(
                    'Wrong number of regions for the vaccine effects on \
                     symptom development for different vaccination statuses.')
        for _ in nu_symp:
            if not isinstance(_, (float, int)):
                raise TypeError('Vaccine effects on symptom development for \
                    different vaccination statuses must be float or integer.')
            if _ < 0:
                raise ValueError('Vaccine effects on symptom development for \
                    different vaccination statuses must be => 0.')

        if isinstance(nu_inf, (float, int)):
            nu_inf = [nu_inf]
        if np.asarray(nu_inf).ndim != 1:
            raise ValueError('Vaccine effects on infectiousness for different \
                vaccination statuses storage format must be 1-dimensional.')
        if (np.asarray(nu_inf).shape[0] != 6) and (
                np.asarray(nu_inf).shape[0] != 1):
            raise ValueError(
                    'Wrong number of regions for the vaccine effects on \
                     infectiousness for different vaccination statuses.')
        for _ in nu_inf:
            if not isinstance(_, (float, int)):
                raise TypeError('Vaccine effects on infectiousness for \
                    different vaccination statuses must be float or integer.')
            if _ < 0:
                raise ValueError('Vaccine effects on infectiousness for \
                    different vaccination statuses must be => 0.')

        if isinstance(nu_sev_h, (float, int)):
            nu_sev_h = [nu_sev_h]
        if np.asarray(nu_sev_h).ndim != 1:
            raise ValueError('Vaccine effects on hospitalised severe outcomes\
                 for different vaccination statuses storage format must be \
                 1-dimensional.')
        if (np.asarray(nu_sev_h).shape[0] != 6) and (
                np.asarray(nu_sev_h).shape[0] != 1):
            raise ValueError(
                    'Wrong number of regions for the vaccine effects on \
                     hospitalised severe outcomes for different vaccination \
                     statuses.')
        for _ in nu_sev_h:
            if not isinstance(_, (float, int)):
                raise TypeError('Vaccine effects on hospitalised severe \
                    outcomes for different vaccination statuses must be float \
                    or integer.')
            if _ < 0:
                raise ValueError('Vaccine effects on hospitalised severe \
                    outcomes for different vaccination statuses must be => 0.')

        if isinstance(nu_sev_d, (float, int)):
            nu_sev_d = [nu_sev_d]
        if np.asarray(nu_sev_d).ndim != 1:
            raise ValueError('Vaccine effects on dead severe outcomes for \
                different vaccination statuses storage format must be \
                1-dimensional.')
        if (np.asarray(nu_sev_d).shape[0] != 6) and (
                np.asarray(nu_sev_d).shape[0] != 1):
            raise ValueError(
                    'Wrong number of regions for the vaccine effects on \
                     dead severe outcomes for different vaccination statuses.')
        for _ in nu_sev_d:
            if not isinstance(_, (float, int)):
                raise TypeError('Vaccine effects on dead severe outcomes for \
                    different vaccination statuses must be float or integer.')
            if _ < 0:
                raise ValueError('Vaccine effects on dead severe outcomes for \
                    different vaccination statuses must be => 0.')

    def __call__(self):
        """
        Returns the vaccine-specific parameters of the
        :class:`WarwickLancSEIRModel` the class relates to.

        Returns
        -------
        List
            List of the vaccine-specific parameters of the
            :class:`WarwickLancSEIRModel` the class relates to.

        """
        return [
            self.vac, self.vacb, self.adult, self.nu_tra, self.nu_symp,
            self.nu_inf, self.nu_sev_h, self.nu_sev_d]

#
# ParametersController Class
#


class ParametersController(object):
    """ParametersController Class:
    Base class for the paramaters of the model: a
    deterministic SEIR used by the Universities of Warwick and Lancaster to
    model the Covid-19 epidemic and the effects of vaccines and waning
    immunity on the epidemic trajectory in different countries.

    In order to simulate using the Warwick model, the following parameters are
    required, which are stored as part of this class.

    Parameters
    ----------
    model : WarwickLancSEIRModel
        The model whose parameters are stored.
    regional_parameters : RegParameters
        Class of the regional and time dependent parameters used in the
        simulation of the model.
    ICs_parameters : ICs
        Class of the Ics used in the simulation of the model.
    disease_parameters : DiseaseParameters
        Class of the disease-specific parameters used in the simulation of
        the model.
    transmission_parameters : Transmission
        Class of the rates of progression parameters used in the simulation of
        the model.
    simulation_parameters : SimParameters
        Class of the simulation method's parameters used in the simulation of
        the model.
    vaccine_parameters : VaccineParameters
        Class of the vaccine-specific parameters used in the simulation of
        the model.
    soc_dist_parameters : SocDistParameters
        Class of the social distancing parameters used in the simulation of
        the model.

    """
    def __init__(
            self, model, regional_parameters, ICs_parameters,
            disease_parameters, transmission_parameters, simulation_parameters,
            vaccine_parameters, soc_dist_parameters=None):
        # Instantiate class
        super(ParametersController, self).__init__()

        # Set model
        if not isinstance(model, wm.WarwickLancSEIRModel):
            raise TypeError(
                'The model must be a Warwick-Lancaster SEIR Model.')

        self.model = model

        # Check inputs format
        self._check_parameters_input(
            regional_parameters, ICs_parameters, disease_parameters,
            transmission_parameters, simulation_parameters,
            vaccine_parameters, soc_dist_parameters)

        # Set regional and time dependent parameters
        self.regional_parameters = regional_parameters

        # Set ICs parameters
        self.ICs = ICs_parameters

        # Set disease-specific parameters
        self.disease_parameters = disease_parameters

        # Set transmission-specific parameters
        self.transmission_parameters = transmission_parameters

        # Set other simulation parameters
        self.simulation_parameters = simulation_parameters

        # Set vaccine-specific parameters
        self.vaccine_parameters = vaccine_parameters

        # Set social distancing parameters
        if soc_dist_parameters is not None:
            self.soc_dist_parameters = soc_dist_parameters
        else:
            self.soc_dist_parameters = SocDistParameters(model)

    def _check_parameters_input(
            self, regional_parameters, ICs_parameters, disease_parameters,
            transmission_parameters, simulation_parameters,
            vaccine_parameters, soc_dist_parameters):
        """
        Check correct format of input of simulate method.

        Parameters
        ----------
        model : WarwickLancSEIRModel
            The model whose parameters are stored.
        regional_parameters : RegParameters
            Class of the regional and time dependent parameters used in the
            simulation of the model.
        ICs_parameters : ICs
            Class of the Ics used in the simulation of the model.
        disease_parameters : DiseaseParameters
            Class of the disease-specific parameters used in the simulation of
            the model.
        transmission_parameters : Transmission
            Class of the rates of progression parameters used in the
            simulation of the model.
        accine_parameters : VaccineParameters
            Class of the vaccine-specific parameters used in the simulation of
            the model.
        simulation_parameters : SimParameters
            Class of the simulation method's parameters used in the
            simulation of the model.

        """
        if not isinstance(regional_parameters, RegParameters):
            raise TypeError('The model`s regional and time dependent\
                parameters must be of a Warwick-Lancaster SEIR Model.')
        if regional_parameters.model != self.model:
            raise ValueError('The regional and time dependent parameters do \
                not correspond to the right model.')

        if not isinstance(ICs_parameters, ICs):
            raise TypeError('The model`s ICs parameters must be of a Warwick\
                -Lancaster SEIR Model.')
        if ICs_parameters.model != self.model:
            raise ValueError('ICs do not correspond to the right model.')

        if not isinstance(disease_parameters, DiseaseParameters):
            raise TypeError('The model`s disease-specific parameters must be \
                of a Warwick-Lancaster SEIR Model.')
        if disease_parameters.model != self.model:
            raise ValueError('The disease-specific parameters do not \
            correspond to the right model.')

        if not isinstance(transmission_parameters, Transmission):
            raise TypeError('The model`s transmission-specific parameters must\
                be a of a Warwick-Lancaster SEIRD Model.')
        if transmission_parameters.model != self.model:
            raise ValueError('The transmission-specific parameters do not \
                correspond to the right model.')

        if not isinstance(simulation_parameters, SimParameters):
            raise TypeError('The model`s simulation method`s parameters must\
                be of a Warwick-Lancaster SEIR Model.')
        if simulation_parameters.model != self.model:
            raise ValueError('The simulation method`s parameters do not \
                correspond to the right model.')

        if not isinstance(vaccine_parameters, VaccineParameters):
            raise TypeError('The model`s vaccine-specific parameters must\
                be a of a Warwick-Lancaster SEIRD Model.')
        if vaccine_parameters.model != self.model:
            raise ValueError('The vaccine-specific parameters do not \
                correspond to the right model.')

        if soc_dist_parameters is not None:
            if not isinstance(soc_dist_parameters, SocDistParameters):
                raise TypeError('The model`s social distancing parameters must\
                    be of a Warwick-Lancaster SEIR Model.')
            if soc_dist_parameters.model != self.model:
                raise ValueError('The simulation method`s parameters do not \
                    correspond to the right model.')

    def __call__(self):
        """
        Returns the list of all the parameters used for the simulation of the
        Warwick model in their order, which will be then separated within the
        :class:`WarwickLancSEIRModel` class.

        Returns
        -------
        list
            List of all the parameters used for the simulation of the
            Warwick model in their order.

        """
        parameters = []

        # Add the regional and time dependent parameters
        parameters.append(self.regional_parameters())

        # Add ICs
        parameters.extend(self.ICs())

        # Add transmission-specific parameters
        parameters.extend(self.transmission_parameters())

        # Add disease-specific
        parameters.extend(self.disease_parameters())

        # Add other simulation parameters
        parameters.extend(self.simulation_parameters())

        return list(deepflatten(parameters, ignore=str))
