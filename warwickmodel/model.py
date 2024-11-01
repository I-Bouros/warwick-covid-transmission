#
# WarwickLancSEIRModel Class
#
# This file is part of WARWICKMODEL
# (https://github.com/I-Bouros/warwick-covid-transmission.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""
This script contains code for modelling the extended SEIR model created by
Universities of Warwick and Lancaster. This model is used to produce a number
of research reports for SAGE Working Group on COVID-19.

It uses an extended version of an SEIR model and contact and region specific
matrices.

"""

from itertools import chain

import numpy as np
import pints
from scipy.stats import nbinom, binom
from scipy.integrate import solve_ivp

import epimodels as em


class WarwickLancSEIRModel(pints.ForwardModel):
    r"""WarwickLancSEIRModel Class:
    Base class for constructing the ODE model: deterministic SEIR developed by
    Universities of Warwick and Lancaster to model the Covid-19 epidemic and
    the effects of vaccines and waning immunity on the epidemic trajectory
    in different countries.

    The population is structured such that every individual will belong to one
    of the compartments of the extended SEIRD model.

    The general SEIR Model has four compartments - susceptible individuals
    (:math:`S`), exposed but not yet infectious (:math:`E`), infectious
    (:math:`I`) and recovered (:math:`R`).

    In the Warwick-Lancaster model framework, the exposed are split into 5x6
    compartments to allow waiting time as an exposed to be Erlang-distributed,
    as well as depending on the immunity status due to the vaccine of the
    individual exposed. Meanwhile, the infectious compartment is split into 10
    distinct ones: depending on whether they are symptomatic or asymptomatic
    infectious, and on the immunity status due to the vaccine of the
    individual infected. We also consider a population divided in age groups,
    as we expect people of different ages to interact diferently between
    themselves and to be affected differently by the virus, i.e. have
    different succeptibilities to infection and proportions of
    asymptomatic individuals. The model structure now
    becomes, for each region:

    .. math::
       :nowrap:

        \begin{eqnarray}
            \frac{dS^i}{dt} &=& - \phi \omega \nu_\text{tra} \beta^i
                \lambda^i S^i - \text{Vac} S^i -\text{VacB} S^i +
                \text{WE2} S_{W2}^i + \text{WE} S_{W3}^i \\
            \frac{dS_F^i}{dt} &=& - \phi \omega \nu_\text{tra,F} \beta^i
                \lambda^i S_F^i + \text{Vac} S^i + \text{Vac} S_{W1}^i +
                \text{Vac} S_{W2}^i + \text{Vac} S_{W3}^i -
                \text{VacB} S_F^i - \text{WE} S_F^i \\
            \frac{dS_B^i}{dt} &=& - \phi \omega \nu_\text{tra,B}
                \beta^i \lambda^i S_B^i + \text{VacB} S_F^i + \text{VacB}
                S_{W1}^i + \text{VacB} S_{W2}^i + \text{VacB} S_{W3}^i -
                \text{WE} S_B^i + \epsilon \text{VacB} R^i\\
            \frac{dS_{W1}^i}{dt} &=& - \phi \omega \nu_\text{tra,W1}
                \beta^i \lambda^i S_{W1}^i - \text{Vac} S_{W1}^i -
                \text{VacB} S_{W1}^i - \text{WE} S_{W1}^i + \text{WE} S_B^i +
                \text{WE} R^i \\
            \frac{dS_{W2}^i}{dt} &=& - \phi \omega \nu_\text{tra,W2}
                \beta^i \lambda^i S_{W2}^i - \text{Vac} S_{W2}^i -
                \text{VacB} S_{W2}^i - \text{WE2} S_{W2}^i + \text{WE} S_F^i +
                \text{WE} S_{W1}^i - \text{WE3} S_{W2}^i \\
            \frac{dS_{W3}^i}{dt} &=& - \phi \omega \nu_\text{tra,W3}
                \beta^i \lambda^i S_{W3}^i - \text{Vac} S_{W3}^i -
                \text{VacB} S_{W3}^i - \text{WE} S_{W3}^i +
                \text{WE3} S_{W2}^i \\
            \frac{dE_1^i}{dt} &=& \phi \omega \nu_\text{tra} \beta^i
                \lambda^i S^i - \alpha E_1^i \\
            \frac{dE_2^i}{dt} &=& \alpha (E_1^i - E_2^i) \\
            \frac{dE_3^i}{dt} &=& \alpha (E_2^i - E_3^i) \\
            \frac{dE_4^i}{dt} &=& \alpha (E_3^i - E_4^i) \\
            \frac{dE_5^i}{dt} &=& \alpha (E_4^i - E_5^i) \\
            \frac{dE_{1,F}^i}{dt} &=& \phi \omega \nu_\text{tra,F}
                \beta^i \lambda^i S_F^i - \alpha E_{1,F}^i \\
            \frac{dE_{2,F}^i}{dt} &=& \alpha (E_{1,F}^i -
                E_{2,F}^i) \\
            \frac{dE_{3,F}^i}{dt} &=& \alpha (E_{2,F}^i -
                E_{3,F}^i) \\
            \frac{dE_{4,F}^i}{dt} &=& \alpha (E_{3,F}^i -
                E_{4,F}^i) \\
            \frac{dE_{5,F}^i}{dt} &=& \alpha (E_{4,F}^i -
                E_{5,F}^i) \\
            \frac{dE_{1,B}^i}{dt} &=& \phi \omega \nu_\text{tra,B}
                \beta^i \lambda^i S_B^i - \alpha E_{1,B}^i \\
            \frac{dE_{2,B}^i}{dt} &=& \alpha (E_{1,B}^i -
                E_{2,B}^i) \\
            \frac{dE_{3,B}^i}{dt} &=& \alpha (E_{2,B}^i -
                E_{3,B}^i) \\
            \frac{dE_{4,B}^i}{dt} &=& \alpha (E_{3,B}^i -
                E_{4,B}^i) \\
            \frac{dE_{5,B}^i}{dt} &=& \alpha (E_{4,B}^i -
                E_{5,B}^i) \\
            \frac{dE_{1,W1}^i}{dt} &=& \phi \omega \nu_\text{tra,W1}
                \beta^i \lambda^i S_{W1}^i - \alpha E_{1,W1}^i \\
            \frac{dE_{2,W1}^i}{dt} &=& \alpha (E_{1,W1}^i -
                E_{2,W1}^i) \\
            \frac{dE_{3,W1}^i}{dt} &=& \alpha (E_{2,W1}^i -
                E_{3,W1}^i) \\
            \frac{dE_{4,W1}^i}{dt} &=& \alpha (E_{3,W1}^i -
                E_{4,W1}^i) \\
            \frac{dE_{5,W1}^i}{dt} &=& \alpha (E_{4,W1}^i -
                E_{5,W1}^i) \\
            \frac{dE_{1,W2}^i}{dt} &=& \phi \omega \nu_\text{tra,W2}
                \beta^i \lambda^i S_{W2}^i - \alpha E_{1,W2}^i \\
            \frac{dE_{2,W2}^i}{dt} &=& \alpha (E_{1,W2}^i -
                E_{2,W2}^i) \\
            \frac{dE_{3,W2}^i}{dt} &=& \alpha (E_{2,W2}^i -
                E_{3,W2}^i) \\
            \frac{dE_{4,W2}^i}{dt} &=& \alpha (E_{3,W2}^i -
                E_{4,W2}^i) \\
            \frac{dE_{5,W2}^i}{dt} &=& \alpha (E_{4,W2}^i -
                E_{5,W2}^i) \\
            \frac{dE_{1,W3}^i}{dt} &=& \phi \omega \nu_\text{tra,W3}
                \beta^i \lambda^i S_{W3}^i - \alpha E_{1,W3}^i \\
            \frac{dE_{2,W3}^i}{dt} &=& \alpha (E_{1,W3}^i -
                E_{2,W3}^i) \\
            \frac{dE_{3,W3}^i}{dt} &=& \alpha (E_{2,W3}^i -
                E_{3,W3}^i) \\
            \frac{dE_{4,W3}^i}{dt} &=& \alpha (E_{3,W3}^i -
                E_{4,W3}^i) \\
            \frac{dE_{5,W3}^i}{dt} &=& \alpha (E_{4,W3}^i -
                E_{5,W3}^i) \\
            \frac{dI^i}{dt} &=& \nu_\text{symp}d^i \alpha
                E_5^i - \gamma^i I^i \\
            \frac{dI_F^i}{dt} &=& \nu_\text{symp,F}d^i \alpha
                E_{5,F}^i - \gamma^i I_F^i \\
            \frac{dI_B^i}{dt} &=& \nu_\text{symp,B}d^i \alpha
                E_{5,B}^i - \gamma^i I_B^i \\
            \frac{dI_{W1}^i}{dt} &=& \nu_\text{symp,W1}d^i
                \alpha E_{5,W1}^i - \gamma^i I_{W1}^i \\
            \frac{dI_{W2}^i}{dt} &=& \nu_\text{symp,W2}d^i
                \alpha E_{5,W2}^i - \gamma^i I_{W2}^i \\
            \frac{dI_{W3}^i}{dt} &=& \nu_\text{symp,W3}d^i
                \alpha E_{5,W3}^i - \gamma^i I_{W3}^i \\
            \frac{dA^i}{dt} &=& (1-\nu_\text{symp}d^i) \alpha E_5^i
                - \gamma^i A^i \\
            \frac{dA_F^i}{dt} &=& (1-\nu_\text{symp,F}d^i) \alpha
                E_{5,F}^i - \gamma^i A_F^i \\
            \frac{dA_B^i}{dt} &=& (1-\nu_\text{symp,B}d^i) \alpha
                E_{5,B}^i - \gamma^i A_B^i \\
            \frac{dA_{W1}^i}{dt} &=& (1-\nu_\text{symp,W1}d^i)
                \alpha E_{5,W1}^i - \gamma^i A_{W1}^i \\
            \frac{dA_{W2}^i}{dt} &=& (1-\nu_\text{symp,W2}d^i)
                \alpha E_{5,W2}^i - \gamma^i A_{W2}^i \\
            \frac{dA_{W3}^i}{dt} &=& (1-\nu_\text{symp,W3}d^i)
                \alpha E_{5,W3}^i - \gamma^i A_{W3}^i \\
            \frac{dR^i}{dt} &=& \gamma^i \Big(I^i + A^i\Big) +
                \gamma^i \Big(I_F^i + A_F^i\Big) +
                \gamma^i \Big(I_B^i + A_B^i\Big) +
                \gamma^i \Big(I_{W1}^i + A_{W1}^i\Big) +
                \gamma^i \Big(I_{W2}^i + A_{W2}^i\Big) +
                \gamma^i \Big(I_{W3}^i + A_{W3}^i\Big) -
                \text{WE} R^i - \epsilon \text{VacB} R^i
        \end{eqnarray}

    with:

    .. math::
       :nowrap:

        \begin{eqnarray}
            \lambda^i = \sum_{j} \nu_\text{inf}C^{ij} \frac{1}{N^j}
                \Big(I^j + \tau^j A^j\Big) +
                    \sum_{j} \nu_\text{inf,F} C^{ij} \frac{1}{N^j}
                \Big(I_F^j + \tau^j A_F^j\Big) +
                    \sum_{j} \nu_\text{inf,B} C^{ij} \frac{1}{N^j}
                \Big(I_B^j + \tau^j A_B^j\Big) +
                    \sum_{j} \nu_\text{inf,W1} C^{ij} \frac{1}{N^j}
                \Big(I_{W1}^j + \tau^j A_{W1}^j\Big) +
                    \sum_{j} \nu_\text{inf,W2} C^{ij} \frac{1}{N^j}
                \Big(I_{W2}^j + \tau^j A_{W2}^j\Big) +
                    \sum_{j} \nu_\text{inf,W3} C^{ij} \frac{1}{N^j}
                \Big(I_{W3}^j + \tau^j A_{W3}^j\Big)
        \end{eqnarray}

    where :math:`i` is the age group of the individual, :math:`C^{ij}` is the
    :math:`(i,j)` th element of the regional contact matrix, and represents
    the expected number of new infections in age group :math:`i` caused by an
    infectious in age group :math:`j`.

    :math:`\beta` is the age-dependent transmission parameter

    :math:`S(0) = S_0, E(0) = E_0, I(0) = I_0, A(0) = A_0, R(0) = R_0` are also
    parameters of the model (evaluation at 0 refers to the compartments'
    structure at intial time; we use S, E, I, A as blanket terms for all the
    different types of compartments that fall under these labels).

    Extends :class:`pints.ForwardModel`.

    """
    def __init__(self):
        super(WarwickLancSEIRModel, self).__init__()

        # Asbetan default values
        self._output_names = [
            'S', 'Sf', 'Sb', 'Sw1', 'Sw2', 'Sw3', 'E1', 'E2', 'E3', 'E4', 'E5',
            'E1f', 'E2f', 'E3f', 'E4f', 'E5f', 'E1b', 'E2b', 'E3b', 'E4b',
            'E5b', 'E1w1', 'E2w1', 'E3w1', 'E4w1', 'E5w1', 'E1w2', 'E2w2',
            'E3w2', 'E4w2', 'E5w2', 'E1w3', 'E2w3', 'E3w3', 'E4w3', 'E5w3',
            'I', 'If', 'Ib', 'Iw1', 'Iw2', 'Iw3', 'A', 'Af', 'Ab', 'Aw1',
            'Aw2', 'Aw3', 'R', 'Incidence']
        self._parameter_names = [
            'S0', 'Sf0', 'Sb0', 'Sw10', 'Sw20', 'Sw30', 'E10', 'E20', 'E30',
            'E40', 'E50', 'E1f0', 'E2f0', 'E3f0', 'E4f0', 'E5f0', 'E1b0',
            'E2b0', 'E3b0', 'E4b0', 'E5b0', 'E1w10', 'E2w10', 'E3w10', 'E4w10',
            'E5w10', 'E1w20', 'E2w20', 'E3w20', 'E4w20', 'E5w20', 'E1w30',
            'E2w30', 'E3w30', 'E4w30', 'E5w30', 'I0', 'If0', 'Ib0', 'Iw10',
            'Iw20', 'Iw30', 'A0', 'Af0', 'Ab0', 'Aw10', 'Aw20', 'Aw30', 'R0',
            'beta', 'alpha', 'gamma', 'd', 'tau', 'we', 'omega']

        # The default number of outputs is 50,
        # i.e. S, Sf, Sb, Sw1, Sw2, Sw3, E1, ..., E5, E1f, ..., E5f, E1b, ...
        # E5b, E1w1, ... E5w1, E1w2, Aw2, Aw3, R and Incidence
        self._n_outputs = len(self._output_names)

        # The default number of parameters is 56,
        # i.e. 49 initial conditions and 7 parameters
        self._n_parameters = len(self._parameter_names)

        self._output_indices = np.arange(self._n_outputs)

    def n_outputs(self):
        """
        Returns the number of outputs.

        Returns
        -------
        int
            Number of outputs.

        """
        return self._n_outputs

    def n_parameters(self):
        """
        Returns the number of parameters.

        Returns
        -------
        int
            Number of parameters.

        """
        return self._n_parameters

    def output_names(self):
        """
        Returns the (selected) output names.

        Returns
        -------
        list
            List of the (selected) output names.

        """
        names = [self._output_names[x] for x in self._output_indices]
        return names

    def parameter_names(self):
        """
        Returns the parameter names.

        Returns
        -------
        list
            List of the parameter names.

        """
        return self._parameter_names

    def set_regions(self, regions):
        """
        Sets region names.

        Parameters
        ----------
        regions : list
            List of region names considered by the model.

        """
        self.regions = regions

    def set_age_groups(self, age_groups):
        """
        Sets age group names and counts their number.

        Parameters
        ----------
        age_groups : list
            List of age group names considered by the model.

        """
        self.age_groups = age_groups
        self._num_ages = len(self.age_groups)

    def region_names(self):
        """
        Returns the regions names.

        Returns
        -------
        list
            List of the regions names.

        """
        return self.regions

    def age_groups_names(self):
        """
        Returns the age group names.

        Returns
        -------
        list
            List of the age group names.

        """
        return self.age_groups

    def set_outputs(self, outputs):
        """
        Checks existence of outputs and selects only those remaining.

        Parameters
        ----------
        outputs : list
            List of output names that are selected.

        """
        for output in outputs:
            if output not in self._output_names:
                raise ValueError(
                    'The output names specified must be in correct forms')

        output_indices = []
        for output_id, output in enumerate(self._output_names):
            if output in outputs:
                output_indices.append(output_id)

        # Remember outputs
        self._output_indices = output_indices
        self._n_outputs = len(outputs)

    def _right_hand_side(self, t, r, y, c, num_a_groups):
        r"""
        Constructs the RHS of the equations of the system of ODEs for given a
        region and time point.

        Parameters
        ----------
        t : float
            Time point at which we compute the evaluation.
        r : int
            The index of the region to which the current instance of the ODEs
            system refers.
        y : numpy.array
            Array of all the compartments of the ODE system, segregated
            by age-group. It assumes y = [S, Sf, Sb, Sw1, Sw2, Sw3, E1, ...,
            E5, E1f, ..., E5f, E1b, ... E5b, E1w1, ... E5w1, E1w2, ... E5w2,
            E1w3, ... E5w3, I, If, Ib, Iw1, Iw2, Iw3, A, Af, Ab, Aw1, Aw2, Aw3,
            R] where each letter actually refers to all compartment of that
            type. (e.g. S refers to the compartments of all ages of
            non-vaccinated susceptibles).
        c : list
            List of values used to compute the parameters of the ODEs
            system. It assumes c = [beta, alpha, gamma, d, tau, we, omega],
            where :math:`beta` represents the age-dependent susceptibility of
            individuals to infection, :math:`alpha` is the rate of progression
            to infectious disease, :math:`gamma` is the recovery rate,
            :math:`d` represents the age-dependent probability of displaying
            symptoms, :math:`tau` is the reduction in the transmission rate of
            infection for asymptomatic individuals, :math:`we` are the rates
            of waning of immunity :math:`omega` is and the change in
            susceptibility due to the variant.
        num_a_groups : int
            Number of age groups in which the population is split. It
            refers to the number of compartments of each type.

        Returns
        -------
        numpy.array
            Age-strictured matrix representation of the RHS of the ODEs system.

        """
        # Read in the number of age-groups
        n = num_a_groups

        # Split compartments into their types
        # S, Sf, Sb, Sw1, Sw2, Sw3
        s, sF, sB, sW1, sW2, sW3 = (
            y[:n], y[n:(2*n)], y[(2*n):(3*n)],
            y[(3*n):(4*n)], y[(4*n):(5*n)], y[(5*n):(6*n)])

        # E1, ..., E5
        e1, e2, e3, e4, e5 = (
            y[(6*n):(7*n)], y[(7*n):(8*n)], y[(8*n):(9*n)],
            y[(9*n):(10*n)], y[(10*n):(11*n)])

        # E1f, ..., E5f
        e1F, e2F, e3F, e4F, e5F = (
            y[(11*n):(12*n)], y[(12*n):(13*n)], y[(13*n):(14*n)],
            y[(14*n):(15*n)], y[(15*n):(16*n)])

        # E1b, ... E5b
        e1B, e2B, e3B, e4B, e5B = (
            y[(16*n):(17*n)], y[(17*n):(18*n)], y[(18*n):(19*n)],
            y[(19*n):(20*n)], y[(20*n):(21*n)])

        # E1w1, ... E5w1
        e1W1, e2W1, e3W1, e4W1, e5W1 = (
            y[(21*n):(22*n)], y[(22*n):(23*n)], y[(23*n):(24*n)],
            y[(24*n):(25*n)], y[(25*n):(26*n)])

        # E1w2, ... E5w2
        e1W2, e2W2, e3W2, e4W2, e5W2 = (
            y[(26*n):(27*n)], y[(27*n):(28*n)], y[(28*n):(29*n)],
            y[(29*n):(30*n)], y[(30*n):(31*n)])

        # E1w3, ... E5w3
        e1W3, e2W3, e3W3, e4W3, e5W3 = (
            y[(31*n):(32*n)], y[(32*n):(33*n)], y[(33*n):(34*n)],
            y[(34*n):(35*n)], y[(35*n):(36*n)])

        # If, Ib, Iw1, Iw2, Iw3
        i, iF, iB, iW1, iW2, iW3 = (
            y[(36*n):(37*n)], y[(37*n):(38*n)], y[(38*n):(39*n)],
            y[(39*n):(40*n)], y[(40*n):(41*n)], y[(41*n):(42*n)])

        # A, Af, Ab, Aw1, Aw2, R
        a, aF, aB, aW1, aW2, aW3, _ = (
            y[(42*n):(43*n)], y[(43*n):(44*n)], y[(44*n):(45*n)],
            y[(45*n):(46*n)], y[(46*n):(47*n)], y[(47*n):(48*n)],
            y[(48*n):])

        # Read the social distancing parameters of the system
        phi_all = self.social_distancing_param
        phi = phi_all[r-1]

        # Read the vaccination parameters of the system
        vac_all, vacb_all, adult, nu_tra, nu_symp, nu_inf = \
            self.vaccine_param[:6]

        vac, vacb = vac_all[r-1], vacb_all[r-1]

        # Read parameters of the system
        beta, alpha, gamma, d, tau, we, omega = c

        # waning rates of the current & older variant respectively
        we1, we2, we3 = we

        # Identify the appropriate contact matrix for the ODE system
        cont_mat = \
            self.contacts_timeline.identify_current_contacts(r, t)

        # Write actual RHS
        lam = nu_tra[0] * np.multiply(beta, np.dot(
            cont_mat, np.multiply(
                np.asarray(i) + tau * np.asarray(a), (1 / self._N[r-1]))))
        lam += nu_tra[1] * np.multiply(beta, np.dot(
            cont_mat, np.multiply(
                np.asarray(iF) + tau * np.asarray(aF), (1 / self._N[r-1]))))
        lam += nu_tra[2] * np.multiply(beta, np.dot(
            cont_mat, np.multiply(
                np.asarray(iB) + tau * np.asarray(aB), (1 / self._N[r-1]))))
        lam += nu_tra[3] * np.multiply(beta, np.dot(
            cont_mat, np.multiply(
                np.asarray(iW1) + tau * np.asarray(aW1), (1 / self._N[r-1]))))
        lam += nu_tra[4] * np.multiply(beta, np.dot(
            cont_mat, np.multiply(
                np.asarray(iW2) + tau * np.asarray(aW2), (1 / self._N[r-1]))))
        lam += nu_tra[5] * np.multiply(beta, np.dot(
            cont_mat, np.multiply(
                np.asarray(iW3) + tau * np.asarray(aW3), (1 / self._N[r-1]))))

        lam_times_s = omega * phi * nu_inf[0] * np.multiply(s, lam)

        lam_times_sF = omega * phi * nu_inf[1] * np.multiply(sF, lam)

        lam_times_sB = omega * phi * nu_inf[2] * np.multiply(sB, lam)

        lam_times_sW1 = omega * phi * nu_inf[3] * np.multiply(sW1, lam)

        lam_times_sW2 = omega * phi * nu_inf[4] * np.multiply(sW2, lam)

        lam_times_sW3 = omega * phi * nu_inf[5] * np.multiply(sW3, lam)

        dydt = np.concatenate((
            -lam_times_s - vac * np.multiply(adult, s) - vacb * np.multiply(
                adult, s) + we2 * np.asarray(sW2) + 0 * np.asarray(sW3),
            -lam_times_sF + vac * np.multiply(adult, s) + vac * np.multiply(
                adult, sW1) + vac * np.multiply(
                adult, sW2) + vac * np.multiply(adult, sW3) - we1 * np.asarray(
                sF) - vacb * np.multiply(adult, sF),
            -lam_times_sB + vacb * np.multiply(adult, s) + vacb * np.multiply(
                adult, sF) + vacb * np.multiply(
                adult, sW1) + vacb * np.multiply(
                adult, sW2) + self._eps * vacb * np.multiply(
                adult, _) - we1 * np.asarray(sB),
            -lam_times_sW1 - vac * np.multiply(
                adult, sW1) - vacb * np.multiply(
                adult, sW1) - we1 * np.asarray(sW1) + we1 * np.array(
                sB) + we1 * np.array(_),
            -lam_times_sW2 - vac * np.multiply(
                adult, sW2) - vacb * np.multiply(
                adult, sW2) + we1 * np.asarray(sF) + we1 * np.asarray(
                sW1) - we2 * np.asarray(sW2) - we3 * np.asarray(sW2),
            -lam_times_sW3 - vac * np.multiply(
                adult, sW3) - vacb * np.multiply(
                adult, sW3) + we3 * np.asarray(sW2) - 0 * np.asarray(sW3),
            lam_times_s - alpha * np.asarray(e1),
            alpha * (np.asarray(e1) - np.asarray(e2)),
            alpha * (np.asarray(e2) - np.asarray(e3)),
            alpha * (np.asarray(e3) - np.asarray(e4)),
            alpha * (np.asarray(e4) - np.asarray(e5)),
            lam_times_sF - alpha * np.asarray(e1F),
            alpha * (np.asarray(e1F) - np.asarray(e2F)),
            alpha * (np.asarray(e2F) - np.asarray(e3F)),
            alpha * (np.asarray(e3F) - np.asarray(e4F)),
            alpha * (np.asarray(e4F) - np.asarray(e5F)),
            lam_times_sB - alpha * np.asarray(e1B),
            alpha * (np.asarray(e1B) - np.asarray(e2B)),
            alpha * (np.asarray(e2B) - np.asarray(e3B)),
            alpha * (np.asarray(e3B) - np.asarray(e4B)),
            alpha * (np.asarray(e4B) - np.asarray(e5B)),
            lam_times_sW1 - alpha * np.asarray(e1W1),
            alpha * (np.asarray(e1W1) - np.asarray(e2W1)),
            alpha * (np.asarray(e2W1) - np.asarray(e3W1)),
            alpha * (np.asarray(e3W1) - np.asarray(e4W1)),
            alpha * (np.asarray(e4W1) - np.asarray(e5W1)),
            lam_times_sW2 - alpha * np.asarray(e1W2),
            alpha * (np.asarray(e1W2) - np.asarray(e2W2)),
            alpha * (np.asarray(e2W2) - np.asarray(e3W2)),
            alpha * (np.asarray(e3W2) - np.asarray(e4W2)),
            alpha * (np.asarray(e4W2) - np.asarray(e5W2)),
            lam_times_sW3 - alpha * np.asarray(e1W3),
            alpha * (np.asarray(e1W3) - np.asarray(e2W3)),
            alpha * (np.asarray(e2W3) - np.asarray(e3W3)),
            alpha * (np.asarray(e3W3) - np.asarray(e4W3)),
            alpha * (np.asarray(e4W3) - np.asarray(e5W3)),
            alpha * np.multiply(
                nu_symp[0] * np.array(d), e5) - np.multiply(gamma, i),
            alpha * np.multiply(
                nu_symp[1] * np.array(d), e5F) - np.multiply(gamma, iF),
            alpha * np.multiply(
                nu_symp[2] * np.array(d), e5B) - np.multiply(gamma, iB),
            alpha * np.multiply(
                nu_symp[3] * np.array(d), e5W1) - np.multiply(gamma, iW1),
            alpha * np.multiply(
                nu_symp[4] * np.array(d), e5W2) - np.multiply(gamma, iW2),
            alpha * np.multiply(
                nu_symp[5] * np.array(d), e5W3) - np.multiply(gamma, iW3),
            alpha * np.multiply(
                1 - nu_symp[0] * np.array(d), e5) - np.multiply(gamma, a),
            alpha * np.multiply(
                1 - nu_symp[1] * np.array(d), e5F) - np.multiply(gamma, aF),
            alpha * np.multiply(
                1 - nu_symp[2] * np.array(d), e5B) - np.multiply(gamma, aB),
            alpha * np.multiply(
                1 - nu_symp[3] * np.array(d), e5W1) - np.multiply(gamma, aW1),
            alpha * np.multiply(
                1 - nu_symp[4] * np.array(d), e5W2) - np.multiply(gamma, aW2),
            alpha * np.multiply(
                1 - nu_symp[5] * np.array(d), e5W3) - np.multiply(gamma, aW3),
            np.multiply(
                gamma,
                np.asarray(i) + np.asarray(a) + np.asarray(iF) +
                np.asarray(aF) + np.asarray(iB) + np.asarray(aB) +
                np.asarray(iW1) + np.asarray(aW1) + np.asarray(iW2) +
                np.asarray(aW2) + np.asarray(iW3) + np.asarray(aW3)
                ) - we1 * np.asarray(_) - self._eps * vacb * np.multiply(
                adult, _)
            ))

        return dydt

    def _scipy_solver(self, times, num_a_groups, method):
        """
        Computes the values in each compartment of the Warwick ODEs system
        using the 'off-the-shelf' solver of the IVP from :module:`scipy`.

        Parameters
        ----------
        times : list
            List of time points at which we wish to evaluate the ODEs system.
        num_a_groups : int
            Number of age groups in which the population is split. It
            refers to the number of compartments of each type.
        method : str
            The type of solver implemented by the :meth:`scipy.solve_ivp`.

        Returns
        -------
        dict
            Solution of the ODE system at the time points provided.

        """
        # Initial conditions
        si, sFi, sBi, sW1i, sW2i, sW3i, e1i, e2i, e3i, e4i, e5i, e1Fi, e2Fi, \
            e3Fi, e4Fi, e5Fi, e1Bi, e2Bi, e3Bi, e4Bi, e5Bi, e1W1i, \
            e2W1i, e3W1i, e4W1i, e5W1i, e1W2i, e2W2i, e3W2i, e4W2i, \
            e5W2i, e1W3i, e2W3i, e3W3i, e4W3i, e5W3i, ii, iFi, iBi, iW1i, \
            iW2i, iW3i, ai, aFi, aBi, aW1i, aW2i, aW3i, \
            _i = np.asarray(self._y_init)[:, self._region-1]

        init_cond = list(
            chain(
                si.tolist(), sFi.tolist(), sBi.tolist(), sW1i.tolist(),
                sW2i.tolist(), sW3i.tolist(), e1i.tolist(), e2i.tolist(),
                e3i.tolist(), e4i.tolist(), e5i.tolist(), e1Fi.tolist(),
                e2Fi.tolist(), e3Fi.tolist(), e4Fi.tolist(), e5Fi.tolist(),
                e1Bi.tolist(), e2Bi.tolist(), e3Bi.tolist(),
                e4Bi.tolist(), e5Bi.tolist(), e1W1i.tolist(),
                e2W1i.tolist(), e3W1i.tolist(), e4W1i.tolist(),
                e5W1i.tolist(), e1W2i.tolist(), e2W2i.tolist(),
                e3W2i.tolist(), e4W2i.tolist(), e5W2i.tolist(),
                e1W3i.tolist(), e2W3i.tolist(), e3W3i.tolist(),
                e4W3i.tolist(), e5W3i.tolist(), ii.tolist(), iFi.tolist(),
                iBi.tolist(), iW1i.tolist(), iW2i.tolist(), iW3i.tolist(),
                ai.tolist(), aFi.tolist(), aBi.tolist(), aW1i.tolist(),
                aW2i.tolist(), aW3i.tolist(), _i.tolist()))

        # Solve the system of ODEs
        sol = solve_ivp(
            lambda t, y: self._right_hand_side(
                t, self._region, y, self._c, num_a_groups),
            [times[0], times[-1]], init_cond, method=method, t_eval=times)

        return sol

    def _split_simulate(
            self, parameters, times, method):
        r"""
        Computes the number of individuals in each compartment at the given
        time points and specified region.

        Parameters
        ----------
        parameters : list
            List of quantities that characterise the Warwick SEIR model in
            this order: index of region for which we wish to simulate,
            initial conditions matrices classifed by age and variant (column
            name) and region (row name) for each type of compartment (s, sF,
            sB, sW1, sW2, sW3, e1, ... e5, e1F, ... e5F, e1B, ... e5B,
            e1W1, ... e5W1, e1W2, ... e5W2, e1W3, ... e5W3, i, iF, iB, iW1,
            iW2, iW3, a, aF, aB, aW1, aW2, aW3, _), the age-dependent
            susceptibility of individuals to infection (beta), the rate of
            progression to infectious disease (alpha), the recovery rate
            (gamma), the age-dependent probability of displaying symptoms (d),
            the reduction in the transmission rate of infection
            for asymptomatic individuals (tau), the rates of waning of
            immunity (we) and the change in susceptibility due to the variant
            (omega).
        times : list
            List of time points at which we wish to evaluate the ODEs system.
        method : str
            The type of solver implemented by the :meth:`scipy.solve_ivp`.

        Returns
        -------
        numpy.array
            Age-structured output matrix of the simulation for the specified
            region.

        """
        # Split parameters into the features of the model
        self._region = parameters[0]
        self._y_init = parameters[1:50]
        self._c = parameters[50:57]
        self.contacts_timeline = em.MultiTimesContacts(
            self.matrices_contact,
            self.time_changes_contact,
            self.regions,
            self.matrices_region,
            self.time_changes_region)

        self._times = np.asarray(times)

        # Simulation using the scipy solver
        sol = self._scipy_solver(times, self._num_ages, method)
        n = self._num_ages

        output = sol['y']

        # Age-based total infected is infectious 'i' plus recovered 'r'
        total_infected = output[
            (36*self._num_ages):(37*self._num_ages), :] + output[
            (37*self._num_ages):(38*self._num_ages), :] + output[
            (38*self._num_ages):(39*self._num_ages), :] + output[
            (39*self._num_ages):(40*self._num_ages), :] + output[
            (40*self._num_ages):(41*self._num_ages), :] + output[
            (41*self._num_ages):(42*self._num_ages), :] + output[
            (42*self._num_ages):(43*self._num_ages), :] + output[
            (43*self._num_ages):(44*self._num_ages), :] + output[
            (44*self._num_ages):(45*self._num_ages), :] + output[
            (45*self._num_ages):(46*self._num_ages), :] + output[
            (46*self._num_ages):(47*self._num_ages), :] + output[
            (47*self._num_ages):(48*self._num_ages), :] + output[
            (48*self._num_ages):(49*self._num_ages), :]

        # Number of incidences is the increase in total_infected
        # between the time points (add a 0 at the front to
        # make the length consistent with the solution
        n_incidence = np.zeros((n, len(times)))
        n_incidence[:, 1:] = total_infected[:, 1:] - total_infected[:, :-1]

        # Append n_incidence to output
        # Output is a matrix with rows being S, Es, Is, R and Incidence
        output = np.concatenate((output, n_incidence), axis=0)

        # Get the selected outputs
        self._output_indices = np.arange(self._n_outputs)

        output_indices = []
        for i in self._output_indices:
            output_indices.extend(
                np.arange(i*self._num_ages, (i+1)*self._num_ages)
            )

        output = output[output_indices, :]

        return output.transpose()

    def read_contact_data(self, matrices_contact, time_changes_contact):
        """
        Reads in the timelines of contact data used for the modelling.

        Parameters
        ----------
        matrices_contact : list of ContactMatrix
            List of time-dependent contact matrices used for the modelling.
        time_changes_contact : list
            List of times at which the next contact matrix recorded starts to
            be used. In increasing order.

        """
        self.matrices_contact = matrices_contact
        self.time_changes_contact = time_changes_contact

    def read_regional_data(self, matrices_region, time_changes_region):
        """
        Reads in the timelines of regional data used for the modelling.

        Parameters
        ----------
        matrices_region : lists of RegionMatrix
            List of ime-dependent and region-specific relative susceptibility
            matrices used for the modelling.
        time_changes_region : list
            List of times at which the next instances of region-specific
            relative susceptibility matrices recorded start to be used. In
            increasing order.

        """
        self.matrices_region = matrices_region
        self.time_changes_region = time_changes_region

    def simulate(self, parameters):
        """
        Simulates the Warwick-Lancaster model using a
        :class:`ParametersController` for the model parameters.

        Extends the :meth:`_split_simulate`. Always apply methods
        :meth:`set_regions`, :meth:`set_age_groups`, :meth:`read_contact_data`
        and :meth:`read_regional_data` before
        running the :meth:`WarwickLancSEIRModel.simulate`.

        Parameters
        ----------
        parameters : ParametersController
            Controller class for the parameters used by the forward simulation
            of the model.

        Returns
        -------
        numpy.array
            Age-structured output matrix of the simulation for the specified
            region.

        """
        self.social_distancing_param = parameters.soc_dist_parameters()
        self.vaccine_param = parameters.vaccine_parameters()

        self._N = parameters.ICs.total_population()

        return self._simulate(
            parameters(), parameters.simulation_parameters.times)

    def _simulate(self, parameters, times):
        r"""
        PINTS-configured wrapper for the simulation method of the
        Warwick-Lancaster model.

        Extends the :meth:`_split_simulate`. Always apply methods
        :meth:`set_regions`, :meth:`set_age_groups`, :meth:`read_contact_data`
        and :meth:`read_regional_data` before
        running the :meth:`WarwickLancSEIRModel.simulate`.

        Parameters
        ----------
        parameters : list
            Long vector format of the quantities that characterise the
            Warwick-Lancaster SEIR model in this order:
            (1) index of region for which we wish to simulate,
            (2) initial conditions matrices classifed by age and variant
            (column name) and region (row name) for each type of compartment
            (s, sF, sB, sW1, sW2, sW3, e1, ... e5, e1F, ... e5F, e1B, ... e5FB,
            e1W1, ... e5W1, e1W2, ... e5W2, e1W3, ... e5W3, i, iF, iB, iW1,
            iW2, iW3, a, aF, aB, aW1, aW2, aW3, _),
            (3) the age-dependent susceptibility of individuals to infection
            (beta),
            (4) the rate of progression to infectious disease (alpha),
            (5) the recovery rate (gamma),
            (6) the age-dependent probability of displaying symptoms (d),
            (7) the reduction in the transmission rate of infection
            for asymptomatic individuals (tau),
            (8) the rates of waning of immunity (we)
            (9) the change in susceptibility due to the variant (omega)
            (10) the type of solver implemented by the :meth:`scipy.solve_ivp`
            and
            (11) the indicator parameter for deploying boosters to the
            recovered compartment.
            Splited into the formats necessary for the :meth:`_simulate`
            method.
        times : list
            List of time points at which we wish to evaluate the ODEs
            system.

        Returns
        -------
        numpy.array
            Age-structured output matrix of the simulation for the specified
            region.

        """
        # Number of regions and age groups
        self._num_ages = self.matrices_contact[0]._num_a_groups

        n_ages = self._num_ages
        n_reg = len(self.regions)

        start_index = n_reg * ((len(self._output_names)-1) * n_ages) + 1

        # Separate list of parameters into the structures needed for the
        # simulation
        my_parameters = []

        # Add index of region
        my_parameters.append(parameters[0])

        # Add initial conditions for the s, sF, sB, sW1, sW2, sW3, e1, ... e5,
        # e1F, ... e5F, e1B, ... e5FB, e1W1, ... e5W1, e1W2, ... e5W2,
        # e1W3, ... e5W3, i, iF, iB, iW1, iW2, iW3, a, aF, aB, aW1, aW2, aW3,
        # r compartments
        for c in range(len(self._output_names)-1):
            initial_cond_comp = []
            for r in range(n_reg):
                ind = r * n_ages + n_reg * c * n_ages + 1
                initial_cond_comp.append(
                    parameters[ind:(ind + n_ages)])
            my_parameters.append(initial_cond_comp)

        # Add other parameters
        # beta
        my_parameters.append(parameters[start_index:(start_index + n_ages)])

        # alpha
        my_parameters.append(parameters[start_index + n_ages])

        # gamma
        my_parameters.append(parameters[
            (start_index + 1 + n_ages):(start_index + 1 + 2 * n_ages)])

        # d
        my_parameters.append(parameters[
            (start_index + 1 + 2 * n_ages):(start_index + 1 + 3 * n_ages)])

        # tau
        my_parameters.append(parameters[start_index + 1 + 3 * n_ages])

        # we
        my_parameters.append(parameters[
            (start_index + 2 + 3 * n_ages):(start_index + 5 + 3 * n_ages)])

        # omega
        my_parameters.append(parameters[start_index + 5 + 3 * n_ages])

        # Add method
        method = parameters[start_index + 6 + 3 * n_ages]

        # Add eps
        self._eps = int(parameters[start_index + 7 + 3 * n_ages])

        return self._split_simulate(my_parameters,
                                    times,
                                    method)

    def _check_output_format(self, output):
        """
        Checks correct format of the output matrix.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the WarwickLancSEIRModel.

        """
        if np.asarray(output).ndim != 2:
            raise ValueError(
                'Model output storage format must be 2-dimensional.')
        if np.asarray(output).shape[0] != self._times.shape[0]:
            raise ValueError(
                    'Wrong number of rows for the model output.')
        if np.asarray(output).shape[1] != 50 * self._num_ages:
            raise ValueError(
                    'Wrong number of columns for the model output.')
        for r in np.asarray(output):
            for _ in r:
                if not isinstance(_, (np.integer, np.floating)):
                    raise TypeError(
                        'Model output elements must be integer or float.')

    def new_total_infections(self, output):
        """
        Computes number of new (symptomatic + asymptomatic) infections at each
        time step in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        It uses an output of the simulation method for the
        WarwickLancSEIRModel, taking all the rest of the parameters necessary
        for the computation from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Age-structured output of the simulation method for the
            WarwickLancSEIRModel.

        Returns
        -------
        list of numpy.array
            Age-structured matrices of the number of new (symptomatic +
            asymptomatic)  infections for different vaccination statuses
            (unvaccinated, fully-vaccinated, boosted, partially-waned,
            fully-waned, previous-variant immunity) from the simulation method
            for the WarwickLancSEIRModel.

        Notes
        -----
        Always run :meth:`WarwickLancSEIRModel.simulate` before running this
        one.

        """
        # Check correct format of parameters
        self._check_output_format(output)

        # Read parameters of the system
        alpha = self._c[1]

        d_tot_infec = np.empty((self._times.shape[0], self._num_ages))
        d_tot_infec_F = np.empty((self._times.shape[0], self._num_ages))
        d_tot_infec_B = np.empty((self._times.shape[0], self._num_ages))
        d_tot_infec_W1 = np.empty((self._times.shape[0], self._num_ages))
        d_tot_infec_W2 = np.empty((self._times.shape[0], self._num_ages))
        d_tot_infec_W3 = np.empty((self._times.shape[0], self._num_ages))

        for ind, _ in enumerate(self._times.tolist()):
            # Read from output
            e5 = output[ind, :][(10*self._num_ages):(11*self._num_ages)]
            e5F = output[ind, :][(15*self._num_ages):(16*self._num_ages)]
            e5B = output[ind, :][(20*self._num_ages):(21*self._num_ages)]
            e5W1 = output[ind, :][(25*self._num_ages):(26*self._num_ages)]
            e5W2 = output[ind, :][(30*self._num_ages):(31*self._num_ages)]
            e5W3 = output[ind, :][(35*self._num_ages):(36*self._num_ages)]

            # fraction of new infectives in delta_t time step
            d_tot_infec[ind, :] = alpha * e5
            d_tot_infec_F[ind, :] = alpha * e5F
            d_tot_infec_B[ind, :] = alpha * e5B
            d_tot_infec_W1[ind, :] = alpha * e5W1
            d_tot_infec_W2[ind, :] = alpha * e5W2
            d_tot_infec_W3[ind, :] = alpha * e5W3

            if np.any(d_tot_infec[ind, :] < 0):  # pragma: no cover
                d_tot_infec[ind, :] = np.zeros_like(d_tot_infec[ind, :])
            if np.any(d_tot_infec_F[ind, :] < 0):  # pragma: no cover
                d_tot_infec_F[ind, :] = np.zeros_like(d_tot_infec_F[ind, :])
            if np.any(d_tot_infec_B[ind, :] < 0):  # pragma: no cover
                d_tot_infec_B[ind, :] = np.zeros_like(d_tot_infec_B[ind, :])
            if np.any(d_tot_infec_W1[ind, :] < 0):  # pragma: no cover
                d_tot_infec_W1[ind, :] = np.zeros_like(d_tot_infec_W1[ind, :])
            if np.any(d_tot_infec_W2[ind, :] < 0):  # pragma: no cover
                d_tot_infec_W2[ind, :] = np.zeros_like(d_tot_infec_W2[ind, :])
            if np.any(d_tot_infec[ind, :] < 0):  # pragma: no cover
                d_tot_infec_W3[ind, :] = np.zeros_like(d_tot_infec_W3[ind, :])

        return [
            d_tot_infec, d_tot_infec_F, d_tot_infec_B,
            d_tot_infec_W1, d_tot_infec_W2, d_tot_infec_W3]

    def new_infections(self, output):
        """
        Computes number of new symptomatic infections at each time step in
        specified region, given the simulated timeline of susceptible number
        of individuals, for all age groups in the model.

        It uses an output of the simulation method for the
        WarwickLancSEIRModel, taking all the rest of the parameters necessary
        for the computation from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Age-structured output of the simulation method for the
            WarwickLancSEIRModel.

        Returns
        -------
        list of numpy.array
            Age-structured matrices of the number of new symptomatic infections
            for different vaccination statuses (unvaccinated, fully-vaccinated,
            boosted, partially-waned, fully-waned, previous-variant immunity)
            from the simulation method for the WarwickLancSEIRModel.

        Notes
        -----
        Always run :meth:`WarwickLancSEIRModel.simulate` before running this
        one.

        """
        # Check correct format of parameters
        self._check_output_format(output)

        # Read parameters of the system
        alpha, d = self._c[1], self._c[3]
        nu_symp = self.vaccine_param[4]

        d_infec = np.empty((self._times.shape[0], self._num_ages))
        d_infec_F = np.empty((self._times.shape[0], self._num_ages))
        d_infec_B = np.empty((self._times.shape[0], self._num_ages))
        d_infec_W1 = np.empty((self._times.shape[0], self._num_ages))
        d_infec_W2 = np.empty((self._times.shape[0], self._num_ages))
        d_infec_W3 = np.empty((self._times.shape[0], self._num_ages))

        for ind, _ in enumerate(self._times.tolist()):
            # Read from output
            e5 = output[ind, :][(10*self._num_ages):(11*self._num_ages)]
            e5F = output[ind, :][(15*self._num_ages):(16*self._num_ages)]
            e5B = output[ind, :][(20*self._num_ages):(21*self._num_ages)]
            e5W1 = output[ind, :][(25*self._num_ages):(26*self._num_ages)]
            e5W2 = output[ind, :][(30*self._num_ages):(31*self._num_ages)]
            e5W3 = output[ind, :][(35*self._num_ages):(36*self._num_ages)]

            # fraction of new infectives in delta_t time step
            d_infec[ind, :] = alpha * nu_symp[0] * np.multiply(d, e5)
            d_infec_F[ind, :] = alpha * nu_symp[1] * np.multiply(d, e5F)
            d_infec_B[ind, :] = alpha * nu_symp[2] * np.multiply(d, e5B)
            d_infec_W1[ind, :] = alpha * nu_symp[3] * np.multiply(d, e5W1)
            d_infec_W2[ind, :] = alpha * nu_symp[4] * np.multiply(d, e5W2)
            d_infec_W3[ind, :] = alpha * nu_symp[5] * np.multiply(d, e5W3)

            if np.any(d_infec[ind, :] < 0):  # pragma: no cover
                d_infec[ind, :] = np.zeros_like(d_infec[ind, :])
            if np.any(d_infec_F[ind, :] < 0):  # pragma: no cover
                d_infec_F[ind, :] = np.zeros_like(d_infec_F[ind, :])
            if np.any(d_infec_B[ind, :] < 0):  # pragma: no cover
                d_infec_B[ind, :] = np.zeros_like(d_infec_B[ind, :])
            if np.any(d_infec_W1[ind, :] < 0):  # pragma: no cover
                d_infec_W1[ind, :] = np.zeros_like(d_infec_W1[ind, :])
            if np.any(d_infec_W2[ind, :] < 0):  # pragma: no cover
                d_infec_W2[ind, :] = np.zeros_like(d_infec_W2[ind, :])
            if np.any(d_infec[ind, :] < 0):  # pragma: no cover
                d_infec_W3[ind, :] = np.zeros_like(d_infec_W3[ind, :])

        return [
            d_infec, d_infec_F, d_infec_B, d_infec_W1, d_infec_W2, d_infec_W3]

    def _check_new_infections_format(self, new_infections):
        """
        Checks correct format of the list new symptomatic infections matrices
        for different vaccination statuses (unvaccinated, fully-vaccinated,
        boosted, partially-waned, fully-waned, previous-variant immunity).

        Parameters
        ----------
        new_infections : list of numpy.array
            Age-structured matrices of the number of new symptomatic infections
            for different vaccination statuses (unvaccinated, fully-vaccinated,
            boosted, partially-waned, fully-waned, previous-variant immunity).

        """
        if np.asarray(new_infections).ndim != 3:
            raise ValueError(
                'Model new infections storage format must be 3-dimensional.')
        if np.asarray(new_infections).shape[0] != 6:
            raise ValueError(
                    'Wrong number of vaccination statuses for the model new \
                    infections.')
        if np.asarray(new_infections).shape[1] != self._times.shape[0]:
            raise ValueError(
                    'Wrong number of rows for the model new infections.')
        if np.asarray(new_infections).shape[2] != self._num_ages:
            raise ValueError(
                    'Wrong number of columns for the model new infections.')
        for r in np.asarray(new_infections):
            for _r in r:
                for _ in _r:
                    if not isinstance(_, (np.integer, np.floating)):
                        raise TypeError(
                            'Model`s new infections elements must be integer \
                                or float.')

    def new_hospitalisations(self, new_infections, pItoH, dItoH):
        """
        Computes number of new hospital admissions at each time step in
        specified region, given the simulated timeline of detectable
        symptomatic infected number of individuals, for all age groups
        in the model.

        It uses the array of the number of new symptomatic infections, obtained
        from an output of the simulation method for the WarwickLancSEIRModel,
        a distribution of the delay between onset of symptoms and
        hospitalisation, as well as the fraction of the number of symptomatic
        cases that end up hospitalised.

        Parameters
        ----------
        new_infections : list of numpy.array
            Age-structured arrays of the daily number of new symptomatic
            infections for different vaccination statuses (unvaccinated, fully-
            vaccinated, boosted, partially-waned, fully-waned, previous-variant
            immunity).
        pItoH : list
            Age-dependent fractions of the number of symptomatic cases that
            end up hospitalised.
        dItoH : list
            Distribution of the delay between onset of symptoms and
            hospitalisation. Must be normalised.

        Returns
        -------
        list of numpy.array
            Age-structured matrix of the number of new hospital admissions
            for different vaccination statuses (unvaccinated, fully-vaccinated,
            boosted, partially-waned, fully-waned, previous-variant immunity)
            from the simulation method for the WarwickLancSEIRModel.

        Notes
        -----
        Always run :meth:`WarwickLancSEIRModel.simulate` before running this
        one.

        """
        # Read parameters of the system
        nu_sev_h = self.vaccine_param[6]

        n_daily_hosp = np.zeros((self._times.shape[0], self._num_ages))
        n_daily_hosp_F = np.zeros((self._times.shape[0], self._num_ages))
        n_daily_hosp_B = np.zeros((self._times.shape[0], self._num_ages))
        n_daily_hosp_W1 = np.zeros((self._times.shape[0], self._num_ages))
        n_daily_hosp_W2 = np.zeros((self._times.shape[0], self._num_ages))
        n_daily_hosp_W3 = np.zeros((self._times.shape[0], self._num_ages))

        # Normalise dItoH
        dItoH = ((1/np.sum(dItoH)) * np.asarray(dItoH)).tolist()

        for ind, _ in enumerate(self._times.tolist()):
            if ind >= 30:
                n_daily_hosp[ind, :] = nu_sev_h[0] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:30][::-1]),
                        new_infections[0][(ind-29):(ind+1), :]), axis=0)
                n_daily_hosp_F[ind, :] = nu_sev_h[1] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:30][::-1]),
                        new_infections[1][(ind-29):(ind+1), :]), axis=0)
                n_daily_hosp_B[ind, :] = nu_sev_h[2] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:30][::-1]),
                        new_infections[2][(ind-29):(ind+1), :]), axis=0)
                n_daily_hosp_W1[ind, :] = nu_sev_h[3] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:30][::-1]),
                        new_infections[3][(ind-29):(ind+1), :]), axis=0)
                n_daily_hosp_W2[ind, :] = nu_sev_h[4] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:30][::-1]),
                        new_infections[4][(ind-29):(ind+1), :]), axis=0)
                n_daily_hosp_W3[ind, :] = nu_sev_h[5] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:30][::-1]),
                        new_infections[5][(ind-29):(ind+1), :]), axis=0)
            else:
                n_daily_hosp[ind, :] = nu_sev_h[0] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:(ind+1)][::-1]),
                        new_infections[0][:(ind+1), :]), axis=0)
                n_daily_hosp_F[ind, :] = nu_sev_h[1] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:(ind+1)][::-1]),
                        new_infections[1][:(ind+1), :]), axis=0)
                n_daily_hosp_B[ind, :] = nu_sev_h[2] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:(ind+1)][::-1]),
                        new_infections[2][:(ind+1), :]), axis=0)
                n_daily_hosp_W1[ind, :] = nu_sev_h[3] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:(ind+1)][::-1]),
                        new_infections[3][:(ind+1), :]), axis=0)
                n_daily_hosp_W2[ind, :] = nu_sev_h[4] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:(ind+1)][::-1]),
                        new_infections[4][:(ind+1), :]), axis=0)
                n_daily_hosp_W3[ind, :] = nu_sev_h[5] * np.array(pItoH) * \
                    np.sum(np.matmul(
                        np.diag(dItoH[:(ind+1)][::-1]),
                        new_infections[5][:(ind+1), :]), axis=0)

        for ind, _ in enumerate(self._times.tolist()):  # pragma: no cover
            if np.any(n_daily_hosp[ind, :] < 0):
                n_daily_hosp[ind, :] = np.zeros_like(n_daily_hosp[ind, :])
            if np.any(n_daily_hosp_F[ind, :] < 0):
                n_daily_hosp_F[ind, :] = np.zeros_like(n_daily_hosp_F[ind, :])
            if np.any(n_daily_hosp_B[ind, :] < 0):
                n_daily_hosp_B[ind, :] = np.zeros_like(n_daily_hosp_B[ind, :])
            if np.any(n_daily_hosp_W1[ind, :] < 0):
                n_daily_hosp_W1[ind, :] = np.zeros_like(
                    n_daily_hosp_W1[ind, :])
            if np.any(n_daily_hosp_W2[ind, :] < 0):
                n_daily_hosp_W2[ind, :] = np.zeros_like(
                    n_daily_hosp_W2[ind, :])
            if np.any(n_daily_hosp[ind, :] < 0):
                n_daily_hosp_W3[ind, :] = np.zeros_like(
                    n_daily_hosp_W3[ind, :])

        return [
            n_daily_hosp, n_daily_hosp_F, n_daily_hosp_B,
            n_daily_hosp_W1, n_daily_hosp_W2, n_daily_hosp_W3]

    def check_new_hospitalisation_format(self, new_infections, pItoH, dItoH):
        """
        Checks correct format of the inputs of number of hospitalisation
        calculation.

        Parameters
        ----------
        new_infections : list of numpy.array
            Age-structured arrays of the daily number of new symptomatic
            infections for different vaccination statuses (unvaccinated, fully-
            vaccinated, boosted, partially-waned, fully-waned, previous-variant
            immunity).
        pItoH : list
            Age-dependent fractions of the number of symptomatic cases that
            end up hospitalised.
        dItoH : list
            Distribution of the delay between onset of symptoms and
            hospitalisation. Must be normalised.

        """
        self._check_new_infections_format(new_infections)

        if np.asarray(pItoH).ndim != 1:
            raise ValueError('Fraction of the number of hospitalised \
                symptomatic cases storage format is 1-dimensional.')
        if np.asarray(pItoH).shape[0] != self._num_ages:
            raise ValueError('Wrong number of fractions of the number of\
                hospitalised symptomatic cases .')
        for _ in pItoH:
            if not isinstance(_, (int, float)):
                raise TypeError('Fraction of the number of hospitalised \
                    symptomatic cases must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Fraction of the number of hospitalised \
                    symptomatic cases must be => 0 and <=1.')

        if np.asarray(dItoH).ndim != 1:
            raise ValueError('Delays between onset of symptoms and \
                hospitalisation storage format is 1-dimensional.')
        if np.asarray(dItoH).shape[0] < 30:
            raise ValueError('Wrong number of delays between onset of \
                symptoms and hospitalisation.')
        for _ in dItoH:
            if not isinstance(_, (int, float)):
                raise TypeError('Delays between onset of symptoms and \
                    hospitalisation must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Delays between onset of symptoms and \
                    hospitalisation must be => 0 and <=1.')

    def new_deaths(self, new_hospitalisation, pHtoD, dHtoD):
        """
        Computes number of new deaths at each time step in
        specified region, given the simulated timeline of hospitalised
        number of individuals, for all age groups in the model.

        It uses the array of the number of new symptomatic infections, obtained
        from an output of the simulation method for the WarwickLancSEIRModel,
        a distribution of the delay between onset of symptoms and
        admission to ICU, as well as the fraction of the number of hospitalised
        cases that end up dying.

        Parameters
        ----------
        new_hospitalisation : list of numpy.array
            Age-structured arrays of the daily number of new hospitalised
            cases for different vaccination statuses (unvaccinated, fully-
            vaccinated, boosted, partially-waned, fully-waned, previous-variant
            immunity).
        pHtoD : list
            Age-dependent fractions of the number of hospitalised cases that
            die.
        dHtoD : list
            Distribution of the delay between onset of hospitalisation and
            death. Must be normalised.

        Returns
        -------
        list of numpy.array
            Age-structured matrices of the number of new deaths
            for different vaccination statuses (unvaccinated, fully-vaccinated,
            boosted, partially-waned, fully-waned, previous-variant immunity)
            from the simulation method for the WarwickLancSEIRModel.

        Notes
        -----
        Always run :meth:`WarwickLancSEIRModel.simulate` before running this
        one.

        """
        # Read parameters of the system
        nu_sev_d = self.vaccine_param[7]

        n_daily_dths = np.zeros((self._times.shape[0], self._num_ages))
        n_daily_dths_F = np.zeros((self._times.shape[0], self._num_ages))
        n_daily_dths_B = np.zeros((self._times.shape[0], self._num_ages))
        n_daily_dths_W1 = np.zeros((self._times.shape[0], self._num_ages))
        n_daily_dths_W2 = np.zeros((self._times.shape[0], self._num_ages))
        n_daily_dths_W3 = np.zeros((self._times.shape[0], self._num_ages))

        # Normalise dHtoD
        dHtoD = ((1/np.sum(dHtoD)) * np.asarray(dHtoD)).tolist()

        for ind, _ in enumerate(self._times.tolist()):
            if ind >= 30:
                n_daily_dths[ind, :] = nu_sev_d[0] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:30][::-1]),
                        new_hospitalisation[0][(ind-29):(ind+1), :]), axis=0)
                n_daily_dths_F[ind, :] = nu_sev_d[1] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:30][::-1]),
                        new_hospitalisation[1][(ind-29):(ind+1), :]), axis=0)
                n_daily_dths_B[ind, :] = nu_sev_d[2] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:30][::-1]),
                        new_hospitalisation[2][(ind-29):(ind+1), :]), axis=0)
                n_daily_dths_W1[ind, :] = nu_sev_d[3] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:30][::-1]),
                        new_hospitalisation[3][(ind-29):(ind+1), :]), axis=0)
                n_daily_dths_W2[ind, :] = nu_sev_d[4] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:30][::-1]),
                        new_hospitalisation[4][(ind-29):(ind+1), :]), axis=0)
                n_daily_dths_W3[ind, :] = nu_sev_d[5] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:30][::-1]),
                        new_hospitalisation[5][(ind-29):(ind+1), :]), axis=0)
            else:
                n_daily_dths[ind, :] = nu_sev_d[0] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:(ind+1)][::-1]),
                        new_hospitalisation[0][:(ind+1), :]), axis=0)
                n_daily_dths_F[ind, :] = nu_sev_d[1] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:(ind+1)][::-1]),
                        new_hospitalisation[1][:(ind+1), :]), axis=0)
                n_daily_dths_B[ind, :] = nu_sev_d[2] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:(ind+1)][::-1]),
                        new_hospitalisation[2][:(ind+1), :]), axis=0)
                n_daily_dths_W1[ind, :] = nu_sev_d[3] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:(ind+1)][::-1]),
                        new_hospitalisation[3][:(ind+1), :]), axis=0)
                n_daily_dths_W2[ind, :] = nu_sev_d[4] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:(ind+1)][::-1]),
                        new_hospitalisation[4][:(ind+1), :]), axis=0)
                n_daily_dths_W3[ind, :] = nu_sev_d[5] * np.array(pHtoD) * \
                    np.sum(np.matmul(
                        np.diag(dHtoD[:(ind+1)][::-1]),
                        new_hospitalisation[5][:(ind+1), :]), axis=0)

        for ind, _ in enumerate(self._times.tolist()):  # pragma: no cover
            if np.any(n_daily_dths[ind, :] < 0):
                n_daily_dths[ind, :] = np.zeros_like(n_daily_dths[ind, :])
            if np.any(n_daily_dths_F[ind, :] < 0):
                n_daily_dths_F[ind, :] = np.zeros_like(n_daily_dths_F[ind, :])
            if np.any(n_daily_dths_B[ind, :] < 0):
                n_daily_dths_B[ind, :] = np.zeros_like(n_daily_dths_B[ind, :])
            if np.any(n_daily_dths_W1[ind, :] < 0):
                n_daily_dths_W1[ind, :] = np.zeros_like(
                    n_daily_dths_W1[ind, :])
            if np.any(n_daily_dths_W2[ind, :] < 0):
                n_daily_dths_W2[ind, :] = np.zeros_like(
                    n_daily_dths_W2[ind, :])
            if np.any(n_daily_dths[ind, :] < 0):
                n_daily_dths_W3[ind, :] = np.zeros_like(
                    n_daily_dths_W3[ind, :])

        return [
            n_daily_dths, n_daily_dths_F, n_daily_dths_B,
            n_daily_dths_W1, n_daily_dths_W2, n_daily_dths_W3]

    def check_new_deaths_format(
            self, new_hospitalisation, pHtoD, dHtoD):
        """
        Checks correct format of the inputs of number of death
        calculation.

        Parameters
        ----------
        new_hospitalisation : numpy.array
            Age-structured arrays of the daily number of new hospitalised
            cases for different vaccination statuses (unvaccinated, fully-
            vaccinated, boosted, partially-waned, fully-waned, previous-variant
            immunity).
        pHtoD : int or float
            Age-dependent fractions of the number of hospitalised cases that
            die.
        dHtoD : list
            Distribution of the delay between onset of hospitalisation and
            death. Must be normalised.

        """
        self._check_new_infections_format(new_hospitalisation)

        if np.asarray(pHtoD).ndim != 1:
            raise ValueError('Fraction of the number of deaths \
                from hospitalised cases storage format is 1-dimensional.')
        if np.asarray(pHtoD).shape[0] != self._num_ages:
            raise ValueError('Wrong number of fractions of the number of\
                deaths from hospitalised cases.')
        for _ in pHtoD:
            if not isinstance(_, (int, float)):
                raise TypeError('Fraction of the number of deaths \
                from hospitalised cases must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Fraction of the number of deaths \
                from hospitalised cases must be => 0 and <=1.')

        if np.asarray(dHtoD).ndim != 1:
            raise ValueError('Delays between hospital admission and \
                death storage format is 1-dimensional.')
        if np.asarray(dHtoD).shape[0] < 30:
            raise ValueError('Wrong number of delays between hospital \
                admission and death.')
        for _ in dHtoD:
            if not isinstance(_, (int, float)):
                raise TypeError('Delays between  hospital \
                    admission and death must be integer or float.')
            if (_ < 0) or (_ > 1):
                raise ValueError('Delays between  hospital \
                    admission and death must be => 0 and <=1.')

    def loglik_deaths(self, obs_death, new_deaths, niu, k):
        r"""
        Computes the log-likelihood for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean :math:`\mu_{r,t_k,i}`
        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where
        :math:`\mu_{r,t_k,i}` is the number of new deaths in specified region,
        for age group :math:`i` on day :math:`t_k`.

        It uses new_infections output of the simulation method for the
        WarwickLancSEIRModel, taking all the rest of the parameters necessary
        for the computation from the way its simulation has been fitted.

        Parameters
        ----------
        obs_death : list
            List of number of observed deaths by age group at time point k.
        new_deaths : list of numpy.array
            Age-structured matrix of the number of new deaths for different
            vaccination statuses (unvaccinated, fully-vaccinated,
            boosted, partially-waned, fully-waned, previous-variant immunity)
            from the simulation method for the WarwickLancSEIRModel.
        niu : float
            Dispersion factor for the negative binomial distribution.
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of log-likelihoods for the observed number
            of deaths in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`WarwickLancSEIRModel.new_infections` and
        :meth:`WarwickLancSEIRModel.check_death_format` before running this
        one.

        """
        self._check_time_step_format(k)

        # Check correct format for observed number of deaths
        if np.asarray(obs_death).ndim != 1:
            raise ValueError('Observed number of deaths by age category \
                storage format is 1-dimensional.')
        if np.asarray(obs_death).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of deaths.')
        for _ in obs_death:
            if not isinstance(_, (int, np.integer)):
                raise TypeError('Observed number of deaths must be integer.')
            if _ < 0:
                raise ValueError('Observed number of deaths must be => 0.')

        if not hasattr(self, 'actual_deaths'):
            self.actual_deaths = [0] * 150
        self.actual_deaths[k] = sum(self.mean_deaths(k, new_deaths))

        # Compute mean of negative-binomial
        if k != 0:
            if np.sum(self.mean_deaths(k, new_deaths)) != 0:
                return nbinom.logpmf(
                    k=obs_death,
                    n=(1/niu) * self.mean_deaths(k, new_deaths),
                    p=1/(1+niu))
            else:
                return np.zeros(self._num_ages)
        else:
            return np.zeros(self._num_ages)

    def check_death_format(self, niu):
        """
        Checks correct format of the inputs of number of death calculation.

        Parameters
        ----------
        new_deaths : list of numpy.array
            Age-structured matrices of the number of new deaths for different
            vaccination statuses (unvaccinated, fully-vaccinated,
            boosted, partially-waned, fully-waned, previous-variant immunity)
            from the simulation method for the WarwickLancSEIRModel.
        niu : float
            Dispersion factor for the negative binomial distribution.

        """
        if not isinstance(niu, (int, float)):
            raise TypeError('Dispersion factor must be integer or float.')
        if niu <= 0:
            raise ValueError('Dispersion factor must be > 0.')

    def mean_deaths(self, k, new_deaths):
        """
        Computes the mean of the negative binomial distribution used to
        calculate number of deaths for specified age group.

        Parameters
        ----------
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.
        new_deaths : list of numpy.array
            Age-structured matrices of the number of new deaths  for different
            vaccination statuses (unvaccinated, fully-vaccinated,
            boosted, partially-waned, fully-waned, previous-variant immunity)
            from the simulation method for the WarwickLancSEIRModel.

        Returns
        -------
        numpy.array
            Age-structured matrix of the expected number of deaths to be
            observed in specified region at time :math:`t_k`.

        """
        return new_deaths[0][k, :] + new_deaths[1][k, :] + \
            new_deaths[2][k, :] + new_deaths[4][k, :] + new_deaths[5][k, :]

    def samples_deaths(self, new_deaths, niu, k):
        r"""
        Computes samples for the number of deaths at time step
        :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of deaths is assumed to be distributed according to
        a negative binomial distribution with mean :math:`\mu_{r,t_k,i}`
        and variance :math:`\mu_{r,t_k,i} (\nu + 1)`, where
        :math:`\mu_{r,t_k,i}` is the number of new deaths in specified region,
        for age group :math:`i` on day :math:`t_k`.

        It uses an output of the simulation method for the
        WarwickLancSEIRModel, taking all the rest of the parameters necessary
        for the computation from the way its simulation has been fitted.

        Parameters
        ----------
        new_deaths : numpy.array
            Age-structured matrices of the number of new deaths for different
            vaccination statuses (unvaccinated, fully-vaccinated,
            boosted, partially-waned, fully-waned, previous-variant immunity)
            from the simulation method for the WarwickLancSEIRModel.
        niu : float
            Dispersion factor for the negative binomial distribution.
        k : int
            Index of day for which we intend to sample the number of deaths for
            by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of sampled number of deaths in specified
            region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`WarwickLancSEIRModel.new_infections` and
        :meth:`WarwickLancSEIRModel.check_death_format` before running this
        one.

        """
        self._check_time_step_format(k)

        # Compute mean of negative-binomial
        if k != 0:
            if np.sum(self.mean_deaths(k, new_deaths)) != 0:
                return nbinom.rvs(
                    n=(1/niu) * self.mean_deaths(k, new_deaths),
                    p=1/(1+niu))
            else:
                return np.zeros(self._num_ages)
        else:
            return np.zeros_like(self.mean_deaths(k, new_deaths))

    def loglik_positive_tests(self, obs_pos, output, tests, sens, spec, k):
        r"""
        Computes the log-likelihood for the number of positive tests at time
        step :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of positive tests is assumed to be distributed according to
        a binomial distribution with parameters :math:`n = n_{r,t_k,i}` and

        .. math::
            p = k_{sens} (1-\frac{S_{r,t_k,i}}{N_{r,i}}) + (
                1-k_{spec}) \frac{S_{r,t_k,i}}{N_{r,i}}

        where :math:`n_{r,t_k,i}` is the number of tests conducted for
        people in age group :math:`i` in specified region :math:`r` at time
        atep :math:`t_k`, :math:`k_{sens}` and :math:`k_{spec}` are the
        sensitivity and specificity respectively of a test, while
        is the probability of demise :math:`k-l` days after infection and
        :math:`\delta_{r,t_l,i}^{infec}` is the number of new infections
        in specified region, for age group :math:`i` on day :math:`t_l`.

        It uses an output of the simulation method for the
        WarwickLancSEIRModel, taking all the rest of the parameters necessary
        for the computation from the way its simulation has been fitted.

        Parameters
        ----------
        obs_pos : list
            List of number of observed positive test results by age group at
            time point k.
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the WarwickLancSEIRModel.
        tests : list
            List of conducted tests in specified region and at time point k
            classifed by age groups.
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).
        k : int
            Index of day for which we intend to sample the number of positive
            test results by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of log-likelihoods for the obsereved number
            of positive test results for each age group in specified region at
            time :math:`t_k`.

        Notes
        -----
        Always run :meth:`WarwickLancSEIRModel.simulate` and
        :meth:`WarwickLancSEIRModel.check_positives_format` before running this
        one.

        """
        self._check_time_step_format(k)

        # Check correct format for observed number of positive results
        if np.asarray(obs_pos).ndim != 1:
            raise ValueError('Observed number of postive tests results by age \
                category storage format is 1-dimensional.')
        if np.asarray(obs_pos).shape[0] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of postive tests results.')
        for _ in obs_pos:
            if not isinstance(_, (int, np.integer)):
                raise TypeError('Observed number of postive tests results must\
                    be integer.')
            if _ < 0:
                raise ValueError('Observed number of postive tests results \
                    must be => 0.')

        # Check correct format for number of tests based on the observed number
        # of positive results
        for i, _ in enumerate(tests):
            if _ < obs_pos[i]:
                raise ValueError('Not enough performed tests for the number \
                    of observed positives.')

        a = self._num_ages
        # Compute parameters of binomial
        suscep = output[k, :a]
        pop = 0
        for i in range(6):
            pop += output[k, (i*a):((i+1)*a)]

        return binom.logpmf(
            k=obs_pos,
            n=tests,
            p=self.mean_positives(sens, spec, suscep, pop))

    def _check_time_step_format(self, k):
        if not isinstance(k, int):
            raise TypeError('Index of time of computation of the \
                log-likelihood must be integer.')
        if k < 0:
            raise ValueError('Index of time of computation of the \
                log-likelihood must be >= 0.')
        if k >= self._times.shape[0]:
            raise ValueError('Index of time of computation of the \
                log-likelihood must be within those considered in the output.')

    def check_positives_format(self, output, tests, sens, spec):
        """
        Checks correct format of the inputs of number of positive test results
        calculation.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the WarwickLancSEIRModel.
        tests : list
            List of conducted tests in specified region and at time point k
            classifed by age groups.
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).

        """
        self._check_output_format(output)
        if np.asarray(tests).ndim != 2:
            raise ValueError('Number of tests conducted by age category \
                storage format is 2-dimensional.')
        if np.asarray(tests).shape[1] != self._num_ages:
            raise ValueError('Wrong number of age groups for observed number \
                of tests conducted.')
        for i in tests:
            for _ in i:
                if not isinstance(_, (int, np.integer)):
                    raise TypeError('Number of tests conducted must be \
                        integer.')
                if _ < 0:
                    raise ValueError('Number of tests conducted ratio must \
                        be => 0.')
        if not isinstance(sens, (int, float)):
            raise TypeError('Sensitivity must be integer or float.')
        if (sens < 0) or (sens > 1):
            raise ValueError('Sensitivity must be >= 0 and <=1.')
        if not isinstance(spec, (int, float)):
            raise TypeError('Specificity must be integer or float.')
        if (spec < 0) or (spec > 1):
            raise ValueError('Specificity must be >= 0 and >=1.')

    def mean_positives(self, sens, spec, suscep, pop):
        """
        Computes the mean of the binomial distribution used to
        calculate number of positive test results for specified age group.

        Parameters
        ----------
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).
        suscep : numpy.array
            Age-structured matrix of the current number of susceptibles
            in the population.
        pop : numpy.array
            Age-structured matrix of the current number of individuals
            in the population.

        Returns
        -------
        numpy.array
            Age-structured matrix of the expected number of positive test
            results to be observed in specified region at time :math:`t_k`.

        """
        return sens * (1-np.divide(suscep, pop)) + (1-spec) * np.divide(
            suscep, pop)

    def samples_positive_tests(self, output, tests, sens, spec, k):
        r"""
        Computes the samples for the number of positive tests at time
        step :math:`k` in specified region, given the simulated timeline of
        susceptible number of individuals, for all age groups in the model.

        The number of positive tests is assumed to be distributed according to
        a binomial distribution with parameters :math:`n = n_{r,t_k,i}` and

        .. math::
            p = k_{sens} (1-\frac{S_{r,t_k,i}}{N_{r,i}}) + (
                1-k_{spec}) \frac{S_{r,t_k,i}}{N_{r,i}}

        where :math:`n_{r,t_k,i}` is the number of tests conducted for
        people in age group :math:`i` in specified region :math:`r` at time
        atep :math:`t_k`, :math:`k_{sens}` and :math:`k_{spec}` are the
        sensitivity and specificity respectively of a test, while
        is the probability of demise :math:`k-l` days after infection and
        :math:`\delta_{r,t_l,i}^{infec}` is the number of new infections
        in specified region, for age group :math:`i` on day :math:`t_l`.

        It uses an output of the simulation method for the
        WarwickLancSEIRModel, taking all the rest of the parameters necessary
        for the computation from the way its simulation has been fitted.

        Parameters
        ----------
        output : numpy.array
            Age-structured output matrix of the simulation method
            for the WarwickLancSEIRModel.
        tests : list
            List of conducted tests in specified region and at time point k
            classifed by age groups.
        sens : float or int
            Sensitivity of the test (or ratio of true positives).
        spec : float or int
            Specificity of the test (or ratio of true negatives).
        k : int
            Index of day for which we intend to sample the number of positive
            test results by age group.

        Returns
        -------
        numpy.array
            Age-structured matrix of sampled number of positive test results
            in specified region at time :math:`t_k`.

        Notes
        -----
        Always run :meth:`WarwickLancSEIRModel.simulate` and
        :meth:`WarwickLancSEIRModel.check_positives_format` before running this
        one.

        """
        self._check_time_step_format(k)

        a = self._num_ages
        # Compute parameters of binomial
        suscep = output[k, :a]
        pop = 0
        for i in range(6):
            pop += output[k, (i*a):((i+1)*a)]

        return binom.rvs(
            n=tests,
            p=self.mean_positives(sens, spec, suscep, pop))
