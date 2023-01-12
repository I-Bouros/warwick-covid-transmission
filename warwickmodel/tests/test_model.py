#
# WarwickLancWarwickLancSEIRModel Class
#
# This file is part of WARWICKMODEL
# (https://github.com/I-Bouros/warwick-covid-transmission.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#

import unittest

import numpy as np
import numpy.testing as npt

import epimodels as em
import warwickmodel as wm


class TestWarwickLancSEIRModel(unittest.TestCase):
    """
    Test the 'WarwickLancSEIRModel' class.
    """
    def test__init__(self):
        model = wm.WarwickLancSEIRModel()

        self.assertEqual(
            model._output_names,
            ['S', 'Sf', 'Sb', 'Sw1', 'Sw2', 'Sw3', 'E1', 'E2', 'E3', 'E4',
             'E5', 'E1f', 'E2f', 'E3f', 'E4f', 'E5f', 'E1b', 'E2b', 'E3b',
             'E4b', 'E5b', 'E1w1', 'E2w1', 'E3w1', 'E4w1', 'E5w1', 'E1w2',
             'E2w2', 'E3w2', 'E4w2', 'E5w2', 'E1w3', 'E2w3', 'E3w3', 'E4w3',
             'E5w3', 'I', 'If', 'Ib', 'Iw1', 'Iw2', 'Iw3', 'A', 'Sf', 'Ab',
             'Aw1', 'Aw2', 'Aw3', 'R', 'Incidence'])
        self.assertEqual(
            model._parameter_names,
            ['S0', 'Sf0', 'Sb0', 'Sw10', 'Sw20', 'Sw30', 'E10', 'E20', 'E30',
             'E40', 'E50', 'E1f0', 'E2f0', 'E3f0', 'E4f0', 'E5f0', 'E1b0',
             'E2b0', 'E3b0', 'E4b0', 'E5b0', 'E1w10', 'E2w10', 'E3w10',
             'E4w10', 'E5w10', 'E1w20', 'E2w20', 'E3w20', 'E4w20', 'E5w20',
             'E1w30', 'E2w30', 'E3w30', 'E4w30', 'E5w30', 'I0', 'If0', 'Ib0',
             'Iw10', 'Iw20', 'Iw30', 'A0', 'Sf0', 'Ab0', 'Aw10', 'Aw20',
             'Aw30', 'R0', 'beta', 'alpha', 'gamma', 'd', 'tau', 'we',
             'omega'])
        self.assertEqual(model._n_outputs, 50)
        self.assertEqual(model._n_parameters, 56)

    def test_n_outputs(self):
        model = wm.WarwickLancSEIRModel()
        self.assertEqual(model.n_outputs(), 50)

    def test_n_parameters(self):
        model = wm.WarwickLancSEIRModel()
        self.assertEqual(model.n_parameters(), 56)

    def test_output_names(self):
        model = wm.WarwickLancSEIRModel()
        self.assertEqual(
            model.output_names(),
            ['S', 'Sf', 'Sb', 'Sw1', 'Sw2', 'Sw3', 'E1', 'E2', 'E3', 'E4',
             'E5', 'E1f', 'E2f', 'E3f', 'E4f', 'E5f', 'E1b', 'E2b', 'E3b',
             'E4b', 'E5b', 'E1w1', 'E2w1', 'E3w1', 'E4w1', 'E5w1', 'E1w2',
             'E2w2', 'E3w2', 'E4w2', 'E5w2', 'E1w3', 'E2w3', 'E3w3', 'E4w3',
             'E5w3', 'I', 'If', 'Ib', 'Iw1', 'Iw2', 'Iw3', 'A', 'Sf', 'Ab',
             'Aw1', 'Aw2', 'Aw3', 'R', 'Incidence'])

    def test_parameter_names(self):
        model = wm.WarwickLancSEIRModel()
        self.assertEqual(
            model.parameter_names(),
            ['S0', 'Sf0', 'Sb0', 'Sw10', 'Sw20', 'Sw30', 'E10', 'E20', 'E30',
             'E40', 'E50', 'E1f0', 'E2f0', 'E3f0', 'E4f0', 'E5f0', 'E1b0',
             'E2b0', 'E3b0', 'E4b0', 'E5b0', 'E1w10', 'E2w10', 'E3w10',
             'E4w10', 'E5w10', 'E1w20', 'E2w20', 'E3w20', 'E4w20', 'E5w20',
             'E1w30', 'E2w30', 'E3w30', 'E4w30', 'E5w30', 'I0', 'If0', 'Ib0',
             'Iw10', 'Iw20', 'Iw30', 'A0', 'Sf0', 'Ab0', 'Aw10', 'Aw20',
             'Aw30', 'R0', 'beta', 'alpha', 'gamma', 'd', 'tau', 'we',
             'omega'])

    def test_set_regions(self):
        model = wm.WarwickLancSEIRModel()
        regions = ['UK', 'FR']
        model.set_regions(regions)

        self.assertEqual(
            model.region_names(),
            ['UK', 'FR'])

    def test_set_age_groups(self):
        model = wm.WarwickLancSEIRModel()
        age_groups = ['0-10', '10-20']
        model.set_age_groups(age_groups)

        self.assertEqual(
            model.age_groups_names(),
            ['0-10', '10-20'])

    def test_set_outputs(self):
        model = wm.WarwickLancSEIRModel()
        outputs = ['S', 'I', 'If', 'Incidence']
        model.set_outputs(outputs)

        with self.assertRaises(ValueError):
            outputs1 = ['S', 'E', 'I', 'If', 'Incidence']
            model.set_outputs(outputs1)

    def test_simulate(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=2
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=[1, 2],
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        output_my_solver = model.simulate(parameters)

        npt.assert_almost_equal(
            output_my_solver,
            np.array([
                [7, 8] + [0, 0] * 49,
                [7, 8] + [0, 0] * 49
            ]), decimal=3)

    def test_new_infections(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=1
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=[1, 2],
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        output = model.simulate(parameters)

        npt.assert_array_equal(
            model.new_infections(output)[0],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_infections(output)[1],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_infections(output)[2],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_infections(output)[3],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_infections(output)[4],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_infections(output)[5],
            np.array([[0, 0], [0, 0]]))

        with self.assertRaises(ValueError):
            output1 = np.array([5, 6] + [0, 0] * 49)
            model.new_infections(output1)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6] + [0, 0] * 48,
                [5, 6] + [0, 0] * 48])
            model.new_infections(output1)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6] + [0, 0] * 49,
                [5, 6] + [0, 0] * 49,
                [5, 6] + [0, 0] * 49])
            model.new_infections(output1)

        with self.assertRaises(TypeError):
            output1 = np.array([
                ['5', 6] + [0, 0] * 49,
                [5, 6] + ['0', 0] * 49])
            model.new_infections(output1)

    def test_new_hospitalisations(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=1
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=[1, 2],
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        new_infections = model.new_infections(
            model.simulate(parameters))

        pItoH = np.ones(len(age_groups))
        dItoH = np.ones(30)

        npt.assert_array_equal(
            model.new_hospitalisations(
                new_infections, pItoH, dItoH)[0],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_hospitalisations(
                new_infections, pItoH, dItoH)[1],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_hospitalisations(
                new_infections, pItoH, dItoH)[2],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_hospitalisations(
                new_infections, pItoH, dItoH)[3],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_hospitalisations(
                new_infections, pItoH, dItoH)[4],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_hospitalisations(
                new_infections, pItoH, dItoH)[5],
            np.array([[0, 0], [0, 0]]))

    def test_check_new_hospitalisations(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=1
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=[1, 2],
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        new_infections = model.new_infections(
            model.simulate(parameters))

        pItoH = [1, 1]
        dItoH = [1] * 30

        with self.assertRaises(ValueError):
            new_infections1 = np.array([[0, 0], [0, 0]])

            model.check_new_hospitalisation_format(
                new_infections1, pItoH, dItoH)

        with self.assertRaises(ValueError):
            new_infections1 = [
                np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0]])]

            model.check_new_hospitalisation_format(
                new_infections1, pItoH, dItoH)

        with self.assertRaises(ValueError):
            new_infections1 = [
                np.array([[0, 0], [0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0], [0, 0]])]

            model.check_new_hospitalisation_format(
                new_infections1, pItoH, dItoH)

        with self.assertRaises(ValueError):
            new_infections1 = [
                np.array([[0, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0]])]

            model.check_new_hospitalisation_format(
                new_infections1, pItoH, dItoH)

        with self.assertRaises(TypeError):
            new_infections1 = [
                np.array([[0, '0'], [0, 0]]), np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [0, '0']]), np.array([[0, 0], [0, 0]])]

            model.check_new_hospitalisation_format(
                new_infections1, pItoH, dItoH)

        with self.assertRaises(ValueError):
            pItoH1 = {'0.9': 0}

            model.check_new_hospitalisation_format(
                new_infections, pItoH1, dItoH)

        with self.assertRaises(ValueError):
            pItoH1 = [1, 1, 1]

            model.check_new_hospitalisation_format(
                new_infections, pItoH1, dItoH)

        with self.assertRaises(TypeError):
            pItoH1 = [1, '1']

            model.check_new_hospitalisation_format(
                new_infections, pItoH1, dItoH)

        with self.assertRaises(ValueError):
            pItoH1 = [-0.2, 0.5]

            model.check_new_hospitalisation_format(
                new_infections, pItoH1, dItoH)

        with self.assertRaises(ValueError):
            pItoH1 = [0.5, 1.5]

            model.check_new_hospitalisation_format(
                new_infections, pItoH1, dItoH)

        with self.assertRaises(ValueError):
            dItoH1 = {'0.9': 0}

            model.check_new_hospitalisation_format(
                new_infections, pItoH, dItoH1)

        with self.assertRaises(ValueError):
            dItoH1 = [1, 1, 1]

            model.check_new_hospitalisation_format(
                new_infections, pItoH, dItoH1)

        with self.assertRaises(TypeError):
            dItoH1 = [1, '1'] * 17

            model.check_new_hospitalisation_format(
                new_infections, pItoH, dItoH1)

        with self.assertRaises(ValueError):
            dItoH1 = [-0.2] + [0.5] * 30

            model.check_new_hospitalisation_format(
                new_infections, pItoH, dItoH1)

        with self.assertRaises(ValueError):
            dItoH1 = [0.5] * 30 + [1.5]

            model.check_new_hospitalisation_format(
                new_infections, pItoH, dItoH1)

    def test_new_deaths(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=1
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=[1, 2],
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        pItoH = [1, 1]
        dItoH = [1] * 30

        pHtoD = [1, 1]
        dHtoD = [1] * 30

        new_infections = model.new_infections(
            model.simulate(parameters))

        new_hospitalisations = model.new_hospitalisations(
            new_infections, pItoH, dItoH)

        npt.assert_array_equal(
            model.new_deaths(
                new_hospitalisations, pHtoD, dHtoD)[0],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_deaths(
                new_hospitalisations, pHtoD, dHtoD)[1],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_deaths(
                new_hospitalisations, pHtoD, dHtoD)[2],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_deaths(
                new_hospitalisations, pHtoD, dHtoD)[3],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_deaths(
                new_hospitalisations, pHtoD, dHtoD)[4],
            np.array([[0, 0], [0, 0]]))
        npt.assert_array_equal(
            model.new_deaths(
                new_hospitalisations, pHtoD, dHtoD)[5],
            np.array([[0, 0], [0, 0]]))

    def test_check_new_deaths(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=1
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=[1, 2],
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        pItoH = [1, 1]
        dItoH = [1] * 30

        pHtoD = [1, 1]
        dHtoD = [1] * 30

        new_infections = model.new_infections(
            model.simulate(parameters))

        new_hospitalisations = model.new_hospitalisations(
            new_infections, pItoH, dItoH)

        with self.assertRaises(ValueError):
            new_hospitalisations1 = np.array([[0, 0], [0, 0]])

            model.check_new_deaths_format(
                new_hospitalisations1, pHtoD, dHtoD)

        with self.assertRaises(ValueError):
            new_hospitalisations1 = [
                np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0]])]

            model.check_new_deaths_format(
                new_hospitalisations1, pHtoD, dHtoD)

        with self.assertRaises(ValueError):
            new_hospitalisations1 = [
                np.array([[0, 0], [0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0], [0, 0]])]

            model.check_new_deaths_format(
                new_hospitalisations1, pHtoD, dHtoD)

        with self.assertRaises(ValueError):
            new_hospitalisations1 = [
                np.array([[0, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0]]),
                np.array([[0, 0, 0], [0, 0, 0]])]

            model.check_new_deaths_format(
                new_hospitalisations1, pHtoD, dHtoD)

        with self.assertRaises(TypeError):
            new_hospitalisations1 = [
                np.array([[0, '0'], [0, 0]]), np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]),
                np.array([[0, 0], [0, '0']]), np.array([[0, 0], [0, 0]])]

            model.check_new_deaths_format(
                new_hospitalisations1, pHtoD, dHtoD)

        with self.assertRaises(ValueError):
            pHtoD1 = {'0.9': 0}

            model.check_new_deaths_format(
                new_hospitalisations, pHtoD1, dHtoD)

        with self.assertRaises(ValueError):
            pHtoD1 = [1, 1, 1]

            model.check_new_deaths_format(
                new_hospitalisations, pHtoD1, dHtoD)

        with self.assertRaises(TypeError):
            pHtoD1 = [1, '1']

            model.check_new_deaths_format(
                new_hospitalisations, pHtoD1, dHtoD)

        with self.assertRaises(ValueError):
            pHtoD1 = [-0.2, 0.5]

            model.check_new_deaths_format(
                new_hospitalisations, pHtoD1, dHtoD)

        with self.assertRaises(ValueError):
            pHtoD1 = [0.5, 1.5]

            model.check_new_deaths_format(
                new_hospitalisations, pHtoD1, dHtoD)

        with self.assertRaises(ValueError):
            dHtoD1 = {'0.9': 0}

            model.check_new_deaths_format(
                new_hospitalisations, pHtoD, dHtoD1)

        with self.assertRaises(ValueError):
            dHtoD1 = [1, 1, 1]

            model.check_new_deaths_format(
                new_hospitalisations, pHtoD, dHtoD1)

        with self.assertRaises(TypeError):
            dHtoD1 = [1, '1'] * 17

            model.check_new_deaths_format(
                new_hospitalisations, pHtoD, dHtoD1)

        with self.assertRaises(ValueError):
            dHtoD1 = [-0.2] + [0.5] * 30

            model.check_new_deaths_format(
                new_hospitalisations, pHtoD, dHtoD1)

        with self.assertRaises(ValueError):
            dHtoD1 = [0.5] * 30 + [1.5]

            model.check_new_deaths_format(
                new_hospitalisations, pHtoD, dHtoD1)

    def test_loglik_deaths(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=2
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=[1, 2],
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        pItoH = [1, 1]
        dItoH = [1] * 30

        pHtoD = [1, 1]
        dHtoD = [1] * 30

        new_infections = model.new_infections(
            model.simulate(parameters))

        new_hospitalisations = model.new_hospitalisations(
            new_infections, pItoH, dItoH)

        new_deaths = model.new_deaths(
            new_hospitalisations, pHtoD, dHtoD)

        obs_death = [10, 12]

        self.assertEqual(
            model.loglik_deaths(
                obs_death, new_deaths, 0.5, 1).shape,
            (len(age_groups),))

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                obs_death, new_deaths, 0.5, -1)

        with self.assertRaises(TypeError):
            model.loglik_deaths(
                obs_death, new_deaths, 0.5, '1')

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                obs_death, new_deaths, 0.5, 2)

        with self.assertRaises(ValueError):
            model.loglik_deaths(
                0, new_deaths, 0.5, 1)

        with self.assertRaises(ValueError):
            obs_death1 = np.array([5, 6, 0, 0])

            model.loglik_deaths(
                obs_death1, new_deaths, 0.5, 1)

        with self.assertRaises(TypeError):
            obs_death1 = np.array(['5', 6])

            model.loglik_deaths(
                obs_death1, new_deaths, 0.5, 1)

        with self.assertRaises(ValueError):
            obs_death1 = np.array([5, -1])

            model.loglik_deaths(
                obs_death1, new_deaths, 0.5, 1)

    def test_check_death_format(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=2
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=[1, 2],
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        model.simulate(parameters)

        with self.assertRaises(TypeError):
            model.check_death_format('0.5')

        with self.assertRaises(ValueError):
            model.check_death_format(-2)

    def test_samples_deaths(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=2
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=np.arange(1, 61).tolist(),
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        pItoH = [1, 1]
        dItoH = [1] * 30

        pHtoD = [1, 1]
        dHtoD = [1] * 30

        new_infections = model.new_infections(
            model.simulate(parameters))

        new_hospitalisations = model.new_hospitalisations(
            new_infections, pItoH, dItoH)

        new_deaths = model.new_deaths(
            new_hospitalisations, pHtoD, dHtoD)

        self.assertEqual(
            model.samples_deaths(new_deaths, 0.5, 41).shape,
            (len(age_groups),))

        self.assertEqual(
            model.samples_deaths(new_deaths, 0.5, 1).shape,
            (len(age_groups),))

        with self.assertRaises(ValueError):
            model.samples_deaths(new_deaths, 0.5, -1)

        with self.assertRaises(TypeError):
            model.samples_deaths(new_deaths, 0.5, '1')

        with self.assertRaises(ValueError):
            model.samples_deaths(new_deaths, 0.5, 62)

    def test_loglik_positive_tests(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=2
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=np.arange(1, 61).tolist(),
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        output = model.simulate(parameters)

        obs_pos = [10, 12]
        tests = [[20, 30], [10, 0]]
        sens = 0.9
        spec = 0.1

        self.assertEqual(
            model.loglik_positive_tests(
                obs_pos, output, tests[0], sens, spec, 0).shape,
            (len(age_groups),))

        with self.assertRaises(TypeError):
            model.loglik_positive_tests(
                obs_pos, output, tests[0], sens, spec, '1')

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                obs_pos, output, tests[0], sens, spec, -1)

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                obs_pos, output, tests[0], sens, spec, 63)

        with self.assertRaises(ValueError):
            model.loglik_positive_tests(
                0, output, tests[0], sens, spec, 0)

        with self.assertRaises(ValueError):
            obs_pos1 = np.array([5, 6, 0, 0])

            model.loglik_positive_tests(
                obs_pos1, output, tests[0], sens, spec, 0)

        with self.assertRaises(TypeError):
            obs_pos1 = np.array(['5', 6])

            model.loglik_positive_tests(
                obs_pos1, output, tests[0], sens, spec, 0)

        with self.assertRaises(ValueError):
            obs_pos1 = np.array([5, -1])

            model.loglik_positive_tests(
                obs_pos1, output, tests[0], sens, spec, 0)

        with self.assertRaises(ValueError):
            obs_pos1 = np.array([5, 40])

            model.loglik_positive_tests(
                obs_pos1, output, tests[0], sens, spec, 0)

    def test_check_positives_format(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=1
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=[1, 2],
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        output = model.simulate(parameters)

        tests = [[20, 30], [10, 0]]
        sens = 0.9
        spec = 0.1

        with self.assertRaises(ValueError):
            output1 = np.array([5, 6] + [0, 0] * 49)

            model.check_positives_format(
                output1, tests, sens, spec)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6] + [0, 0] * 48,
                [5, 6] + [0, 0] * 48])

            model.check_positives_format(
                output1, tests, sens, spec)

        with self.assertRaises(ValueError):
            output1 = np.array([
                [5, 6] + [0, 0] * 49,
                [5, 6] + [0, 0] * 49,
                [5, 6] + [0, 0] * 49])

            model.check_positives_format(
                output1, tests, sens, spec)

        with self.assertRaises(TypeError):
            output1 = np.array([
                [5, 6] + [0, '0'] * 49,
                [5, '6'] + [0, 0] * 49])

            model.check_positives_format(
                output1, tests, sens, spec)

        with self.assertRaises(ValueError):
            tests1 = 100

            model.check_positives_format(
                output, tests1, sens, spec)

        with self.assertRaises(ValueError):
            tests1 = np.array([2, 50])

            model.check_positives_format(
                output, tests1, sens, spec)

        with self.assertRaises(ValueError):
            tests1 = np.array([[20, 30, 1], [10, 0, 0]])

            model.check_positives_format(
                output, tests1, sens, spec)

        with self.assertRaises(TypeError):
            tests1 = np.array([[20, '30'], [10, 0]])

            model.check_positives_format(
                output, tests1, sens, spec)

        with self.assertRaises(ValueError):
            tests1 = np.array([[-1, 50], [10, 0]])

            model.check_positives_format(
                output, tests1, sens, spec)

        with self.assertRaises(TypeError):
            model.check_positives_format(
                output, tests, '0.9', spec)

        with self.assertRaises(ValueError):
            model.check_positives_format(
                output, tests, -0.2, spec)

        with self.assertRaises(ValueError):
            model.check_positives_format(
                output, tests, 1.2, spec)

        with self.assertRaises(TypeError):
            model.check_positives_format(
                output, tests, sens, '0.1')

        with self.assertRaises(ValueError):
            model.check_positives_format(
                output, tests, sens, -0.1)

        with self.assertRaises(ValueError):
            model.check_positives_format(
                output, tests, sens, 1.2)

    def test_samples_positive_tests(self):
        model = wm.WarwickLancSEIRModel()

        # Populate the model
        regions = ['UK', 'FR']
        age_groups = ['0-10', '10-25']

        # Initial state of the system
        region_data_matrix_0 = np.array([[1, 10], [1, 6]])
        region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

        regional_0 = em.RegionMatrix(
            regions[0], age_groups, region_data_matrix_0)
        regional_1 = em.RegionMatrix(
            regions[1], age_groups, region_data_matrix_1)

        contacts = em.ContactMatrix(
            age_groups, np.ones((len(age_groups), len(age_groups))))
        matrices_contact = [contacts]

        # Matrices contact
        time_changes_contact = [1]
        time_changes_region = [1]

        matrices_region = [[regional_0, regional_1]]

        model.set_regions(regions)
        model.set_age_groups(age_groups)
        model.read_contact_data(matrices_contact, time_changes_contact)
        model.read_regional_data(matrices_region, time_changes_region)

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=2
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[5, 6] + [0, 0] * 5, [7, 8] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_asym_IC=[[0, 0] * 6, [0, 0] * 6],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0],
            omega=1
        )

        # Set transmission parameters
        transmission_parameters = wm.Transmission(
            model=model,
            beta=0.5 * np.ones(len(age_groups)),
            alpha=0.5,
            gamma=1 * np.ones(len(age_groups))
        )

        # Set other simulation parameters
        simulation_parameters = wm.SimParameters(
            model=model,
            method='RK45',
            times=[1, 2],
            eps=False
        )

        # Set vaccination parameters
        vaccine_parameters = wm.VaccineParameters(
            model=model,
            vac=0,
            vacb=0,
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=1
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters)

        output = model.simulate(parameters)

        tests = [[20, 30], [10, 0]]
        sens = 0.9
        spec = 0.1

        self.assertEqual(
            model.samples_positive_tests(
                output, tests[0], sens, spec, 0).shape,
            (len(age_groups),))

        with self.assertRaises(TypeError):
            model.samples_positive_tests(
                output, tests[0], sens, spec, '1')

        with self.assertRaises(ValueError):
            model.samples_positive_tests(
                output, tests[0], sens, spec, -1)

        with self.assertRaises(ValueError):
            model.samples_positive_tests(
                output, tests[0], sens, spec, 3)
