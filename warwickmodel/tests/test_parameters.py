#
# WarwickLancSEIRModel Class
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

#
# Model Examples
#

examplemodel = wm.WarwickLancSEIRModel()
examplemodel2 = wm.WarwickLancSEIRModel()

# Populate the model
regions = ['UK', 'FR']
age_groups = ['0-10', '10-25']

# Initial state of the system
region_data_matrix_0 = np.array([[1, 10], [1, 6]])
region_data_matrix_1 = np.array([[0.5, 3], [0.3, 3]])

region_data_matrix_0_2 = np.array([[1, 10], [1, 6]])
region_data_matrix_1_2 = np.array([[5, 3], [3, 3]])

regional_0 = em.RegionMatrix(
    regions[0], age_groups, region_data_matrix_0)
regional_1 = em.RegionMatrix(
    regions[1], age_groups, region_data_matrix_1)

regional_0_2 = em.RegionMatrix(
    regions[0], age_groups, region_data_matrix_0_2)
regional_1_2 = em.RegionMatrix(
    regions[1], age_groups, region_data_matrix_1_2)

contacts = em.ContactMatrix(
    age_groups, np.ones((len(age_groups), len(age_groups))))
matrices_contact = [contacts]

# Matrices contact
time_changes_contact = [1]
time_changes_region = [1]

matrices_region = [[regional_0, regional_1]]
matrices_region_2 = [[regional_0_2, regional_1_2]]

examplemodel.set_regions(regions)
examplemodel.set_age_groups(age_groups)
examplemodel.read_contact_data(matrices_contact, time_changes_contact)
examplemodel.read_regional_data(matrices_region, time_changes_region)

examplemodel2.set_regions(regions)
examplemodel2.set_age_groups(age_groups)
examplemodel2.read_contact_data(matrices_contact, time_changes_contact)
examplemodel2.read_regional_data(matrices_region_2, time_changes_region)

#
# Test ICs Class
#


class TestICs(unittest.TestCase):
    """
    Test the 'ICs' class.
    """
    def test__init__(self):
        model = examplemodel

        # ICs parameters
        susceptibles = [[500, 600] + [0, 0] * 5, [700, 800] + [0, 0] * 5]
        exposed1 = [[0, 0] * 6, [0, 0] * 6]
        exposed2 = [[0, 0] * 6, [0, 0] * 6]
        exposed3 = [[0, 0] * 6, [0, 0] * 6]
        exposed4 = [[0, 0] * 6, [0, 0] * 6]
        exposed5 = [[0, 0] * 6, [0, 0] * 6]
        infectives_sym = [[10, 20] * 5 + [0, 0], [0, 5] * 5 + [0, 0]]
        infectives_asym = [[15, 10] * 5 + [0, 0], [30, 10] * 5 + [0, 0]]
        recovered = [[0, 0], [0, 0]]

        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=susceptibles,
            exposed1_IC=exposed1,
            exposed2_IC=exposed2,
            exposed3_IC=exposed3,
            exposed4_IC=exposed4,
            exposed5_IC=exposed5,
            infectives_sym_IC=infectives_sym,
            infectives_asym_IC=infectives_asym,
            recovered_IC=recovered
        )

        self.assertEqual(ICs_parameters.model, model)

        npt.assert_array_equal(np.array(ICs_parameters.susceptibles),
                               np.array([
                                   [500, 600] + [0, 0] * 5,
                                   [700, 800] + [0, 0] * 5]))

        npt.assert_array_equal(np.array(ICs_parameters.exposed1),
                               np.array([[0, 0] * 6, [0, 0] * 6]))

        npt.assert_array_equal(np.array(ICs_parameters.exposed2),
                               np.array([[0, 0] * 6, [0, 0] * 6]))

        npt.assert_array_equal(np.array(ICs_parameters.exposed3),
                               np.array([[0, 0] * 6, [0, 0] * 6]))

        npt.assert_array_equal(np.array(ICs_parameters.exposed4),
                               np.array([[0, 0] * 6, [0, 0] * 6]))

        npt.assert_array_equal(np.array(ICs_parameters.exposed5),
                               np.array([[0, 0] * 6, [0, 0] * 6]))

        npt.assert_array_equal(np.array(ICs_parameters.infectives_sym),
                               np.array([
                                   [10, 20] * 5 + [0, 0],
                                   [0, 5] * 5 + [0, 0]]))

        npt.assert_array_equal(np.array(ICs_parameters.infectives_asym),
                               np.array([
                                   [15, 10] * 5 + [0, 0],
                                   [30, 10] * 5 + [0, 0]]))

        npt.assert_array_equal(np.array(ICs_parameters.recovered),
                               np.array([[0, 0], [0, 0]]))

        with self.assertRaises(TypeError):
            model1 = 0

            wm.ICs(
                model=model1,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            susceptibles1 = [1500, 600]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            susceptibles1 = [
                [1500, 600] + [0, 0] * 5,
                [700, 400] + [0, 0] * 5,
                [100, 200] + [0, 0] * 5]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            susceptibles1 = [
                [1500, 600, 700] + [0, 0, 0] * 5,
                [400, 100, 200] + [0, 0, 0] * 5]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            susceptibles1 = [
                [1500, '600'] + [0, 0] * 5,
                [700, 400] + [0, 0] * 5]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles1,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed11 = [0, 0]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed11,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed11 = [[0, 0] * 6, [0, 0] * 6, [0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed11,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed11 = [[0, 0, 0] * 6, [0, 0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed11,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed11 = [[0, '0'] + [0, 0] * 5, [0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed11,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed21 = [0, 0]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed21,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed21 = [[0, 0] * 6, [0, 0] * 6, [0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed21,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed21 = [[0, 0, 0] * 6, [0, 0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed21,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed21 = [[0, '0'] + [0, 0] * 5, [0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed21,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed31 = [0, 0]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed31,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed31 = [[0, 0] * 6, [0, 0] * 6, [0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed31,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed31 = [[0, 0, 0] * 6, [0, 0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed31,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed31 = [[0, '0'] + [0, 0] * 5, [0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed31,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed41 = [0, 0]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed41,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed41 = [[0, 0] * 6, [0, 0] * 6, [0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed41,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            exposed41 = [[0, 0, 0] * 6, [0, 0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed41,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            exposed41 = [[0, '0'] + [0, 0] * 5, [0, 0] * 6]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed41,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives_sym1 = [10, 20]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym1,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives_sym1 = [
                [10, 20] * 5 + [0, 0],
                [0, 5] * 5 + [0, 0],
                [1, 2] * 5 + [0, 0]]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym1,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives_sym1 = [
                [10, 20, 15] * 5 + [0, 0, 0],
                [0, 5, 7] * 5 + [0, 0, 0]]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym1,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            infectives_sym1 = [
                [10, 20] * 5 + ['0', 0],
                [0, 5] * 5 + [0, 0]]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym1,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives_asym1 = [10, 20]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym1,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives_asym1 = [
                [10, 20] * 5 + [0, 0],
                [0, 5] * 5 + [0, 0],
                [1, 2] * 5 + [0, 0]]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym1,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            infectives_asym1 = [
                [10, 20, 15] * 5 + [0, 0, 0],
                [0, 5, 7] * 5 + [0, 0, 0]]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym1,
                recovered_IC=recovered)

        with self.assertRaises(TypeError):
            infectives_asym1 = [
                [10, 20] * 5 + ['0', 0],
                [0, 5] * 5 + [0, 0]]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym1,
                recovered_IC=recovered)

        with self.assertRaises(ValueError):
            recovered1 = [0, 0]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered1)

        with self.assertRaises(ValueError):
            recovered1 = [[0, 0], [0, 0], [0, 0]]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered1)

        with self.assertRaises(ValueError):
            recovered1 = [[0, 0, 0], [0, 0, 0]]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered1)

        with self.assertRaises(TypeError):
            recovered1 = [[0, 0], ['0', 0]]

            wm.ICs(
                model=model,
                susceptibles_IC=susceptibles,
                exposed1_IC=exposed1,
                exposed2_IC=exposed2,
                exposed3_IC=exposed3,
                exposed4_IC=exposed4,
                exposed5_IC=exposed5,
                infectives_sym_IC=infectives_sym,
                infectives_asym_IC=infectives_asym,
                recovered_IC=recovered1)

    def test_total_population(self):
        model = examplemodel

        # ICs parameters
        susceptibles = [[500, 600] + [0, 0] * 5, [700, 800] + [0, 0] * 5]
        exposed1 = [[0, 0] * 6, [0, 0] * 6]
        exposed2 = [[0, 0] * 6, [0, 0] * 6]
        exposed3 = [[0, 0] * 6, [0, 0] * 6]
        exposed4 = [[0, 0] * 6, [0, 0] * 6]
        exposed5 = [[0, 0] * 6, [0, 0] * 6]
        infectives_sym = [[10, 20] * 5 + [0, 0], [0, 5] * 5 + [0, 0]]
        infectives_asym = [[15, 10] * 5 + [0, 0], [30, 10] * 5 + [0, 0]]
        recovered = [[0, 0], [0, 0]]

        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=susceptibles,
            exposed1_IC=exposed1,
            exposed2_IC=exposed2,
            exposed3_IC=exposed3,
            exposed4_IC=exposed4,
            exposed5_IC=exposed5,
            infectives_sym_IC=infectives_sym,
            infectives_asym_IC=infectives_asym,
            recovered_IC=recovered
        )

        npt.assert_array_equal(
            ICs_parameters.total_population(),
            np.asarray([[625, 750], [850, 875]])
        )

    def test__call__(self):
        model = examplemodel

        # ICs parameters
        susceptibles = [[500, 600] + [0, 0] * 5, [700, 800] + [0, 0] * 5]
        exposed1 = [[0, 0] * 6, [0, 0] * 6]
        exposed2 = [[0, 0] * 6, [0, 0] * 6]
        exposed3 = [[0, 0] * 6, [0, 0] * 6]
        exposed4 = [[0, 0] * 6, [0, 0] * 6]
        exposed5 = [[0, 0] * 6, [0, 0] * 6]
        infectives_sym = [[10, 20] * 5 + [0, 0], [0, 5] * 5 + [0, 0]]
        infectives_asym = [[15, 10] * 5 + [0, 0], [30, 10] * 5 + [0, 0]]
        recovered = [[0, 0], [0, 0]]

        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=susceptibles,
            exposed1_IC=exposed1,
            exposed2_IC=exposed2,
            exposed3_IC=exposed3,
            exposed4_IC=exposed4,
            exposed5_IC=exposed5,
            infectives_sym_IC=infectives_sym,
            infectives_asym_IC=infectives_asym,
            recovered_IC=recovered
        )

        npt.assert_array_equal(
            ICs_parameters(),
            np.asarray([[[500, 600], [700, 800]]] + [[[0, 0], [0, 0]]] * 35 + [
                [[10, 20], [0, 5]]] * 5 + [[[0, 0], [0, 0]]] + [
                [[15, 10], [30, 10]]] * 5 + [[[0, 0], [0, 0]]] * 2)
        )

#
# Test RegParameters Class
#


class TestRegParameters(unittest.TestCase):
    """
    Test the 'RegParameters' class.
    """
    def test__init__(self):
        model = examplemodel

        region_index = 2

        RegParameters = wm.RegParameters(
            model=model,
            region_index=region_index
        )

        self.assertEqual(RegParameters.model, model)

        self.assertEqual(RegParameters.region_index, 2)

        with self.assertRaises(TypeError):
            model1 = '0'

            wm.RegParameters(
                model=model1,
                region_index=region_index
            )

        with self.assertRaises(TypeError):
            region_index1 = 0.5

            wm.RegParameters(
                model=model,
                region_index=region_index1
            )

        with self.assertRaises(ValueError):
            region_index1 = 0

            wm.RegParameters(
                model=model,
                region_index=region_index1
            )

        with self.assertRaises(ValueError):
            region_index1 = 3

            wm.RegParameters(
                model=model,
                region_index=region_index1
            )

    def test__call__(self):
        model = examplemodel

        region_index = 2

        RegParameters = wm.RegParameters(
            model=model,
            region_index=region_index
        )

        self.assertEqual(RegParameters(), region_index)

#
# Test DiseaseParameters Class
#


class TestDiseaseParameters(unittest.TestCase):
    """
    Test the 'DiseaseParameters' class.
    """
    def test__init__(self):
        model = examplemodel

        d = [0.05, 0.02]
        tau = 0.4
        we = [0.02, 0.01, 0]
        omega = 1

        DiseaseParameters = wm.DiseaseParameters(
            model=model,
            d=d,
            tau=tau,
            we=we,
            omega=omega
        )

        self.assertEqual(DiseaseParameters.model, model)
        npt.assert_array_equal(DiseaseParameters.d, np.array([0.05, 0.02]))
        self.assertEqual(DiseaseParameters.tau, 0.4)
        npt.assert_array_equal(DiseaseParameters.we, np.array([0.02, 0.01, 0]))
        self.assertEqual(DiseaseParameters.omega, 1)

        with self.assertRaises(TypeError):
            model1 = [4]

            wm.DiseaseParameters(
                model=model1,
                d=d,
                tau=tau,
                we=we,
                omega=omega
            )

        with self.assertRaises(TypeError):
            d1 = ['0.4', 0.2]

            wm.DiseaseParameters(
                model=model,
                d=d1,
                tau=tau,
                we=we,
                omega=omega
            )

        with self.assertRaises(ValueError):
            d1 = [[0.2], [0.4]]

            wm.DiseaseParameters(
                model=model,
                d=d1,
                tau=tau,
                we=we,
                omega=omega
            )

        with self.assertRaises(ValueError):
            d1 = [0.2, 0.4, 0.7]

            wm.DiseaseParameters(
                model=model,
                d=d1,
                tau=tau,
                we=we,
                omega=omega
            )

        with self.assertRaises(ValueError):
            d1 = -0.2

            wm.DiseaseParameters(
                model=model,
                d=d1,
                tau=tau,
                we=we,
                omega=omega
            )

        with self.assertRaises(ValueError):
            d1 = [0.2, 1.3]

            wm.DiseaseParameters(
                model=model,
                d=d1,
                tau=tau,
                we=we,
                omega=omega
            )

        with self.assertRaises(TypeError):
            tau1 = '4'

            wm.DiseaseParameters(
                model=model,
                d=d,
                tau=tau1,
                we=we,
                omega=omega
            )

        with self.assertRaises(ValueError):
            tau1 = -1

            wm.DiseaseParameters(
                model=model,
                d=d,
                tau=tau1,
                we=we,
                omega=omega
            )

        with self.assertRaises(ValueError):
            tau1 = 1.2

            wm.DiseaseParameters(
                model=model,
                d=d,
                tau=tau1,
                we=we,
                omega=omega
            )

        with self.assertRaises(TypeError):
            we1 = ['0.4', 0, 0.2]

            wm.DiseaseParameters(
                model=model,
                d=d,
                tau=tau,
                we=we1,
                omega=omega
            )

        with self.assertRaises(ValueError):
            we1 = [[0.2], [0], [0.4]]

            wm.DiseaseParameters(
                model=model,
                d=d,
                tau=tau,
                we=we1,
                omega=omega
            )

        with self.assertRaises(ValueError):
            we1 = [0.2, 0.4, 0.7, 0]

            wm.DiseaseParameters(
                model=model,
                d=d,
                tau=tau,
                we=we1,
                omega=omega
            )

        with self.assertRaises(ValueError):
            we1 = -0.2

            wm.DiseaseParameters(
                model=model,
                d=d,
                tau=tau,
                we=we1,
                omega=omega
            )

        with self.assertRaises(ValueError):
            we1 = [-0.2, 1.3, 0]

            wm.DiseaseParameters(
                model=model,
                d=d,
                tau=tau,
                we=we1,
                omega=omega
            )

        with self.assertRaises(TypeError):
            omega1 = '4'

            wm.DiseaseParameters(
                model=model,
                d=d,
                tau=tau,
                we=we,
                omega=omega1
            )

        with self.assertRaises(ValueError):
            omega1 = -1

            wm.DiseaseParameters(
                model=model,
                d=d,
                tau=tau,
                we=we,
                omega=omega1
            )

    def test__call__(self):
        model = examplemodel

        d = [0.05, 0.02]
        tau = 0.4
        we = [0.02, 0.02, 0]
        omega = 1

        DiseaseParameters = wm.DiseaseParameters(
            model=model,
            d=d,
            tau=tau,
            we=we,
            omega=omega
        )

        self.assertEqual(DiseaseParameters(),
                         [[0.05, 0.02], 0.4, [0.02, 0.02, 0], 1])

#
# Test Transmission Class
#


class TestTransmission(unittest.TestCase):
    """
    Test the 'Transmission' class.
    """
    def test__init__(self):
        model = examplemodel

        # Transmission parameters
        beta = 0.5
        alpha = 0.5
        gamma = 1

        TransmissionParam = wm.Transmission(
            model=model,
            beta=beta,
            alpha=alpha,
            gamma=gamma
        )

        self.assertEqual(TransmissionParam.model, model)
        npt.assert_array_equal(TransmissionParam.beta, np.array([0.5, 0.5]))
        self.assertEqual(TransmissionParam.alpha, 0.5)
        npt.assert_array_equal(TransmissionParam.gamma, np.array([1, 1]))

        with self.assertRaises(TypeError):
            model1 = '0'

            wm.Transmission(
                model=model1,
                beta=beta,
                alpha=alpha,
                gamma=gamma
            )

        with self.assertRaises(ValueError):
            wm.Transmission(
                model=model,
                beta={'0.9': 0},
                alpha=alpha,
                gamma=gamma
            )

        with self.assertRaises(ValueError):
            wm.Transmission(
                model=model,
                beta=[[0.4], [2.3]],
                alpha=alpha,
                gamma=gamma
            )

        with self.assertRaises(ValueError):
            wm.Transmission(
                model=model,
                beta=[0.5, 0.9, 2],
                alpha=alpha,
                gamma=gamma
            )

        with self.assertRaises(TypeError):
            wm.Transmission(
                model=model,
                beta=[0.5, '2'],
                alpha=alpha,
                gamma=gamma
            )

        with self.assertRaises(ValueError):
            wm.Transmission(
                model=model,
                beta=[-3, 2],
                alpha=alpha,
                gamma=gamma
            )

        with self.assertRaises(TypeError):
            wm.Transmission(
                model=model,
                beta=beta,
                alpha='0.5',
                gamma=gamma
            )

        with self.assertRaises(ValueError):
            wm.Transmission(
                model=model,
                beta=beta,
                alpha=-1,
                gamma=gamma
            )

        with self.assertRaises(ValueError):
            wm.Transmission(
                model=model,
                beta=beta,
                alpha=alpha,
                gamma={'0.9': 0}
            )

        with self.assertRaises(ValueError):
            wm.Transmission(
                model=model,
                beta=beta,
                alpha=alpha,
                gamma=[[0.4], [2.3]]
            )

        with self.assertRaises(ValueError):
            wm.Transmission(
                model=model,
                beta=beta,
                alpha=alpha,
                gamma=[0.5, 0.9, 2]
            )

        with self.assertRaises(TypeError):
            wm.Transmission(
                model=model,
                beta=beta,
                alpha=alpha,
                gamma=[0.5, '2']
            )

        with self.assertRaises(ValueError):
            wm.Transmission(
                model=model,
                beta=beta,
                alpha=alpha,
                gamma=[-3, 2]
            )

    def test__call__(self):
        model = examplemodel

        # Transmission parameters
        beta = [0.5, 0.5]
        alpha = 0.5
        gamma = [1, 1]

        TransmissionParam = wm.Transmission(
            model=model,
            beta=beta,
            alpha=alpha,
            gamma=gamma
        )

        self.assertEqual(TransmissionParam(), [[0.5, 0.5], 0.5, [1, 1]])

#
# Test SimParameters Class
#


class TestSimParameters(unittest.TestCase):
    """
    Test the 'SimParameters' class.
    """
    def test__init__(self):
        model = examplemodel

        method = 'RK45'
        times = [1, 2]

        SimulationParam = wm.SimParameters(
            model=model,
            method=method,
            times=times
        )

        self.assertEqual(SimulationParam.model, model)
        self.assertEqual(SimulationParam.method, 'RK45')
        self.assertEqual(SimulationParam.times, [1, 2])
        self.assertEqual(SimulationParam.eps, False)

        with self.assertRaises(TypeError):
            model1 = {'0': 0}

            wm.SimParameters(
                model=model1,
                method=method,
                times=times
            )

        with self.assertRaises(TypeError):
            wm.SimParameters(
                model=model,
                method=3,
                times=times
            )

        with self.assertRaises(ValueError):
            wm.SimParameters(
                model=model,
                method='my-solver',
                times=times
            )

        with self.assertRaises(TypeError):
            wm.SimParameters(
                model=model,
                method=method,
                times='0'
            )

        with self.assertRaises(TypeError):
            wm.SimParameters(
                model=model,
                method=method,
                times=[1, '2']
            )

        with self.assertRaises(ValueError):
            wm.SimParameters(
                model=model,
                method=method,
                times=[0, 1]
            )

        with self.assertRaises(TypeError):
            wm.SimParameters(
                model=model,
                method=method,
                times=times,
                eps=0
            )

    def test__call__(self):
        model = examplemodel

        # Set other simulation parameters
        method = 'RK45'
        times = [1, 2]
        eps = True

        SimulationParam = wm.SimParameters(
            model=model,
            method=method,
            times=times,
            eps=eps
        )

        self.assertEqual(SimulationParam(), ['RK45', True])

#
# Test SocDistParameters Class
#


class TestSocDistParameters(unittest.TestCase):
    """
    Test the 'SocDistParameters' class.
    """
    def test__init__(self):
        model = examplemodel

        SocDistParam = wm.SocDistParameters(
            model=model)

        self.assertEqual(SocDistParam.model, model)
        npt.assert_array_equal(SocDistParam.phi, np.ones(2))

        phi = 0.2

        with self.assertRaises(TypeError):
            model1 = {'0': 0}

            wm.SocDistParameters(
                model=model1,
                phi=phi
            )

        with self.assertRaises(ValueError):
            wm.SocDistParameters(
                model=model,
                phi={'0.9': 0}
            )

        with self.assertRaises(ValueError):
            wm.SocDistParameters(
                model=model,
                phi=[[0.4], [0.3]]
            )

        with self.assertRaises(ValueError):
            wm.SocDistParameters(
                model=model,
                phi=[0.5, 0.9, 1]
            )

        with self.assertRaises(TypeError):
            wm.SocDistParameters(
                model=model,
                phi=[0.5, '0.2']
            )

        with self.assertRaises(ValueError):
            wm.SocDistParameters(
                model=model,
                phi=[-3, 0.2]
            )

        with self.assertRaises(ValueError):
            wm.SocDistParameters(
                model=model,
                phi=[0.3, 2]
            )

    def test__call__(self):
        model = examplemodel

        # Set social distancing parameters
        phi = 0.2

        SocDistParam = wm.SocDistParameters(
            model=model,
            phi=phi)

        npt.assert_array_equal(SocDistParam(), np.array([0.2, 0.2]))

#
# Test VaccineParameters Class
#


class TestVaccineParameters(unittest.TestCase):
    """
    Test the 'VaccineParameters' class.
    """
    def test__init__(self):
        model = examplemodel

        vac = 3
        vacb = 0.5
        adult = [0, 0.9]
        nu_tra = [1] * 6
        nu_symp = [1] * 6
        nu_inf = [1] * 6
        nu_sev_h = [1] * 6
        nu_sev_d = [1] * 6

        VaccineParam = wm.VaccineParameters(
            model=model,
            vac=vac,
            vacb=vacb,
            adult=adult,
            nu_tra=nu_tra,
            nu_symp=nu_symp,
            nu_inf=nu_inf,
            nu_sev_h=nu_sev_h,
            nu_sev_d=nu_sev_d
        )

        self.assertEqual(VaccineParam.model, model)
        npt.assert_array_equal(VaccineParam.vac, np.array([3, 3]))
        npt.assert_array_equal(VaccineParam.vacb, np.array([0.5, 0.5]))
        npt.assert_array_equal(VaccineParam.adult, np.array([0, 0.9]))
        npt.assert_array_equal(VaccineParam.nu_tra, np.ones(6))
        npt.assert_array_equal(VaccineParam.nu_symp, np.ones(6))
        npt.assert_array_equal(VaccineParam.nu_inf, np.ones(6))
        npt.assert_array_equal(VaccineParam.nu_sev_h, np.ones(6))
        npt.assert_array_equal(VaccineParam.nu_sev_d, np.ones(6))

        with self.assertRaises(TypeError):
            model1 = {'0': 0}

            wm.VaccineParameters(
                model=model1,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac={'0.9': 0},
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=[[0.4], [0.3]],
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=[0.5, 0.9, 1],
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(TypeError):
            wm.VaccineParameters(
                model=model,
                vac=[0.5, '0.2'],
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=[-3, 0.2],
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb={'0.9': 0},
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=[[0.4], [0.3]],
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=[0.5, 0.9, 1],
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(TypeError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=[0.5, '0.2'],
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=[-3, 0.2],
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=[[0], [0.9]],
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=[0, 0.9, 1],
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(TypeError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=[0, '0.9'],
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=[1, -0.9],
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=[0, 1.9],
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra={'0.9': 0},
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=[[0.4], [0.3], [1], [1], [1], [1]],
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=[0.5, 0.9, 1, 1, 1, 1, 1],
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(TypeError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=[0.5, '0.2', 1, 1, 1, 1],
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=[-3, 0.2, 1, 1, 1, 1],
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp={'0.9': 0},
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=[[0.4], [0.3], [1], [1], [1], [1]],
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=[0.5, 0.9, 1, 1, 1, 1, 1],
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(TypeError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=[0.5, '0.2', 1, 1, 1, 1],
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=[-3, 0.2, 1, 1, 1, 1],
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf={'0.9': 0},
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=[[0.4], [0.3], [1], [1], [1], [1]],
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=[0.5, 0.9, 1, 1, 1, 1, 1],
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(TypeError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=[0.5, '0.2', 1, 1, 1, 1],
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=[-3, 0.2, 1, 1, 1, 1],
                nu_sev_h=nu_sev_h,
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h={'0.9': 0},
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=[[0.4], [0.3], [1], [1], [1], [1]],
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=[0.5, 0.9, 1, 1, 1, 1, 1],
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(TypeError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=[0.5, '0.2', 1, 1, 1, 1],
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=[-3, 0.2, 1, 1, 1, 1],
                nu_sev_d=nu_sev_d
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d={'0.9': 0}
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=[[0.4], [0.3], [1], [1], [1], [1]]
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=[0.5, 0.9, 1, 1, 1, 1, 1]
            )

        with self.assertRaises(TypeError):
            wm.VaccineParameters(
                model=model,
                vvac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_dac=[0.5, '0.2', 1, 1, 1, 1]
            )

        with self.assertRaises(ValueError):
            wm.VaccineParameters(
                model=model,
                vac=vac,
                vacb=vacb,
                adult=adult,
                nu_tra=nu_tra,
                nu_symp=nu_symp,
                nu_inf=nu_inf,
                nu_sev_h=nu_sev_h,
                nu_sev_d=[-3, 0.2, 1, 1, 1, 1]
            )

    def test__call__(self):
        model = examplemodel

        # Set vaccine-specific parameters
        vac = [3, 2]
        vacb = [0.5, 1]
        adult = [0, 0.9]
        nu_tra = [1] * 6
        nu_symp = [1] * 6
        nu_inf = [1] * 6
        nu_sev_h = [1] * 6
        nu_sev_d = [1] * 6

        VaccineParam = wm.VaccineParameters(
            model=model,
            vac=vac,
            vacb=vacb,
            adult=adult,
            nu_tra=nu_tra,
            nu_symp=nu_symp,
            nu_inf=nu_inf,
            nu_sev_h=nu_sev_h,
            nu_sev_d=nu_sev_d
        )

        self.assertEqual(
            VaccineParam(),
            [[3, 2], [0.5, 1], [0, 0.9]] + [[1] * 6] * 5)

#
# Test ParametersController Class
#


class TestParametersController(unittest.TestCase):
    """
    Test the 'ParametersController' class.
    """
    def test__init__(self):
        model = examplemodel

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=1
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[500, 600] + [0, 0] * 5, [700, 800] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[10, 20] * 5 + [0, 0], [0, 5] * 5 + [0, 0]],
            infectives_asym_IC=[[15, 10] * 5 + [0, 0], [30, 10] * 5 + [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0.02, 0],
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
            vac=[3, 3],
            vacb=[0.5, 0.5],
            adult=[0, 0.9],
            nu_tra=[1] * 6,
            nu_symp=[1] * 6,
            nu_inf=[1] * 6,
            nu_sev_h=[1] * 6,
            nu_sev_d=[1] * 6
        )

        # Set all parameters in the controller
        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters
        )

        self.assertEqual(parameters.model, model)
        self.assertEqual(parameters.ICs, ICs_parameters)
        self.assertEqual(parameters.regional_parameters, regional_parameters)
        self.assertEqual(parameters.disease_parameters, disease_parameters)
        self.assertEqual(parameters.transmission_parameters,
                         transmission_parameters)
        self.assertEqual(parameters.simulation_parameters,
                         simulation_parameters)
        self.assertEqual(parameters.vaccine_parameters(),
                         [[3, 3], [0.5, 0.5], [0, 0.9],
                         [1] * 6, [1] * 6, [1] * 6, [1] * 6, [1] * 6])
        npt.assert_array_equal(parameters.soc_dist_parameters(),
                               np.array([1, 1]))

        # Set social distancing parameters
        soc_dist_parameters = wm.SocDistParameters(
            model=model,
            phi=0.2)

        parameters = wm.ParametersController(
            model=model,
            regional_parameters=regional_parameters,
            ICs_parameters=ICs_parameters,
            disease_parameters=disease_parameters,
            transmission_parameters=transmission_parameters,
            simulation_parameters=simulation_parameters,
            vaccine_parameters=vaccine_parameters,
            soc_dist_parameters=soc_dist_parameters
        )

        self.assertEqual(parameters.soc_dist_parameters,
                         soc_dist_parameters)

        with self.assertRaises(TypeError):
            model1 = 0.3

            wm.ParametersController(
                model=model1,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters
            )

        with self.assertRaises(TypeError):
            regional_parameters1 = 0

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters1,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters
            )

        with self.assertRaises(ValueError):
            regional_parameters1 = wm.RegParameters(
                model=examplemodel2,
                region_index=1
            )

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters1,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters
            )

        with self.assertRaises(TypeError):
            ICs1 = '0'

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs=ICs1,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters
            )

        with self.assertRaises(ValueError):
            ICs1 = wm.ICs(
                model=examplemodel2,
                susceptibles_IC=[
                    [500, 600] + [0, 0] * 5, [700, 800] + [0, 0] * 5],
                exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
                exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
                exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
                exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
                exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
                infectives_sym_IC=[
                    [10, 20] * 5 + [0, 0], [0, 5] * 5 + [0, 0]],
                infectives_asym_IC=[
                    [15, 10] * 5 + [0, 0], [30, 10] * 5 + [0, 0]],
                recovered_IC=[[0, 0], [0, 0]]
            )

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs1,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters
            )

        with self.assertRaises(TypeError):
            disease_parameters1 = [0]

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters1,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters
            )

        with self.assertRaises(ValueError):
            disease_parameters1 = wm.DiseaseParameters(
                model=examplemodel2,
                d=0.4 * np.ones(len(age_groups)),
                tau=0.4,
                we=1,
                omega=1
            )

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters1,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters
            )

        with self.assertRaises(TypeError):
            transmission_parameters1 = [0]

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters1,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters
            )

        with self.assertRaises(ValueError):
            transmission_parameters1 = wm.Transmission(
                model=examplemodel2,
                beta=0.5 * np.ones(len(age_groups)),
                alpha=0.5,
                gamma=1
            )

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters1,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters
            )

        with self.assertRaises(TypeError):
            simulation_parameters1 = {'0': 0}

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters1,
                vaccine_parameters=vaccine_parameters
            )

        with self.assertRaises(ValueError):
            simulation_parameters1 = wm.SimParameters(
                model=examplemodel2,
                method='RK45',
                times=np.arange(1, 20.5, 0.5).tolist()
            )

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters1,
                vaccine_parameters=vaccine_parameters
            )

        with self.assertRaises(TypeError):
            vaccine_parameters1 = {'0': 0}

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters1,
                soc_dist_parameters=soc_dist_parameters
            )

        with self.assertRaises(ValueError):
            vaccine_parameters1 = wm.VaccineParameters(
                model=examplemodel2,
                vac=3,
                vacb=0.5,
                adult=[0, 0.9],
                nu_tra=[1] * 6,
                nu_symp=[1] * 6,
                nu_inf=[1] * 6,
                nu_sev_h=[1] * 6,
                nu_sev_d=[1] * 6
            )

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters1,
                soc_dist_parameters=soc_dist_parameters
            )

        with self.assertRaises(TypeError):
            soc_dist_parameters1 = {'0': 0}

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters,
                soc_dist_parameters=soc_dist_parameters1
            )

        with self.assertRaises(ValueError):
            soc_dist_parameters1 = wm.SocDistParameters(
                model=examplemodel2,
                phi=0.2)

            wm.ParametersController(
                model=model,
                regional_parameters=regional_parameters,
                ICs_parameters=ICs_parameters,
                disease_parameters=disease_parameters,
                transmission_parameters=transmission_parameters,
                simulation_parameters=simulation_parameters,
                vaccine_parameters=vaccine_parameters,
                soc_dist_parameters=soc_dist_parameters1
            )

    def test__call__(self):
        model = examplemodel

        # Set regional and time dependent parameters
        regional_parameters = wm.RegParameters(
            model=model,
            region_index=1
        )

        # Set ICs parameters
        ICs_parameters = wm.ICs(
            model=model,
            susceptibles_IC=[[500, 600] + [0, 0] * 5, [700, 800] + [0, 0] * 5],
            exposed1_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed2_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed3_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed4_IC=[[0, 0] * 6, [0, 0] * 6],
            exposed5_IC=[[0, 0] * 6, [0, 0] * 6],
            infectives_sym_IC=[[10, 20] * 5 + [0, 0], [0, 5] * 5 + [0, 0]],
            infectives_asym_IC=[[15, 10] * 5 + [0, 0], [30, 10] * 5 + [0, 0]],
            recovered_IC=[[0, 0], [0, 0]]
        )

        # Set disease-specific parameters
        disease_parameters = wm.DiseaseParameters(
            model=model,
            d=0.4 * np.ones(len(age_groups)),
            tau=0.4,
            we=[0.02, 0.02, 0],
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
            vac=3,
            vacb=0.5,
            adult=[0, 0.9],
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
            soc_dist_parameters=soc_dist_parameters
        )

        self.assertEqual(
            parameters(),
            [1, 500, 600, 700, 800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
             10, 20, 0, 5, 10, 20, 0, 5,
             10, 20, 0, 5, 10, 20, 0, 5, 10, 20, 0, 5, 0, 0, 0, 0,
             15, 10, 30, 10, 15, 10, 30, 10, 15, 10, 30, 10,
             15, 10, 30, 10, 15, 10, 30, 10, 0, 0, 0, 0, 0, 0, 0, 0,
             0.5, 0.5, 0.5, 1, 1, 0.4, 0.4, 0.4, 0.02, 0.02, 0, 1, 'RK45',
             False]
        )
