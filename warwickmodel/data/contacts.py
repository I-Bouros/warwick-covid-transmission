#
# This file is part of WARWICKMODEL
# (https://github.com/I-Bouros/warwick-covid-transmission.git) which is
# released under the MIT license. See accompanying LICENSE for copyright
# notice and full license details.
#
"""Processing script for the contact matrices from [1]_.

It computes the baseline and time-dependent country-specific contact matrices
which are then stored in separate csv files.

References
----------
.. [1] Prem K, Cook AR, Jit M (2017) Projecting social contact matrices in 152
       countries using contact surveys and demographic data. PLOS Computational
       Biology 13(9): e1005697.
       https://doi.org/10.1371/journal.pcbi.1005697
"""

import os
import pandas as pd
import numpy as np


def read_contact_matrices(
        file_index: int = 2,
        state: str = 'United Kingdom of Great Britain'):
    """
    Read the baseline contact matices for different activities recorded
    for the given state from the appropriate Excel file.

    Parameters
    ----------
    file_index : int
        Index of the file containg the baseline contact matrices
        used in the model.
    state : str
        Name of the country whose the baseline contact matrices are used in
        the model.

    Retruns
    -------
    list of pandas.Dataframe
        List of the baseline contact matices for each activitiy recorded
        for different for the given state.

    """
    header_opt = [0, None]

    # Select contact matrices from the given state and activity
    path = os.path.join(
            os.path.dirname(__file__), 'raw_contact_matrices/')
    school = pd.read_excel(
        os.path.join(path, 'MUestimates_school_{}.xlsx').format(file_index),
        sheet_name=state, header=header_opt[file_index - 1]).to_numpy()
    home = pd.read_excel(
        os.path.join(path, 'MUestimates_home_{}.xlsx').format(file_index),
        sheet_name=state, header=header_opt[file_index - 1]).to_numpy()
    work = pd.read_excel(
        os.path.join(path, 'MUestimates_work_{}.xlsx').format(file_index),
        sheet_name=state, header=header_opt[file_index - 1]).to_numpy()
    others = pd.read_excel(
        os.path.join(path, 'MUestimates_other_locations_{}.xlsx').format(
            file_index),
        sheet_name=state, header=header_opt[file_index - 1]).to_numpy()

    return school, home, work, others


def change_age_groups(matrix: np.array):
    """
    Reprocess the contact matrix so that it has the appropriate age groups.

    Parameters
    ----------
    matrix : numpy.array
        Contact matrix with old age groups.

    Returns
    -------
    numpy.array
        New contact matrix with correct age groups.

    """
    new_matrix = np.empty((8, 8))

    ind_old = [
        np.array([0]),
        np.array([0]),
        np.array(range(1, 3)),
        np.array(range(3, 5)),
        np.array(range(5, 9)),
        np.array(range(9, 13)),
        np.array(range(13, 15)),
        np.array([15])]

    for i in range(8):
        for j in range(8):
            new_matrix[i, j] = np.mean(
                matrix[ind_old[i][:, None], ind_old[j]][:, None])

    return new_matrix


def main():
    """
    Combines the timelines of deviation percentages and baseline
    activity-specific contact matrices to get weekly, c-specific
    contact matrices.

    Returns
    -------
    csv
        Processed files for the baseline and country-specific time-dependent
        contact matrices for each different country found in the default file.

    """
    activity = ['school', 'home', 'work', 'others']

    all_countries = ['United Kingdom of Great Britain', 'France']
    all_ctry_codes = ['UK', 'FR']
    all_file_indeces = [2, 1]

    for file_index, country, ctry in zip(
            all_file_indeces, all_countries, all_ctry_codes):
        baseline_matrices = read_contact_matrices(file_index, country)
        baseline_contact_matrix = np.zeros_like(baseline_matrices[0])
        for ind, _ in enumerate(activity):
            baseline_contact_matrix += baseline_matrices[ind]

        # Transform recorded matrix of serial intervals to csv file
        path_ = os.path.join(
            os.path.dirname(__file__), 'final_contact_matrices/')
        path = os.path.join(
                path_,
                '{}.csv'.format(ctry))

        np.savetxt(
            path, change_age_groups(baseline_contact_matrix),
            delimiter=',')


if __name__ == '__main__':
    main()
