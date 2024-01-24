# warwick-covid-transmission

[![Multiple python versions](https://github.com/I-Bouros/warwick-covid-transmission/actions/workflows/python-version-unittests.yml/badge.svg)](https://github.com/I-Bouros/warwick-covid-transmission/actions/workflows/python-version-unittests.yml)
[![Multiple OS](https://github.com/I-Bouros/warwick-covid-transmission/actions/workflows/os-unittests.yml/badge.svg)](https://github.com/I-Bouros/warwick-covid-transmission/actions/workflows/os-unittests.yml)
[![Copyright License](https://github.com/I-Bouros/warwick-covid-transmission/actions/workflows/check-copyright.yml/badge.svg)](https://github.com/I-Bouros/warwick-covid-transmission/actions/workflows/check-copyright.yml)
[![Documentation Status](https://readthedocs.org/projects/warwick-covid-transmission/badge/?version=latest)](https://warwick-covid-transmission.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/I-Bouros/warwick-covid-transmission/branch/main/graph/badge.svg?token=SNHCUJIS3B)](https://codecov.io/gh/I-Bouros/warwick-covid-transmission)
[![Style (flake8)](https://github.com/I-Bouros/warwick-covid-transmission/actions/workflows/flake8-style-test.yml/badge.svg)](https://github.com/I-Bouros/warwick-covid-transmission/actions/workflows/flake8-style-test.yml)

This is a reimplementation of the Warwick-Lancaster model. This work was commissioned by the World Health Organisation (WHO) Strategic Advisory Group of Experts on Immunisation (SAGE) Covid-19 Working Group. Below we include a model schematic:

![Warwick-Lancaster model conceptualisation](./images/Warwick_model.pdf)

All features of our software are described in detail in our
[full API documentation](https://warwick-covid-transmission.readthedocs.io/en/latest/).

More details on epidemiological models and inference can be found in these
papers:

## References
[1] Moore, S., Hill, E.M., Dyson, L. et al. [Retrospectively modeling the effects of increased global vaccine sharing on the COVID-19 pandemic](https://doi.org/10.1038/s41591-022-02064-y). Nat Med 28, 2416–2423 (2022).

[2] Bouros, I., Thompson, R.N., Keeling, M.J., Hill, E.M., Moore, S., [Warwick-Lancaster Global Covid-19 Model](https://ssrn.com/abstract=4654753 or http://dx.doi.org/10.2139/ssrn.4654753). 9TH INTERNATIONAL CONFERENCE ON INFECTIOUS DISEASE DYNAMICS:P1.082 (2023),

## Installation procedure
***
One way to install the module is to download the repositiory to your machine of choice and type the following commands in the terminal. 
```bash
git clone https://github.com/I-Bouros/warwick-covid-transmission.git
cd ../path/to/the/file
```

A different method to install this is using `pip`:

```bash
pip install -e .
```

## Usage

```python
import epimodels

# create a contact matrix using mobility data e.g. from a POLYMOD matrix
epimodels.ContactMatrix(age_groups, polymod_matrix)
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/)
