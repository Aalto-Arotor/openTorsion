# OpenTorsion

[![PyPi Version](https://img.shields.io/pypi/v/opentorsion.svg)](https://pypi.org/project/opentorsion)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/opentorsion.svg)](https://pypi.org/pypi/opentorsion/)
[![GitHub stars](https://img.shields.io/github/stars/Aalto-Arotor/openTorsion.svg)](https://github.com/Aalto-Arotor/openTorsion)
[![PyPi downloads](https://img.shields.io/pypi/dm/opentorsion.svg)](https://pypistats.org/packages/opentorsion)

[![main branch unittests](https://github.com/aalto-arotor/opentorsion/actions/workflows/unittest.yml/badge.svg?branch=main)](https://github.com/Aalto-Arotor/openTorsion/tree/main/opentorsion/tests)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Aalto-Arotor/openTorsion/blob/main/LICENSE)


Open source library for creating torsional finite element models.

## Documentation

[openTorsion documentation](https://aalto-arotor.github.io/openTorsion/)

## Quickstart
Make sure you have pip3 & pipenv installed in your system. Then simply running ```pipenv install``` will invoke the config files and install the necessary files in your pipenv.

## Tests
Running ```pipenv run python -m unittest``` will run the the tests locally.

## Coverage report
First generate the ```.coverage``` file by running ```pipenv run coverage run -m unittest```. You can access the report easily by running ```pipenv run coverage report``` 

## TODO
The coverage reports should be ran automatically as a workflow. [Additional information](https://about.codecov.io/blog/python-code-coverage-using-github-actions-and-codecov/)

<!--
badge for coverage
[![codecov](https://img.shields.io/codecov/c/github/Aalto-Arotor/openTorsion.svg)](https://codecov.io/gh/Aalto-Arotor/openTorsion)
-->