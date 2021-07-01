# OpenTorsion
![main branch unittests](https://github.com/aalto-arotor/opentorsion/actions/workflows/unittest.yml/badge.svg?branch=main)

Open source library for creating torsional finite element models.

## Quickstart
Make sure you have pip3 & pipenv installed in your system. Then simply running ```pipenv install``` will invoke the config files and install the necessary files in your pipenv.

## Tests
Running ```pipenv run python -m unittest``` will run the the tests locally.

## Coverage report
First generate the ```.coverage``` file by running ```pipenv run coverage run -m unittest```. You can access the report easily by running ```pipenv run coverage report``` 

## TODO
The coverage reports should be ran automatically as a workflow. [Additional information](https://about.codecov.io/blog/python-code-coverage-using-github-actions-and-codecov/)