# OpenTorsion: Open-Source Backend for Torsional Vibration Analysis

[![PyPi Version](https://img.shields.io/pypi/v/opentorsion.svg)](https://pypi.org/project/opentorsion)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/opentorsion.svg)](https://pypi.org/pypi/opentorsion/)
[![GitHub stars](https://img.shields.io/github/stars/Aalto-Arotor/openTorsion.svg)](https://github.com/Aalto-Arotor/openTorsion)
[![PyPi downloads](https://img.shields.io/pypi/dm/opentorsion.svg)](https://pypistats.org/packages/opentorsion)
[![main branch unittests](https://github.com/aalto-arotor/opentorsion/actions/workflows/unittest.yml/badge.svg?branch=main)](https://github.com/Aalto-Arotor/openTorsion/tree/main/opentorsion/tests)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![MIT license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/Aalto-Arotor/openTorsion/blob/main/LICENSE)
<img src="./figures/opentorsion_logo.png" width="60">

![Small-scale marine thruster testbench](./figures/testbench_all.png "Small-scale marine thruster testbench")
Open-source software for torsional vibration analysis. Supported features include
* finite element model creation based on dimensions or datasheet specifications
* natural frequency calculation
* eigenmodes
* forced response analysis
* time-stepping simulation

## Introduction
OpenTorsion includes tools for creating shaft-line finite element models and calculation of torsional response in time or frequency domain.
Please note that the software is still in development and the authors are not able to responsibility for the functioning or effects of future changes. See the license for more information.

## Documentation

[openTorsion documentation](https://aalto-arotor.github.io/openTorsion/)

## Quickstart
Install openTorsion by running the command ```pip install opentorsion```. Folder ```opentorsion``` includes the software. Folder ```examples``` contains scripts to run example powertrains and analyses.

Make sure you have pip3 & pipenv installed in your system. Then simply running ```pipenv install``` will invoke the config files and install the necessary files in your pipenv.

Two examples are found in opentorsion/examples folder.

## Contact
The main developers are Sampo Laine and Urho Hakonen from Arotor lab at Aalto University Finland.
https://www.aalto.fi/en/department-of-mechanical-engineering/aalto-arotor-lab

For questions regarding the software please contact arotor.software@aalto.fi
