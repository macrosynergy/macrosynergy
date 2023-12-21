.. _getting_started:

Getting Started
===============

How to get started with the Macrosynergy package in 5-minutes.

Installation
------------

The easiest method for installing the package is to use the PyPI installation method:

Installing from PyPI
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install macrosynergy

Alternatively, if you want to install the latest development version, you can install directly from GitHub:

Installing from GitHub/Source
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    pip install git+https://github.com/macrosynergy/macrosynergy@develop

This extends to any branch; not just the `develop` branch.

Building from Source
~~~~~~~~~~~~~~~~~~~~

If you want to build the package from source, you would need to clone the repository and run the build command:

.. code-block:: bash

    git clone https://github.com/macrosynergy/macrosynergy
    cd macrosynergy
    python setup.py build

Usage
-----

The package is designed to be used in conjunction with JPMaQS Quantamental Data from the 
J.P. Morgan DataQuery API. If you do not have access to the DataQuery API, you can still 
use the package with simulated data from `macrosynergy.management.simulate <../gen_rsts/macrosynergy.management.simulate.html>`_.
