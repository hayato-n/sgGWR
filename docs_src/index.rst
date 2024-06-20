.. sgGWR documentation master file, created by
   sphinx-quickstart on Thu Dec 14 11:56:57 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to sgGWR's documentation!
=================================

About sgGWR
=================================
Welcome to **sgGWR**! sgGWR (Stochastic Gradient approach for Geographically Weighted Regression) is scalable bandwidth calibration software for GWR.

Installation
=================================

We recommend installing JAX (https://github.com/google/jax) package for efficient computation.
To install sgGWR with JAX, please execute the following on your terminal.

.. code-block ::

   pip install sgGWR[jax]

If you cannot install JAX (e.g., Windows users), you can omit `[jax]` option.


.. code-block ::

   pip install sgGWR



Reference
=================================

Please cite the following article:

- Nishi, H., & Asami, Y. (2024). Stochastic gradient geographical weighted regression (sgGWR): Scalable bandwidth optimization for geographically weighted regression. International Journal of Geographical Information Science, 38(2), 354–380. https://doi.org/10.1080/13658816.2023.2285471


.. toctree::
   :maxdepth: 1
   :caption: Tutorials:

   examples/introduction.ipynb
   examples/init_bandwidth.ipynb

.. toctree::
   :maxdepth: 2
   :caption: Examples:

   examples/*

.. toctree::
   :maxdepth: 3
   :caption: Package References:

   sgGWR
   sgGWR.optimizers


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
