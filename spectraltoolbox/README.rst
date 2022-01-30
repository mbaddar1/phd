================
Spectral Toolbox
================

The SpectralToolbox is a collection of tools useful for spectral approximation methods in one or more dimensions.
It include the construction of traditional orthogonal polynomials. Additionally one can construct orthogonal polynomials with respect to a selected measure.

Status
======

`PyPi <https://pypi.python.org/pypi/SpectralToolbox/>`_:

.. image:: http://southpacific.no-ip.org:8080/buildStatus/icon?job=pypi-SpectralToolbox
   :target: http://southpacific.no-ip.org:8080/buildStatus/icon?job=pypi-SpectralToolbox

`LaunchPad <https://launchpad.net/spectraltoolbox>`_:

.. image:: http://southpacific.no-ip.org:8080/buildStatus/icon?job=SpectralToolbox
   :target: http://southpacific.no-ip.org:8080/buildStatus/icon?job=SpectralToolbox

`TestPyPi <https://testpypi.python.org/pypi/SpectralToolbox/>`_:

.. image:: http://southpacific.no-ip.org:8080/buildStatus/icon?job=testpypi-SpectralToolbox
   :target: http://southpacific.no-ip.org:8080/buildStatus/icon?job=testpypi-SpectralToolbox

Description
===========

Implementation of Spectral Methods in N dimension.

Available polynomials:
    * Jacobi
    * Hermite Physicists'
    * Hermite Physicists' Function
    * Hermite Probabilists'
    * Hermite Probabilists' Function
    * Laguerre Polynomial
    * Laguerre Function
    * Fourier
    * ORTHPOL package (generation of recursion coefficients using [1]_)

Available quadrature rules (related to selected polynomials):
    * Gauss
    * Gauss-Lobatto
    * Gauss-Radau

Available quadrature rules (without polynomial selection):
    * Kronrod-Patterson on the real line
    * Kronrod-Patterson uniform
    * Clenshaw-Curtis
    * Fejer's

Requirements
============

  * `numpy <https://pypi.python.org/pypi/numpy>`_
  * `scipy <https://pypi.python.org/pypi/scipy>`_
  * `Sphinx <https://pypi.python.org/pypi/sphinx>`_
  * `sphinxcontrib-bibtex <https://pypi.python.org/pypi/sphinxcontrib-bibtex>`_
  * `orthpol <https://pypi.python.org/pypi/orthpol>`_ >= 0.2.2

Installation
============

We reccommend to work in a virtual environment using `virtualenv <https://virtualenv.readthedocs.org/en/latest/>`_, or on the system python installation. The use of alternative virtual environment systems (such as `Anaconda <https://www.continuum.io/why-anaconda>`_) is not guaranteed to be automatically supported, thus a manual installation is suggested in such case.

Make sure to have an up-to-date version of pip:

   $ pip install --upgrade pip

Automatic installation
----------------------

Run the command:

   $ pip install --upgrade numpy
   $ pip install --upgrade SpectralToolbox

Manual installation (using pip)
-------------------------------

Install the following dependencies separately:

   $ pip install <package>

where <package> are the Python dependencies as listed in Requirements and X.X.X is the current revision version.

You should intall the `orthpol <https://pypi.python.org/pypi/orthpol>`_ package. This dependency is required since v. 0.2.0. The installation might require you to tweak some flags for the compiler (with gcc nothing should be needed).

   $ pip install --no-binary :all: orthpol

Finally you can install the toolbox by:

   $ pip install --no-binary :all: SpectralToolbox

Manual installation (from source files)
---------------------------------------

Note: This method may apply also to virtual environment systems different from `virtualenv <https://virtualenv.readthedocs.org/en/latest/>`_.

download and install each dependency manually with the following commands:

   $ pip download <package>

   $ tar xzf <package>-X.X.X.tar.gz

   $ cd <package>-X.X.X

   $ python setup.py install

   $ cd ..

where <package> are the Python dependencies as listed in Requirements and X.X.X is the current revision version.


References
==========
.. [1] W. Gautschi, "Algorithm 726: ORTHPOL -- a package of routines for generating orthogonal polynomials and Gauss-type quadrature rules". ACM Trans. Math. Softw., vol. 20, issue 1, pp. 21-62, 1994


Change Log
==========

0.1.0:
  * Implementation of Poly1D, PolyND, and additional quadrature rules

0.2.0:
  * New interface for Spectral1D. 
  * All polynomials are now classes.
  * Complete integration of `orthpol <https://pypi.python.org/pypi/orthpol>`_

0.2.7:
  * Python3 support. And fixed installation procedure.

0.2.8:
  * Bug fix from `orthpol <https://pypi.python.org/pypi/orthpol>`_ package

0.2.11:
  * Added function ``generate`` for the generation of polynomials from type and parameters.

0.2.27
  * Added class ``ConstantExtendedHermiteProbabilistsFunction``, used for external projects.

0.2.38
  * Added functions ``from_xml_element`` in order to generate basis from XML structures.

0.2.39
  * Added class handling algebraic operations between polynomials (class ``SquaredOrthogonalPolynomial``)

0.2.41
  * Added algebraic function class ``SquaredConstantExtendedHermitePhysicistsFunction``

0.2.42
  * Added algebraic function class ``PositiveDefiniteSquaredConstantExtendedHermitePhysicistsFunction``

0.2.45-46
  * Exact factorials in AlgebraicPolynomilas.

0.2.47
  * Memoization of coefficients
