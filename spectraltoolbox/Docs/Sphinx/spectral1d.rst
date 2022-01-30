Spectral 1D
-----------

Implementation of Spectral Methods in 1 dimension.

Available polynomials:
    * :ref:`ref_jacobi` or ``Spectral1D.JACOBI``
    * Hermite Physicist or ``Spectral1D.HERMITEP``
    * Hermite Function or ``Spectral1D.HERMITEF``
    * Hermite Probabilistic or ``Spectral1D.HERMITEP_PROB``
    * Laguerre Polynomial or ``Spectral1D.LAGUERREP``
    * Laguerre Function or ``Spectral1D.LAGUERREF``
    * ORTHPOL package (generation of recursion coefficients using :cite:`Gautschi1994`)  or ``Spectral1D.ORTHPOL``

Available quadrature rules (related to selected polynomials):
    * Gauss or ``Spectral1D.GAUSS``
    * Gauss-Lobatto or ``Spectral1D.GAUSSLOBATTO``
    * Gauss-Radau or ``Spectral1D.GAUSSRADAU``

Available quadrature rules (without polynomial selection):
    * Kronrod-Patterson on the real line or ``Spectral1D.KPN`` (function ``Spectral1D.kpn(n)``)
    * Kronrod-Patterson uniform or ``Spectral1D.KPU`` (function ``Spectral1D.kpu(n)``)
    * Clenshaw-Curtis or ``Spectral1D.CC`` (function ``Spectral1D.cc(n)``)
    * Fejer's or ``Spectral1D.FEJ`` (function ``Spectral1D.fej(n)``)

.. _ref_jacobi:

Jacobi Polynomials
^^^^^^^^^^^^^^^^^^

Jacobi polynomials are defined on the domain :math:`\Omega=[-1,1]` by the recurrence relation

.. math:: 
    
    xP^{(\alpha,\beta)}_n(x) =    & \frac{2(n+1)(n+\alpha+\beta+1)}{(2n+\alpha+\beta+1)(2n+\alpha+\beta+2)} P^{(\alpha,\beta)}_{n+1}(x) \\
                                    & + \frac{\beta^2 - \alpha^2}{(2n+\alpha+\beta)(2n+\alpha+\beta+2)} P^{(\alpha,\beta)}_{n}(x) \\
                                    & + \frac{2(n+\alpha)(n+\beta)}{(2n+\alpha+\beta)(2n+\alpha+\beta+1)} P^{(\alpha,\beta)}_{n-1}(x)

with weight function

.. math::
    
    w(x;\alpha,\beta) = \frac{\Gamma(\alpha+\beta+2)}{2^{\alpha+\beta+1}\Gamma(\alpha+1)\Gamma(\beta+1)}(1-x)^\alpha (1+x)^\beta

.. note::
    
    In probability theory, the Beta distribution is defined on :math:`\Psi=[0,1]` and its the Probability Distribution Function is
    
    .. math::
        
        \rho_B(x;\alpha,\beta) = \frac{\Gamma(\alpha+\beta)}{\Gamma(\alpha)\Gamma(\beta)} x^{\alpha-1} (1-x)^{(\beta-1)}
    
    The relation betwen :math:`w(x;\alpha,\beta)` and :math:`\rho_B(x;\alpha,\beta)` for :math:`x \in \Psi` is
    
    .. math::
        
        \rho_B(x;\alpha,\beta) = 2 * w(2*x-1;\beta-1,\alpha-1)
    
    For example:
    
    >>> from scipy import stats
    >>> plot(xp,stats.beta(3,5).pdf(xp))
    >>> plot(xp,2.*Bx(xx,4,2),'--')
    >>> plot(xp,stats.beta(3,8).pdf(xp))
    >>> plot(xp,2.*Bx(xx,7,2),'--')
