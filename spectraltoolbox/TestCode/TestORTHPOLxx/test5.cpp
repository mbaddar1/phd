#include <iostream>
#include <boost/format.hpp>
#include <ORTHPOLxx/ORTHPOLPP.hpp>

int ncapm = 41;
double da[41];
double db[41];
double e[41];

void dquad_func(int& n, double* x, double* w, int& i, int& ierr)
{
  int E = 3;
  double epsma = d1mach_(E);
  dgauss_(n, da, db, epsma, x, w, ierr, e);
  for (int k = 0; k < n; k++){
    w[k] = w[k] / db[0];
  }
}

main()
{
  orthpol::set_dquad_callback(dquad_func);

  // Params for mcdis
  int n = 40;
  int ncapm = 41;
  int mc = 1;
  int mp = 1;
  int iq = 1;
  int idelta = 2;
  int irout = 1;
  int E = 3;
  double epsma = d1mach_(E);
  double eps = 5000.*epsma;
  double xp[mp];
  double yp[mp];
  int finl;
  int finr;
  double endl[mc];
  double endr[mc];
  double xfer[1];
  double wfer[1];
  double alpha[n];
  double beta[n];
  int ncap;
  int kount;
  int ierr;
  int ie;
  double be[n];
  double x[ncapm];
  double w[ncapm];
  double xm[mc*ncapm+mp];
  double wm[mc*ncapm+mp];
  double p0[mc*ncapm+mp];
  double p1[mc*ncapm+mp];
  double p2[mc*ncapm+mp];

  // Extra params for this test
  double dy, dalj, dbej, dalpbe, daex[n], dbex[n], dnum, dd, dkm1, dden;
  double y = .25;
  double erra, errb, erram, errbm, alpham, betam;
  int kam, kbm, km1;

  std::ios_base::fmtflags original_flags = std::cout.flags();
  std::cout << std::endl;

  xp[0] = -1.;
  for (int iy = 1; iy <= 5; iy++){
    y = 2.*y;
    dy = y;
    yp[0] = y;
    std::cout << "\t" << "y = " << y << std::endl;
    std::cout << "  alj  bej    erra        errb       alpham     betam       ka   kb ierr ie   it" << std::endl;
    for (int ia = 1; ia <= 10; ia++){
      dalj = -1. + .2 * (double)ia;
      for (int ib = 1; ib <= 10; ib++){
	dbej = -1. + .2 * (double)ib;
	dalpbe = dalj + dbej;
	
	// Jacobi recursion coeff
	int ipoly = 6;
	orthpol::drecur(ncapm,ipoly,dalj,dbej, da, db, ierr);

	// Recursion coeff
	orthpol::dmcdis(n, ncapm, mc, mp, xp, yp, eps, iq, idelta, irout, finl, finr, endl, endr, xfer, wfer, alpha, beta, ncap, kount, ierr, ie, be, x, w, xm, wm, p0, p1, p2);

	// Exact coefficients
	daex[0] = (da[0] - dy)/(1.+dy);
	dbex[0] = 1. + dy;
	erra = abs(alpha[0]-daex[0]);
	if (abs(daex[0]) > eps) erra = erra / abs(daex[0]);
	errb = abs((beta[0]-dbex[0])/dbex[0]);
	erram = erra;
	errbm = errb;
	alpham = alpha[0];
	betam = beta[0];
	kam = 0; kbm =0;
	dnum = 1. + dy;
	dd = 1.;
	for (int k = 2; k <= n; k++){
	  km1 = k-1;
	  dkm1 = (double) km1;
	  dden = dnum;
	  if (k > 2) dd = (dbej + dkm1) * (dalpbe + dkm1) * dd / ((dalj + dkm1 - 1.) * (dkm1 - 1.));
	  dnum = (1. + (dbej + dkm1 + 1.) * (dalpbe + dkm1 + 1.) * dy * dd / (dkm1*(dalj+dkm1))) / (1. + dy * dd);
	  daex[km1] = da[km1] + 2. * dkm1 * (dkm1+dalj) * (dnum - 1.) / ((dalpbe + 2. * dkm1) * (dalpbe + 2. * dkm1 + 1.)) + 2. * (dbej + dkm1 + 1.) * (dalpbe + dkm1 + 1.) * ((1./dnum) -1.)/((dalpbe + 2. * dkm1 + 1.)*(dalpbe + 2. * dkm1 + 2.));
	  dbex[km1] = dnum * db[km1] / dden;
	  erra = abs(alpha[km1] - daex[km1]);
	  if (abs(daex[km1]) > eps) erra = erra / abs(daex[km1]);
	  errb = abs((beta[km1] - dbex[km1])/ dbex[km1]);
	  if (erra > erram) {
	    erram = erra;
	    alpham = alpha[km1];
	    kam = km1;
	  }
	  if (errb > errbm) {
	    errbm = errb;
	    betam = beta[km1];
	    kbm = km1;
	  }
	}

	// Print results
	std::cout << boost::format(" %5.2f %5.2f %11.4e %11.4e %9.6f %9.6f %4i %4i %4i %4i %2i") % dalj % dbej % erram % errbm % alpham % betam % kam % kbm % ierr % ie % kount << std::endl;
      }
      std::cout << std::endl;
    }
  }
}
