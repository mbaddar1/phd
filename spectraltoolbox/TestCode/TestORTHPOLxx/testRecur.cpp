#include <iostream>
#include <ORTHPOLxx/ORTHPOLPP.hpp>

main()
{
  int n = 4;
  int ipoly = 1;
  float al = 0.0;
  float be = 0.0;
  float a[n];
  float b[n];
  double dal = 0.0;
  double dbe = 0.0;
  double da[n];
  double db[n];
  int ierr;

  orthpol::recur(n, ipoly, al, be, a, b, ierr);
  orthpol::drecur(n, ipoly, dal, dbe, da, db, ierr);

  if (ierr == 0){
    // Print coeficients
    std::cout << "a" << std::endl;
    for (int i=0;i<n;i++){
      std::cout << a[i] << "\t" << da[i] << std::endl;
    }
    std::cout << "b" << std::endl;
    for (int i=0;i<n;i++){
      std::cout << b[i] << "\t" << db[i] << std::endl;
    }
  } else {
    std::cout << "Error: " << ierr;
  }
}
