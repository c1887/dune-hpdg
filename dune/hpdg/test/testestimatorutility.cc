#if HAVE_CONFIG
#include "config.h"
#endif

#include <vector>

#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/exceptions.hh>
#include <dune/common/test/testsuite.hh>

#include <dune/hpdg/estimators/utility.hh>

int main(int argc, char** argv) {

  Dune::MPIHelper::instance(argc, argv);

  using namespace Dune;
  using namespace std;

  TestSuite suite;

  // test quantile:
  auto vec = vector<int>(100);
  iota(begin(vec), end(vec), 1); // vec= [1,2,...,100]

  double quant = 0.5;
  // We expect 51 as the 50%-quantile (median) because
  // our implementation only returns actual values (no averages)
  suite.check(HPDG::quantile(vec, quant) == 51, "Check quantile function") 
    << "Expected 51, got " << HPDG::quantile(vec, quant);

  // test fraction
  
  // the sum over all elements is n(n+1)/2, hence for n=100 it is 5050.
  // Say we want the 10% fraction, that is the largest element v, such that
  // the sum over all elements bigger or equal to v is larger or equal to 505.
  // A small calculation shows that
  // 5050 - sum 1..95 = 490
  // 5050 - sum 1..94 = 585
  // Hence 94 is the value we expect:
  suite.check(HPDG::fraction(vec, 0.1), "Check fraction")
    << "Expected 94, got "<<  HPDG::fraction(vec, 0.1);

}
