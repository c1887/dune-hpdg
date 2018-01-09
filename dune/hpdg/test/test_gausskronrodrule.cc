#include <config.h>

#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>

#include <dune/hpdg/geometry/quadraturerules/gausskronrod.hh>

using namespace Dune;

TestSuite test_GaussKronrod() {
  auto epsilon = 1e-14;
  TestSuite suite("test_GaussKronrod");

  int n = 2; // n Gauss-Legendre nodes + n+1 additonal nodes
  auto rule = Dune::HPDG::GaussKronrod1DRule(n);

  auto func = [](const auto& x) {return x*x;};

  double analyticIntegral = 1.0/3;

  // compute with quadrature rule
  double sum = 0.0;
  for (auto quad = rule.begin(); quad!= rule.end(); quad++)
    sum+=quad->weight()*func(quad->position());

  suite.check(std::abs(sum-analyticIntegral)<epsilon);


  /// compare with corresponding Gauss-Legendre rule.
  auto gl_rule = Dune::QuadratureRules<double,1>::rule(Dune::GeometryType::cube, n ,Dune::QuadratureType::GaussLegendre);
  // make sure we got the right rule:
  suite.check(gl_rule.size() == (size_t) n);
 
  // For every GL node there should be a GK node which is identical.
  // This test obviously has bad complexity, but for n small I don't mind
  for (auto gl=gl_rule.begin(); gl!=gl_rule.end(); ++gl) {
    bool found = false;
    for (auto gk = rule.begin(); gk!=rule.end(); gk++) {
      if (std::abs(gl->position() - gk->position()) < epsilon) {
        found = true;
        break;
      }
    }
    suite.check(found, "Test if Gauss-Legendre nodes are included in Gauss-Kronrod");
  }



  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  TestSuite suite;
  suite.subTest(test_GaussKronrod());
  return suite.exit();
}
