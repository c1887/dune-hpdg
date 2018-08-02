#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <dune/hpdg/matrix-free/operator.hh>
#include <dune/hpdg/matrix-free/localoperators/laplacepointjacobi.hh>
#include <dune/hpdg/matrix-free/localoperators/laplaceoperator.hh>

#include <dune/functions/functionspacebases/pqknodalbasis.hh>
#include <dune/functions/functionspacebases/lagrangedgbasis.hh>
#include <dune/fufem/assemblers/dunefunctionsoperatorassembler.hh>
#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>
#include <dune/fufem/assemblers/localassemblers/interiorpenaltydgassembler.hh>

using namespace Dune;

template<class Op, class V, class S>
void runOperator(Op&& op, const V& x, V& Ax, int iter, S&& s) {
  Dune::Timer timer;
  for (int i =0; i<iter; i++)
    op.apply(x, Ax);
  std::cout << iter << " times matrix-free application of " << s << " took: " << timer.stop() << std::endl;
}

template<class GV>
TestSuite test_sipg(const GV& gv) {
  using Vector = Dune::BlockVector<Dune::FieldVector<double, 1> >;
  Vector b, r, x, c, d, Ax;

  int iter=100;

  TestSuite suite;

  constexpr int order = 1;
  auto basis = Dune::Functions::PQkNodalBasis<GV, order>(gv);

  std::cout << "\nTesting Jacobi with Q1 basis (" << basis.dimension() <<" unknowns):" << std::endl;
  // approx solution
  x.resize(basis.dimension());
  x=0.0;
  // A*approx solution
  Ax.resize(basis.dimension());
  Ax=0.0;
  // right hand side
  b.resize(basis.dimension());
  b=1.0/std::sqrt(gv.size(0));
  // add some noise to the rhs no make it non-constant:
  for(int i =0 ; i < b.size(); i++)
    b[i] += (i%5)*b[i]/10;
  // residual
  r=b;
  // correction
  c.resize(basis.dimension());
  c=0.0;
  // diagonal entries
  d.resize(basis.dimension());
  d=0.0;

  auto jacobi = Dune::Fufem::MatrixFree::LaplaceDiagonal<Vector, GV, decltype(basis)>(basis);
  auto op = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(jacobi)>(gv, jacobi);

  auto laplace = Dune::Fufem::MatrixFree::LaplaceOperator<Vector, GV, decltype(basis)>(basis);
  auto A = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(laplace)>(gv, laplace);

  // compute diagonal entries:
  const auto& dummy =x; // to emphasize that input is not used here
  runOperator(op, dummy, d, 1, "Laplace Jacobi");

  auto norm = [&](const auto& a) {
    auto tmp = a;
    A.apply(a,tmp);
    return a*tmp;
  };

  for (int i=0; i < iter; i++) {
    A.apply(c, Ax);
    r-=Ax;

    for (size_t i = 0; i < c.size(); i++) {
      c[i]=r[i]/d[i];
    }
    auto rnorm = norm(c);

    x+=c;

    std::cout << i << ". correction norm: " << rnorm << std::endl;
    if (rnorm < 1e-8)
      break;

  }

  suite.check(norm(r) < 1e-8, "Norm of residual small enough") << "is " << norm(r);

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  constexpr int dim =2;
  YaspGrid<dim> grid({1,1},{{10,10}});
  TestSuite suite;
  suite.subTest(test_sipg(grid.leafGridView()));
  return suite.exit();
}
