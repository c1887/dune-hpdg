#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <dune/hpdg/matrix-free/operator.hh>
#include <dune/hpdg/matrix-free/localoperators/ipdgblockjacobi.hh>
#include <dune/hpdg/matrix-free/localoperators/ipdgoperator.hh>

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
TestSuite test_sipg(const GV& gv, int iter) {
  using Vector = Dune::BlockVector<Dune::FieldVector<double, 1> >;
  Vector b, r, x, c, Ac;

  TestSuite suite;

  constexpr int order = 1;
  auto penalty = 2.0*order*order;
  //auto basis = Dune::Functions::PQkNodalBasis<GV, order>(gv);
  auto basis = Dune::Functions::LagrangeDGBasis<GV, order>(gv);

  std::cout << "\nTesting matrix-free Block Jacobi with DG Q"<< order << " basis (" << basis.dimension() <<" unknowns):" << std::endl;
  // approx solution
  x.resize(basis.dimension());
  x=0.0;
  // A*approx solution
  Ac.resize(basis.dimension());
  Ac=0.0;
  // right hand side
  b.resize(basis.dimension());
  b=1.0/std::sqrt(gv.size(0));
  // add some noise to the rhs no make it non-constant:
  for(int i =0 ; i < b.size(); i++)
    b[i] += (i%5)*b[i]/10;
   //residual
  r=b;
  // correction
  c.resize(basis.dimension());
  c=0.0;

  // set up local solver: // Das ist natürlich nur die dümmstmögliche Variante
  auto gs = [](const auto& m, const auto& b, auto& x) {
    for (size_t i = 0; i < m.N(); ++i) {
      const auto& mi = m[i];
      const auto& mii = mi[i];
      x[i] = b[i];
      const auto& end = mi.end();
      for (auto it = mi.begin(); it != end; ++it) {
        auto j = it.index();
        if (j != i)
          x[i] -= (*it) * x[j];
      }
      x[i] /= mii;
    }
  };
  using GS = decltype(gs);
  auto jacobi = Dune::Fufem::MatrixFree::IPDGBlockJacobi<Vector, GV, decltype(basis), GS>(basis, gs, penalty, true);
  auto op = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(jacobi)>(gv, jacobi);

  auto laplace = Dune::Fufem::MatrixFree::IPDGOperator<Vector, GV, decltype(basis)>(basis, penalty, true);
  auto A = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(laplace)>(gv, laplace);

  auto norm = [&](const auto& a) {
    auto tmp = a;
    A.apply(a,tmp);
    return a*tmp;
  };

  auto oldcnorm=0.0;
  for (int i=0; i < iter; i++) {
    A.apply(c, Ac);
    r-=Ac;

    // reset correction vector
    c=0;
    // apply one block Jacobi step
    op.apply(r,c);


    auto cnorm = norm(c);
    // TODO: More sophisticated damping
    c*=0.75;
    x+=c;

    printf("%d: \t correction norm: %3.4e, rate %1.3f\n", i, cnorm, (i>0) ? cnorm/oldcnorm : 1);
    oldcnorm=cnorm;
    if (cnorm < 1e-4)
      break;

  }

  suite.check(norm(r) < 1e-2, "Norm of residual small enough") << "is " << norm(r);

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  constexpr int dim =2;
  YaspGrid<dim> grid({1,1},{{10,10}});
  TestSuite suite;
  int iter = 100;
  if (argc>1)
    iter = std::stoi(argv[1]);
  suite.subTest(test_sipg(grid.leafGridView(), iter));
  return suite.exit();
}
