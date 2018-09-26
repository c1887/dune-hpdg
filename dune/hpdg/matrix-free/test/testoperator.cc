#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <dune/hpdg/matrix-free/operator.hh>
#include <dune/hpdg/matrix-free/localoperators/identityoperator.hh>
#include <dune/hpdg/matrix-free/localoperators/localassembleroperator.hh>
#include <dune/hpdg/matrix-free/localoperators/laplaceoperator.hh>
#include <dune/hpdg/matrix-free/localoperators/uniformlaplaceoperator.hh>

#include <dune/functions/functionspacebases/pqknodalbasis.hh>
#include <dune/functions/functionspacebases/lagrangedgbasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
#include <dune/fufem/assemblers/dunefunctionsoperatorassembler.hh>
#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/assemblers/localassemblers/laplaceassembler.hh>

using namespace Dune;

template<class Op, class V, class S>
void runOperator(Op&& op, const V& x, V& Ax, int iter, S&& s) {
      Dune::Timer timer;
      for (int i =0; i<iter; i++)
        op.apply(x, Ax);
      std::cout << iter << " times matrix-free application of " << s << " took: " << timer.stop() << std::endl;
}

template<class Basis, class LA>
auto computeMatrix(const Basis& basis, LA&& localAssembler) {
  using Matrix = BCRSMatrix<FieldMatrix<double, 1,1>>;
  Matrix mat;

  Dune::Timer timer;

  // assemble Matrix
  // Attention: There is still a bug in the assembler, see #18.
  // The built matrix, however, should be still correct.
  auto backend = Dune::Fufem::istlMatrixBackend(mat);

  using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
  auto assembler = Assembler{basis, basis};

  assembler.assembleBulk(backend, localAssembler);
  std::cout << "Assembly of matrix took " << timer.stop() <<std::endl;

  return mat;
}

template<class GV>
TestSuite test_operator(const GV& gv) {
  using Vector = Dune::BlockVector<Dune::FieldVector<double, 1> >;
  Vector x;
  x.resize(42);
  x=1.0;
  Vector Ax(x.size());
  Ax=0.0;

  int iter=10;

  TestSuite suite;

  // construction with just a single local operator:
  {
    auto id = Fufem::MatrixFree::IdentiyOperator<Vector, GV>();
    auto op = Fufem::MatrixFree::Operator<Vector, GV, decltype(id)>(gv, id);

    op.apply(x, Ax);

    // test if x == Ax== Ix == x
    x-=Ax;
    suite.check(x.two_norm() < 1e-15, "Test identiy operator") << "Difference should be zero, is " << x.two_norm();
  }

  // construction with more than a single local operator and non-default factors
  {
    // for demo, we use two times id. Note that local operators are copied. If this is expensive,
    // consider moving them!
    auto id = Fufem::MatrixFree::IdentiyOperator<Vector, GV>();
    auto localOps = std::make_tuple(id, id);
    auto op = Fufem::MatrixFree::Operator<Vector, GV, decltype(localOps)>(gv, std::move(localOps));
    {
      auto& lo = op.localOperators();
      std::get<0>(lo).setFactor(1.0);
      std::get<1>(lo).setFactor(2.0);
    }

    op.apply(x, Ax);

    // test if (1+2)x == Ax
    x*=3.0;
    x-=Ax;
    suite.check(x.two_norm() < 1e-15, "Test identiy operator") << "Difference should be zero, is " << x.two_norm();
  }

  //reset x
  x=1.0;

  // check different local operators that all resemble the product with the Laplace stiffness matrix
  auto bases = std::make_tuple(
      std::make_pair(Dune::Functions::PQkNodalBasis<GV, 1>(gv),
        std::string("Q1")),
      std::make_pair(Dune::Functions::LagrangeDGBasis<GV, 2>(gv),
        std::string("Q2 (DG)"))
      );


  Hybrid::forEach(bases, [&](const auto& basisPair)
  {
    auto basis = basisPair.first;
    std::cout << "\nTesting with " << basisPair.second<< " basis (" << basis.dimension() <<" unknowns):" << std::endl;
    // interpolate a example function
    auto func=[](auto&& x) { return x*x;};
    x.resize(basis.dimension());
    Dune::Functions::interpolate(basis, x, func);
    Ax.resize(x.size());

    using FE = std::decay_t<decltype(basis.localView().tree().finiteElement())>;
    auto laplace_localAssembler = LaplaceAssembler<typename GV::Grid, FE, FE>();

    // compute the matrix and the matrix-vector product for comparision
    auto mat = computeMatrix(basis, laplace_localAssembler);
    auto Ax_matrix = x;
    {
      Dune::Timer timer;
      for (int i =0; i<iter; i++)
        mat.mv(x, Ax_matrix);
      std::cout << "Computation of " <<iter << " matrix-vector products took: " << timer.stop() << std::endl;
    }

    auto energyError = [&](auto& Ax) {
      Ax-=Ax_matrix;
      auto dummy = Ax;
      mat.mv(Ax, dummy);
      return Ax*dummy;
    };

    // check several variants of how to construct a local operator
    auto localOps = std::make_tuple(
        // assembles directly from fufem localassembler. Likely the sloweset!
        std::make_pair(
          Fufem::MatrixFree::LocalAssemblerOperator<Vector, GV, decltype(basis), decltype(laplace_localAssembler)>(basis, laplace_localAssembler),
          "LocalAssemblerOperator"
        ),


        // computes the operator matrix-free
        std::make_pair(
          Fufem::MatrixFree::LaplaceOperator<Vector, GV, decltype(basis)>(basis),
          "LaplaceOperator"
        ),

        // computes the operator matrix-free, assuming geometry and finite elements are the same on all elements.
        std::make_pair(
          Fufem::MatrixFree::UniformLaplaceOperator<Vector, GV, decltype(basis)>(basis),
          "UniformLaplaceOperator"
        )
    );

    Hybrid::forEach(localOps, [&](auto&& laplace)
    {
      auto operatorName = std::string(laplace.second);
      Ax=0;
      auto op = Dune::Fufem::MatrixFree::Operator<Vector, GV, std::decay_t<decltype(laplace.first)>>(gv, laplace.first);
      runOperator(op, x, Ax, iter, operatorName);

      auto error = energyError(Ax);
      suite.check(error<1e-14, "Check if matrix free and matrix based compute the same vector") << "Difference for " << operatorName << " is " << error << std::endl;
    });


  });

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  constexpr int dim =2;
  YaspGrid<dim> grid({1,1},{{16,16}});
  TestSuite suite;
  suite.subTest(test_operator(grid.leafGridView()));
  return suite.exit();
}
