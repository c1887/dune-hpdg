#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <dune/hpdg/matrix-free/operator.hh>
#include <dune/hpdg/matrix-free/localoperators/ipdgoperator.hh>

#include <dune/functions/functionspacebases/lagrangedgbasis.hh>
#include <dune/functions/functionspacebases/interpolate.hh>
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

template<class Basis>
auto computeMatrix(const Basis& basis, double penalty=2.0) {
  using GridType = typename Basis::GridView::Grid;

  using Matrix = Dune::BCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
  Matrix matrix;

  Dune::Timer timer;
  // assemble laplace bulk terms and ipdg terms
  {
    using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
    using FiniteElement = std::decay_t<decltype(basis.localView().tree().finiteElement())>;
    auto matrixBackend = Dune::Fufem::istlMatrixBackend(matrix);

    auto assembler = Assembler{basis, basis};

    auto patternBuilder = matrixBackend.patternBuilder();
    assembler.assembleSkeletonPattern(patternBuilder);

    patternBuilder.setupMatrix();

    auto vintageIPDGAssembler = InteriorPenaltyDGAssembler<GridType, FiniteElement, FiniteElement>(penalty, true);
    auto localBlockAssembler = [&](const auto& edge, auto& matrixContainer,
        auto&& insideTrialLocalView, auto&& insideAnsatzLocalView, auto&& outsideTrialLocalView, auto&& outsideAnsatzLocalView)
    {
        vintageIPDGAssembler.assembleBlockwise(edge, matrixContainer, insideTrialLocalView.tree().finiteElement(),
                                               insideAnsatzLocalView.tree().finiteElement(),
                                               outsideTrialLocalView.tree().finiteElement(),
                                               outsideAnsatzLocalView.tree().finiteElement());
    };
    auto localBoundaryAssembler = [&](const auto& edge, auto& localMatrix, auto&& insideTrialLocalView, auto&& insideAnsatzLocalView)
    {
        vintageIPDGAssembler.assemble(edge, localMatrix, insideTrialLocalView.tree().finiteElement(), insideAnsatzLocalView.tree().finiteElement());
    };

    assembler.assembleSkeletonEntries(matrixBackend, localBlockAssembler, localBoundaryAssembler); // IPDG terms

    auto vintageBulkAssembler = LaplaceAssembler<GridType,FiniteElement, FiniteElement>();

     //We need to construct the stiffness Matrix into a separate (temporary) matrix object as otherwise the previously assembled entries
     //would be lost. This temporary matrix will be deleted after we leave the block scope
    auto bulkMatrix = matrix;
    bulkMatrix=0;
    auto bmatrixBackend = Dune::Fufem::istlMatrixBackend(bulkMatrix);
    assembler.assembleBulkEntries(bmatrixBackend, vintageBulkAssembler);
    matrix+=bulkMatrix;
  }

  std::cout << "Assembly of matrix took " << timer.stop() <<std::endl;
  return matrix;
}

template<class GV>
TestSuite test_sipg(const GV& gv) {
  using Vector = Dune::BlockVector<Dune::FieldVector<double, 1> >;
  Vector x;

  int iter=10;

  TestSuite suite;

  constexpr int order = 2;
  auto basis = Dune::Functions::LagrangeDGBasis<GV, order>(gv);

  std::cout << "\nTesting SIPG with DG Q2 basis (" << basis.dimension() <<" unknowns):" << std::endl;
  // interpolate a example function
  auto func=[](auto&& x) { return x*x;};
  x.resize(basis.dimension());
  Dune::Functions::interpolate(basis, x, func);
  Vector Ax(x.size());
  Ax=0.0;

  //compute the matrix and the matrix-vector product for comparision
  auto mat = computeMatrix(basis, 2.0*order*order);
  auto Ax_matrix = x;
  {
    Dune::Timer timer;
    for (int i =0; i<iter; i++)
      mat.mv(x, Ax_matrix);
    std::cout << "Computation of " <<iter << " matrix-vector products took: " << timer.stop() << std::endl;
  }

  auto energyError = [&](auto Ax) {
    Ax-=Ax_matrix;
    auto dummy = Ax;
    mat.mv(Ax, dummy);
    return Ax*dummy;
  };

  auto ipdg_local_op = Dune::Fufem::MatrixFree::IPDGOperator<Vector, GV, decltype(basis)>(basis, 2.0, true);
  // while we're at it, test with a factor:
  auto factor = 0.5;
  ipdg_local_op.setFactor(factor);

  auto op = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(ipdg_local_op)>(gv, ipdg_local_op);


  runOperator(op, x, Ax, iter, "SIPG Laplace");

  Ax*=1/factor; // remove the factor again for error estimation

  auto error = energyError(Ax);
  if (error<0)
    DUNE_THROW(Dune::Exception, "Supplied matrix was not positive-definite!");
  suite.check(error<1e-14, "Check if matrix free and matrix based compute the same vector") << "Difference for SIPG is " << error << std::endl;

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  constexpr int dim =2;
  YaspGrid<dim> grid({1,1},{{16,16}});
  TestSuite suite;
  suite.subTest(test_sipg(grid.leafGridView()));
  return suite.exit();
}
