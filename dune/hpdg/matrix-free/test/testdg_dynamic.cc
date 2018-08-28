#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#include <dune/grid/yaspgrid.hh>

#include <dune/istl/bvector.hh>
#include <dune/istl/bcrsmatrix.hh>

#include <dune/hpdg/matrix-free/operator.hh>
#include <dune/hpdg/matrix-free/localoperators/ipdgoperator.hh>
#include <dune/hpdg/matrix-free/localoperators/uniformlaplaceoperator.hh>
#include <dune/hpdg/matrix-free/localoperators/sflaplace.hh>
#include <dune/hpdg/matrix-free/localoperators/laplaceoperator.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/common/dynamicbvector.hh>
#include <dune/hpdg/common/resizehelper.hh>

#include <dune/functions/functionspacebases/interpolate.hh>

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

template<class Basis>
auto computeMatrix(const Basis& basis, double penalty=2.0) {
  using GridType = typename Basis::GridView::Grid;

  using Matrix = Dune::HPDG::DynamicBCRSMatrix<double>;
  Matrix matrix;

  Dune::Timer timer;
  // assemble laplace bulk terms and ipdg terms
  {
    using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
    using FiniteElement = std::decay_t<decltype(basis.localView().tree().finiteElement())>;
    auto matrixBackend = Dune::Fufem::istlMatrixBackend(matrix.matrix());

    auto assembler = Assembler{basis, basis};

    auto patternBuilder = matrixBackend.patternBuilder();
    assembler.assembleSkeletonPattern(patternBuilder);
    //assembler.assembleBulkPattern(patternBuilder);

    patternBuilder.setupMatrix();
    Dune::HPDG::resizeFromBasis(matrix, basis); // contains finishIdx, sets block rows and allocates memory

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
    matrix.matrix()+=bulkMatrix.matrix();
    std::cout << "Assembly of matrix took " << timer.stop() <<std::endl;
  }

  return matrix;
}

template<class GV>
TestSuite test_sipg(const GV& gv) {
  using Vector = Dune::HPDG::DynamicBlockVector<double>;
  Vector x;

  int iter=10;

  TestSuite suite;

  constexpr int order = 2;
  auto basis = Dune::Functions::DynamicDGQkGLBlockBasis<GV>(gv, order);

  std::cout << "\nTesting SIPG with dynamic DG basis (" << basis.dimension() <<" unknowns):" << std::endl;
  Dune::HPDG::resizeFromBasis(x, basis);
  auto func=[](auto&& x) {
    return x*x;
  };
  auto xbe = Dune::Functions::hierarchicVector(x);
  Dune::Functions::interpolate(basis, xbe, func);
  auto Ax=x;
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
  //auto ipdg_local_op = Dune::Fufem::MatrixFree::LaplaceOperator<Vector, GV, decltype(basis)>(basis);
  //auto ipdg_local_op = Dune::Fufem::MatrixFree::UniformLaplaceOperator<Vector, GV, decltype(basis)>(basis);
  //auto ipdg_local_op = Dune::Fufem::MatrixFree::SumFactLaplaceOperator<Vector, GV, decltype(basis)>(basis);
  auto op = Dune::Fufem::MatrixFree::Operator<Vector, GV, decltype(ipdg_local_op)>(gv, ipdg_local_op);
  runOperator(op, x, Ax, iter, "SIPG Laplace");

  auto error = energyError(Ax);
  std::cout << "error " << error << std::endl;
  if (error<0)
    DUNE_THROW(Dune::Exception, "Supplied matrix was not positive-definite!");
  suite.check(error<1e-14, "Check if matrix free and matrix based compute the same vector") << "Difference for SIPG is " << error << std::endl;

  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  constexpr int dim =2;
  YaspGrid<dim> grid({1,1},{{50,50}});
  TestSuite suite;
  suite.subTest(test_sipg(grid.leafGridView()));
  return suite.exit();
}