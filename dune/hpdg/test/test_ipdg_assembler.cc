#include <config.h>
#include <dune/common/test/testsuite.hh>
#include <dune/common/parallel/mpihelper.hh>
#include <dune/common/timer.hh>

#include <dune/grid/yaspgrid.hh>
#include <dune/grid/utility/structuredgridfactory.hh>

#include <dune/fufem/assemblers/operatorassembler.hh>
#include <dune/hpdg/assemblers/gausslobattoipdgassembler.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/common/resizehelper.hh>

#include "testobjects.hh"

using namespace Dune;

template<class G>
TestSuite testAssembly(const G& grid, int k) {
  TestSuite suite;

  using GV = typename G::LeafGridView;
  auto gridView = grid.leafGridView();

  const double penalty = 2.0;

  auto basis = Functions::DynamicDGQkGLBlockBasis<GV>(gridView, k);
  using Basis = decltype(basis);

  Dune::Timer timer;
  auto fufemMatrix = dynamicStiffnessMatrix(grid, k, penalty); 
  timer.stop();
  std::cout << "Fufem assembler took " << timer.elapsed() << " seconds." << std::endl;
  timer.reset();

  // now, we assemble a matrix on our own with the new assembler to compare:
  using Matrix = HPDG::DynamicBCRSMatrix<double>;
  Matrix matrix;
  {
    timer.start();
    auto& istlBCRS = matrix.matrix();

    using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
    auto matrixBackend = Dune::Fufem::istlMatrixBackend(istlBCRS);
    auto patternBuilder = matrixBackend.patternBuilder();

    auto assembler = Assembler{basis, basis};

    assembler.assembleSkeletonPattern(patternBuilder);

    patternBuilder.setupMatrix();
    matrix.finishIdx();

    // set block sizes:
    Dune::HPDG::resizeFromBasis(matrix, basis);
    matrix=0;

    auto localAssembler = Dune::HPDG::GaussLobattoIPDGAssembler<Basis>(basis, penalty, true);

    // set up the local lambdas for the global assembler
    auto localBlockAssembler = [&](const auto& edge, auto& matrixContainer,
        auto&& insideTrialLocalView, auto&&, auto&& outsideTrialLocalView, auto&&)
    {
      localAssembler.assembleEdge(edge, matrixContainer, basis.preBasis().degree(insideTrialLocalView.element()), basis.preBasis().degree(outsideTrialLocalView.element()));
    };

    auto localBoundaryAssembler = [&](const auto& edge, auto& localMatrix, auto&& insideTrialLocalView, auto&&)
    {
      localAssembler.assembleBoundary(edge, localMatrix, basis.preBasis().degree(insideTrialLocalView.element()));
    };

    auto bulkMatrix = matrix;
    bulkMatrix.matrix()=0;
    auto bmatrixBackend = Dune::Fufem::istlMatrixBackend(bulkMatrix.matrix());
    // bulk pattern was inherited from the skeleton part

    // we assemble the bulk terms first. This way, many terms will be already cached once the face assembler is called

    assembler.assembleBulkEntries(bmatrixBackend, [&](const auto& e, auto& localMatrix, auto&& lv, auto&&) {
        localAssembler.assembleBulk(e, localMatrix, basis.preBasis().degree(lv.element()));
        }
      );

    assembler.assembleSkeletonEntries(matrixBackend, localBlockAssembler, localBoundaryAssembler); // IPDG terms

    istlBCRS+=bulkMatrix.matrix();
    timer.stop();
    std::cout << "New HPDG assembler took " << timer.elapsed() << " seconds." << std::endl;
  }

  // Check if we computed (roughly) the same matrix:
  matrix.matrix()-= fufemMatrix.matrix();

  /* We use a relatively large error treshold here because the error estimation with the Frobenius norm
   * doesn't seem to be too stable. Even if one compares two matrices both calcuculated with the very same method, one
   * get's some error in the frobenius norm
   */
  suite.check(matrix.matrix().frobenius_norm() < 1e-11, "Check if new and old assemblers compute the same matrix") << "Error was too great: " << matrix.matrix().frobenius_norm();
  return suite;
}

int main(int argc, char** argv) {
  MPIHelper::instance(argc, argv);

  using Grid = YaspGrid<2>;
  auto grid = StructuredGridFactory<Grid>::createCubeGrid({0,0}, {1.0,1.0}, {{16,16}});


  TestSuite suite;
  for (int i = 1; i < 5; i++)
    suite.subTest(testAssembly(*grid, i));

  return suite.exit();
}
