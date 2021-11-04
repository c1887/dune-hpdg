#ifndef DUNE_HPDG_BUILDING_BLOCKS_MATRICES_HH
#define DUNE_HPDG_BUILDING_BLOCKS_MATRICES_HH
#include <dune/common/fmatrix.hh>
#include <dune/common/fvector.hh>
#include <dune/fufem/assemblers/dunefunctionsoperatorassembler.hh>
#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/fufem/assemblers/localassemblers/dunefunctionsweightedmassassembler.hh>
#include <dune/fufem/assemblers/localassemblers/massassembler.hh>
#include <dune/hpdg/assemblers/localassemblers/gausslobattoipdgassembler.hh>
#include <dune/hpdg/assemblers/localassemblers/gausslobattoipdgcoefficientassembler.hh>
#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/common/resizehelper.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <optional>

namespace Dune::HPDG::BuildingBlocks {

/** Returns the matrix (\nabla \phi_i, \nabla \phi_j) + IPDG terms
 *
 * @param basis The basis currently used.
 * @param penalty the penalty constant (will be multiplied by p^2).
 * @param dirichlet: if true, dirichlet boundary terms will be assembled.
 */
template<typename GridView>
auto
laplace(Dune::Functions::DynamicDGQkGLBlockBasis<GridView> const& basis,
        double penalty,
        bool dirichlet)
{
  using Basis = typename Dune::Functions::DynamicDGQkGLBlockBasis<GridView>;
  using Matrix = Dune::HPDG::DynamicBCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
  Matrix matrix;
  using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
  auto matrixBackend = Dune::Fufem::istlMatrixBackend(matrix.asBCRSMatrix());
  auto patternBuilder = matrixBackend.patternBuilder();

  auto assembler = Assembler{ basis, basis };

  assembler.assembleSkeletonPattern(patternBuilder);

  patternBuilder.setupMatrix();
  matrix.finishIdx();

  // set block sizes:
  Dune::HPDG::resizeFromBasis(matrix, basis);
  matrix = 0;

  auto localAssembler =
    Dune::HPDG::GaussLobattoIPDGAssembler<Basis>(basis, penalty, dirichlet);

  // set up the local lambdas for the global assembler
  auto localBlockAssembler = [&](const auto& edge,
                                 auto& matrixContainer,
                                 auto&& insideTrialLocalView,
                                 auto&&,
                                 auto&& outsideTrialLocalView,
                                 auto&&) {
    localAssembler.assembleEdge(
      edge,
      matrixContainer,
      basis.preBasis().degree(insideTrialLocalView.element()),
      basis.preBasis().degree(outsideTrialLocalView.element()));
  };

  auto localBoundaryAssembler = [&](const auto& edge,
                                    auto& localMatrix,
                                    auto&& insideTrialLocalView,
                                    auto&&) {
    localAssembler.assembleBoundary(
      edge,
      localMatrix,
      basis.preBasis().degree(insideTrialLocalView.element()));
  };

  assembler.assembleBulkEntries( // Laplace
    matrixBackend,
    [&](const auto& e, auto& localMatrix, auto&& lv, auto&&) {
      localAssembler.assembleBulk(
        e, localMatrix, basis.preBasis().degree(lv.element()));
    });

  assembler.assembleSkeletonEntries(
    matrixBackend, localBlockAssembler, localBoundaryAssembler); // IPDG terms

  return matrix;
}

/** Returns the matrix (alpha(x)\nabla \phi_i, \nabla \phi_j) + IPDG terms,
 * where alpha is a scalar function.
 *
 * @param basis The basis currently used.
 * @param penalty the penalty constant (will be multiplied by p^2).
 * @param dirichlet if true, dirichlet boundary terms will be assembled.
 * @param alpha Scalar function following the dune-functions localFunction
 * interface.
 */
template<typename GridView, typename F>
auto
laplace(Dune::Functions::DynamicDGQkGLBlockBasis<GridView> const& basis,
        double penalty,
        bool dirichlet,
        const F& alpha)
{
  using Basis = typename Dune::Functions::DynamicDGQkGLBlockBasis<GridView>;
  using Matrix = Dune::HPDG::DynamicBCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
  Matrix matrix;
  using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
  auto matrixBackend = Dune::Fufem::istlMatrixBackend(matrix.asBCRSMatrix());
  auto patternBuilder = matrixBackend.patternBuilder();

  auto assembler = Assembler{ basis, basis };

  assembler.assembleSkeletonPattern(patternBuilder);

  patternBuilder.setupMatrix();
  matrix.finishIdx();

  // set block sizes:
  Dune::HPDG::resizeFromBasis(matrix, basis);
  matrix = 0;

  // TODO: Fix this assembler. It underintegrates the coefficient!
  auto localAssembler = Dune::HPDG::GaussLobattoIPDGCoefficientAssembler<Basis, F>(
    basis, alpha, penalty, dirichlet);

  // set up the local lambdas for the global assembler
  auto localBlockAssembler = [&](const auto& edge,
                                 auto& matrixContainer,
                                 auto&& insideTrialLocalView,
                                 auto&&,
                                 auto&& outsideTrialLocalView,
                                 auto&&) {
    localAssembler.assembleEdge(
      edge,
      matrixContainer,
      basis.preBasis().degree(insideTrialLocalView.element()),
      basis.preBasis().degree(outsideTrialLocalView.element()));
  };

  auto localBoundaryAssembler = [&](const auto& edge,
                                    auto& localMatrix,
                                    auto&& insideTrialLocalView,
                                    auto&&) {
    localAssembler.assembleBoundary(
      edge,
      localMatrix,
      basis.preBasis().degree(insideTrialLocalView.element()));
  };

  assembler.assembleBulkEntries( // Laplace
    matrixBackend,
    [&](const auto& e, auto& localMatrix, auto&& lv, auto&&) {
      localAssembler.assembleBulk(
        e, localMatrix, basis.preBasis().degree(lv.element()));
    });

  assembler.assembleSkeletonEntries(
    matrixBackend, localBlockAssembler, localBoundaryAssembler); // IPDG terms

  return matrix;
}

/** Returns the mass matrix of the Gauss-Lobatto DG basis.
 *
 * The (block-diagonal) bulk pattern is enough, however, sometimes one wants a
 * larger pattern such that one could add, e.g., the IPDG terms into the same
 * matrix object.
 *
 * @param basis The basis currently used.
 * @param useSkeletonPattern Flag to activate the larger pattern
 */
template<typename GridView>
auto
mass(Dune::Functions::DynamicDGQkGLBlockBasis<GridView> const& basis,
     bool useSkeletonPattern = false)
{
  using Basis = typename Dune::Functions::DynamicDGQkGLBlockBasis<GridView>;
  using Matrix = Dune::HPDG::DynamicBCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
  Matrix matrix;
  using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
  auto matrixBackend = Dune::Fufem::istlMatrixBackend(matrix.asBCRSMatrix());
  auto patternBuilder = matrixBackend.patternBuilder();

  auto assembler = Assembler{ basis, basis };

  if (useSkeletonPattern)
    assembler.assembleSkeletonPattern(patternBuilder);
  else
    assembler.assembleBulkPattern(patternBuilder);

  patternBuilder.setupMatrix();
  matrix.finishIdx();

  // set block sizes:
  Dune::HPDG::resizeFromBasis(matrix, basis);
  matrix = 0;

  using FiniteElement =
    std::decay_t<decltype(basis.localView().tree().finiteElement())>;
  auto vintageBulkAssembler =
    MassAssembler<typename GridView::Grid, FiniteElement, FiniteElement>();

  assembler.assembleBulkEntries(matrixBackend, vintageBulkAssembler);
  return matrix;
}

/** Returns the weighted mass matrix of the Gauss-Lobatto DG basis.
 *
 * The (block-diagonal) bulk pattern is enough, however, sometimes one wants a
 * larger pattern such that one could add, e.g., the IPDG terms into the same
 * matrix object.
 *
 * @param basis The basis currently used.
 * @param f GridViewFunction that is used as weighting function
 * @param quadOrder Order of quadraturerule which is needed for f
 * @param useSkeletonPattern Flag to activate the larger pattern
 * 
 * @todo quadorder should be provided by a QuadKey.
 */
template<typename GridView, typename F>
auto
mass(Dune::Functions::DynamicDGQkGLBlockBasis<GridView> const& basis,
     const F& f,
     int quadOrder,
     bool useSkeletonPattern = false)
{
  using Basis = typename Dune::Functions::DynamicDGQkGLBlockBasis<GridView>;
  using Matrix = Dune::HPDG::DynamicBCRSMatrix<Dune::FieldMatrix<double, 1, 1>>;
  Matrix matrix;
  using Assembler = Dune::Fufem::DuneFunctionsOperatorAssembler<Basis, Basis>;
  auto matrixBackend = Dune::Fufem::istlMatrixBackend(matrix.asBCRSMatrix());
  auto patternBuilder = matrixBackend.patternBuilder();

  auto assembler = Assembler{ basis, basis };

  if (useSkeletonPattern)
    assembler.assembleSkeletonPattern(patternBuilder);
  else
    assembler.assembleBulkPattern(patternBuilder);

  patternBuilder.setupMatrix();
  matrix.finishIdx();

  // set block sizes:
  Dune::HPDG::resizeFromBasis(matrix, basis);
  matrix = 0;

  using FiniteElement =
    std::decay_t<decltype(basis.localView().tree().finiteElement())>;
  auto quad = QuadratureRuleKey{ GridView::Grid::dimension, quadOrder };
  auto vintageBulkAssembler =
    Dune::Fufem::DuneFunctionsWeightedMassAssembler<typename GridView::Grid,
                                                    FiniteElement,
                                                    FiniteElement,
                                                    decltype(f)>(
      basis.gridView().grid(), f, quad);

  assembler.assembleBulkEntries(matrixBackend, vintageBulkAssembler);
  return matrix;
}
}
#endif