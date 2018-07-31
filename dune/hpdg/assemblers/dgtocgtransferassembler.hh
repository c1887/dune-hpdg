#ifndef DUNE_HPDG_DG_TO_CG_TRANSFER_ASSEMBLER_HH
#define DUNE_HPDG_DG_TO_CG_TRANSFER_ASSEMBLER_HH

#include <type_traits>
#include <dune/istl/matrixindexset.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh> // contains the LocalBasisComponentWrapper

namespace Dune {
  namespace HPDG{
    template<typename MatrixType, typename CoarseBasisType, typename FineBasisType>
    void assembleDGtoCGTransferOperator(MatrixType& matrix, const CoarseBasisType& coarseBasis, const FineBasisType& fineBasis)
    {
      using GridView = typename CoarseBasisType::GridView;
      using BlockType = typename MatrixType::block_type;

      /* This all might be done more elegantly*/
      using CoarseFiniteElement = typename CoarseBasisType::LocalView::Tree::FiniteElement;
      using FunctionBaseClass = typename Dune::LocalFiniteElementFunctionBase<CoarseFiniteElement>::type;
      using LocalBasisWrapper = LocalBasisComponentWrapper<typename CoarseFiniteElement::Traits::LocalBasisType, FunctionBaseClass>;

      static_assert(std::is_same<GridView, typename FineBasisType::GridView>::value, "GridViews don't match!");

      const GridView& gridView = coarseBasis.gridView();

      // dune-functions' typical views and index sets
      auto fineView = fineBasis.localView();

      auto coarseView = coarseBasis.localView();

      // set index set
      matrix.setSize(fineBasis.size(), coarseBasis.size());
      matrix.setBuildMode(MatrixType::random);
      auto indices = Dune::MatrixIndexSet(fineBasis.size(), coarseBasis.size());
      for (const auto& element: elements(gridView))
      {
        fineView.bind(element);

        coarseView.bind(element);

        const auto elementIndex = fineView.index(0)[0];

        const auto numCoarse = coarseView.size();

        for (size_t j = 0; j<numCoarse; j++)
        {
          auto globalCoarseIndex = coarseView.index(j);

          indices.add(elementIndex, globalCoarseIndex);
        }
      }

      indices.exportIdx(matrix);
      matrix=0;

      // assemble matrix
      for (const auto& element: elements(gridView))
      {
        fineView.bind(element);

        coarseView.bind(element);

        const auto elementIndex = fineView.index(0)[0];

        // TODO: this works only for trivial ansatz trees
        const auto& fineFE = fineView.tree().finiteElement();
        const auto& coarseFE = coarseView.tree().finiteElement();
        const auto numCoarse = coarseView.size();

        std::vector<typename CoarseFiniteElement::Traits::LocalBasisType::Traits::RangeType> values(numCoarse);
        LocalBasisWrapper coarseBasisFunction(coarseFE.localBasis(),0);

        for (size_t j = 0; j<numCoarse; j++)
        {
          /* Interpolate values of the j-th coarse function*/
          coarseBasisFunction.setIndex(j);
          auto globalCoarseIndex = coarseView.index(j);
          fineFE.localInterpolation().interpolate(coarseBasisFunction, values);

          /* copy them into the local block */
          auto& localElementMatrix = matrix[elementIndex][globalCoarseIndex];
          for (size_t i = 0; i < fineView.size(); i++) {
            auto localFine = fineView.index(i)[1];
            localElementMatrix[localFine][0]+=values[i];
          }
        }
      }
    }
  } // end namespace HPDG 
} // end namespace Dune
#endif
