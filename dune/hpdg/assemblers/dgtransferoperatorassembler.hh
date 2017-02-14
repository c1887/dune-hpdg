#ifndef DUNE_HPDG_ASSEMBLERS_DG_TRANSFER_OPERATOR_ASSEMBLER_HH
#define DUNE_HPDG_ASSEMBLERS_DG_TRANSFER_OPERATOR_ASSEMBLER_HH

#include <type_traits>
#include <dune/istl/matrixindexset.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh> // contains the LocalBasisComponentWrapper

namespace Dune {
  namespace HPDG {
    template<typename MatrixType, typename CoarseBasisType, typename FineBasisType>
    void assembleDGTransferOperator(MatrixType& matrix, const CoarseBasisType& coarseBasis, const FineBasisType& fineBasis)
    {
      using GridView = typename CoarseBasisType::GridView;
      using BlockType = typename MatrixType::block_type;

      /* This all might be done more elegantly*/
      using CoarseFiniteElement = typename CoarseBasisType::LocalView::Tree::FiniteElement;
      using FunctionBaseClass = typename Dune::LocalFiniteElementFunctionBase<CoarseFiniteElement>::type;
      using LocalBasisWrapper = LocalBasisComponentWrapper<typename CoarseFiniteElement::Traits::LocalBasisType, FunctionBaseClass>;

      static_assert(std::is_same<GridView, typename FineBasisType::GridView>::value, "GridViews don't match!");

      const GridView& gridView = coarseBasis.gridView();

      /* Assemble matrix. Pattern building should already be complete (after all it's only a block diagonal matrix)!*/

      // dune-functions' typical views and index sets
      auto fineView = fineBasis.localView();
      auto fineIndexSet = fineBasis.localIndexSet();

      auto coarseView = coarseBasis.localView();
      auto coarseIndexSet = coarseBasis.localIndexSet();

      for (const auto& element: elements(gridView))
      {
        fineView.bind(element);
        fineIndexSet.bind(fineView);

        coarseView.bind(element);
        coarseIndexSet.bind(coarseView);

        // Get element index; this is the first index of the twolevel index and should match for coarse and fine basis
        const auto elementIndex = coarseIndexSet.index(0)[0];
        assert(elementIndex == fineIndexSet.index(0)[0]); // todo remove this check?
        auto& localElementMatrix = matrix[elementIndex][elementIndex];

        // TODO: this works only for trivial ansatz trees
        const auto& fineFE = fineView.tree().finiteElement();
        const auto& coarseFE = coarseView.tree().finiteElement();
        const size_t numCoarse = coarseView.size();

        std::vector<typename CoarseFiniteElement::Traits::LocalBasisType::Traits::RangeType> values(numCoarse);
        LocalBasisWrapper coarseBasisFunction(coarseFE.localBasis(),0);

        for (size_t j = 0; j<numCoarse; j++)
        {
          /* Interpolate values of the j-th coarse function*/
          coarseBasisFunction.setIndex(j);
          fineFE.localInterpolation().interpolate(coarseBasisFunction, values);

          /* copy them into the local block */
          auto localCoarse = coarseIndexSet.index(j)[1];
          for (size_t i = 0; i < fineView.size(); i++) {
            auto localFine = fineIndexSet.index(i)[1];
            localElementMatrix[localFine][localCoarse]=values[i];
          }

        }
      }
    }
  } // end namespace HPDG
} // end namespace Dune
#endif
