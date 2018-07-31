#ifndef DUNE_HPDG_DG_TO_CG_NONCONFORMING_TRANSFER_ASSEMBLER_HH
#define DUNE_HPDG_DG_TO_CG_NONCONFORMING_TRANSFER_ASSEMBLER_HH

#include <type_traits>
#include <dune/istl/matrixindexset.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh> // contains the LocalBasisComponentWrapper
#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/geometry/referenceelements.hh>

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
namespace Dune {
  namespace HPDG{
    template<typename MatrixType, typename CoarseBasisType, typename FineBasisType>
    void assembleDGtoCGNonConformingTransferOperator(MatrixType& matrix, const CoarseBasisType& coarseBasis, const FineBasisType& fineBasis)
    {
      using CGridView = typename CoarseBasisType::GridView;
      using FGridView = typename FineBasisType::GridView;
      using BlockType = typename MatrixType::block_type;
      const int dim = FGridView::dimension;

      /* This all might be done more elegantly*/
      using CoarseFiniteElement = typename CoarseBasisType::LocalView::Tree::FiniteElement;
      using FunctionBaseClass = typename Dune::LocalFiniteElementFunctionBase<CoarseFiniteElement>::type;
      using LocalBasisWrapper = LocalBasisComponentWrapper<typename CoarseFiniteElement::Traits::LocalBasisType, FunctionBaseClass>;

      //static_assert(std::is_same<GridView, typename FineBasisType::GridView>::value, "GridViews don't match!");

      const auto& cgridView = coarseBasis.gridView();
      const auto& fgridView = fineBasis.gridView();
      const auto maxLevel = fgridView.grid().maxLevel();

      // dune-functions' typical views
      auto fineView = fineBasis.localView();

      auto coarseView = coarseBasis.localView();

      // set index set
      matrix.setSize(fineBasis.size(), coarseBasis.size());
      matrix.setBuildMode(MatrixType::random);
      auto indices = Dune::MatrixIndexSet(fineBasis.size(), coarseBasis.size());
      for (const auto& element: elements(cgridView))
      {
        coarseView.bind(element);
        const auto numCoarse = coarseView.size();

        if (element.isLeaf()) {
          fineView.bind(element);
          const auto elementIndex = fineView.index(0)[0];

          for (size_t j = 0; j<numCoarse; j++)
          {
            auto globalCoarseIndex = coarseView.index(j);
            indices.add(elementIndex, globalCoarseIndex);
          }
        }

        else {
          for (const auto& leaf : descendantElements(element, std::min(element.level()+1, maxLevel))) {
            fineView.bind(leaf);
            const auto elementIndex = fineView.index(0)[0];

            for (size_t j = 0; j<numCoarse; j++)
            {
              auto globalCoarseIndex = coarseView.index(j);

              indices.add(elementIndex, globalCoarseIndex);
            }
          }
        }
      }

      indices.exportIdx(matrix);
      matrix=0;

      // assemble matrix
      for (const auto& element: elements(cgridView))
      {
        coarseView.bind(element);
        // TODO: this works only for trivial ansatz trees
        const auto& coarseFE = coarseView.tree().finiteElement();
        const auto numCoarse = coarseView.size();
        if (element.isLeaf()) {
          fineView.bind(element);
          const auto elementIndex = fineView.index(0)[0];

          // TODO: this works only for trivial ansatz trees
          const auto& fineFE = fineView.tree().finiteElement();

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
        else {
          for (const auto& leaf : descendantElements(element, std::min(element.level()+1, maxLevel))) {
            if (element.level() == leaf.level()) continue;

            fineView.bind(leaf);
            const auto elementIndex = fineView.index(0)[0];
            const auto numFine = fineView.size();

            const auto& cgeo = leaf.geometryInFather();
            const auto& fineRefElement = Dune::ReferenceElements<typename MatrixType::field_type, dim>::general(leaf.type());

            const auto& fineFE = fineView.tree().finiteElement();

            std::vector<typename CoarseFiniteElement::Traits::LocalBasisType::Traits::RangeType> values(numCoarse);

            for (size_t j = 0; j<numFine; j++)
            {
              /* Interpolate values of the j-th coarse function*/
              const auto& jLocalKey = fineFE.localCoefficients().localKey(j);
              const auto fineBasePosition = fineRefElement.position(jLocalKey.subEntity(), jLocalKey.codim());
              const auto local = cgeo.global(fineBasePosition);

              coarseFE.localBasis().evaluateFunction(local, values);

              /* copy them into the local block */
              auto localFine = fineView.index(j)[1];
              for (size_t i = 0; i < coarseView.size(); i++) {
                auto globalCoarseIndex = coarseView.index(i);
                auto& localElementMatrix = matrix[elementIndex][globalCoarseIndex];
                localElementMatrix[localFine][0]+=values[i];
              }
            }
          }
        }
      }
    }
  } // end namespace HPDG
} // end namespace Dune
#endif
