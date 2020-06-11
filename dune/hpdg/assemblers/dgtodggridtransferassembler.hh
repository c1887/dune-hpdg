#ifndef DUNE_HPDG_DG_TO_DG_GRID_TRANSFER_ASSEMBLER_HH
#define DUNE_HPDG_DG_TO_DG_GRID_TRANSFER_ASSEMBLER_HH

#include <type_traits>
#include <dune/common/math.hh>
#include <dune/istl/matrixindexset.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh> // contains the LocalBasisComponentWrapper
#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/geometry/referenceelements.hh>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>
#include <dune/hpdg/common/dynamicbcrs.hh>
#include <dune/hpdg/transferoperators/fulldomainindexsets.hh>
#include <dune/grid/uggrid.hh>
#include <dune/grid/common/gridenums.hh>

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>

namespace Dune {
  namespace HPDG{
    template<int k=1, typename MatrixType, typename GridType>
    void assembleDGGridTransferHierarchy(std::vector<std::shared_ptr<MatrixType>>& matrixVector, const GridType& grid){
      const auto multiLevelBasis = FullDomainLevelIndexSets<GridType>(grid);
      const auto maxLevel = grid.maxLevel();

      /* Setup indices */
      {
        std::vector<Dune::MatrixIndexSet> indicesVector(maxLevel-0);
        for (int i =0; i<maxLevel; i++)
          indicesVector[i].resize(multiLevelBasis.size(i+1), multiLevelBasis.size(i));

        // iterate over all levels
        for (int level=0; level<maxLevel; level++) {
          auto& indices = indicesVector[level];
          // iterate over elements on this level
          for (const auto& element: elements(grid.levelGridView(level))) {
            if (element.isLeaf()) {
              for (auto coarseLevel =level; coarseLevel<maxLevel; coarseLevel++) {
                indicesVector[coarseLevel].add(multiLevelBasis.index(element, coarseLevel+1),multiLevelBasis.index(element, coarseLevel));
              }
            }
            else {
              for (const auto& son: descendantElements(element, level+1)) {
                if (son.level()==level) continue;
                indices.add(multiLevelBasis.index(son, level+1), multiLevelBasis.index(element, level));
              }
            }
          }
          indices.exportIdx(*matrixVector[level]);
        }
      }

      /* Assemble matrices */
      using LevelBasis = Dune::Functions::DynamicDGQkGLBlockBasis<typename GridType::LevelGridView>;
      using CoarseFE = typename LevelBasis::LocalView::Tree::FiniteElement;

      for (int level=0; level<maxLevel; level++) {
        // Set up bases
        const auto cbasis = LevelBasis(grid.levelGridView(level), k);
        const auto fbasis = LevelBasis(grid.levelGridView(level+1), k);
        auto fineView = fbasis.localView();
        auto coarseView = cbasis.localView();

        auto& matrix = *(matrixVector[level]); // alias for convenience
        // iterate over level elements:
        for (const auto& element: elements(grid.levelGridView(level))) {
          if (element.isLeaf()) {
            // Assemble local identites on all following levels
            for (auto coarseLevel = level; coarseLevel<maxLevel; ++coarseLevel) {
              auto& localElementMatrix = (*matrixVector[coarseLevel])[multiLevelBasis.index(element, coarseLevel+1)][multiLevelBasis.index(element, coarseLevel)];
              // set to id
              localElementMatrix=0.0;
              for (std::size_t i=0; i< localElementMatrix.N(); i++)
                localElementMatrix[i][i]=1.0;
            }
          }
          else {
            // Assemble transfer to next level
            coarseView.bind(element);

            const auto& coarseFE = coarseView.tree().finiteElement();
            const auto numCoarse = coarseView.size();

            for (const auto& son: descendantElements(element, level+1)) {
              if (son.level()==level) continue;
              auto& localElementMatrix = matrix[multiLevelBasis.index(son, level+1)][multiLevelBasis.index(element, level)];
              fineView.bind(son);
              const auto& fineFE = fineView.tree().finiteElement();
              const auto numFine = fineView.size();

              const auto& fatherGeo = son.geometryInFather();
              const auto& referenceElement = Dune::ReferenceElements<typename MatrixType::field_type, GridType::dimension>::general(son.type());

              std::vector<typename CoarseFE::Traits::LocalBasisType::Traits::RangeType> values(numCoarse);

              for (std::size_t j = 0; j<numFine; j++)
              {
                /* Interpolate values of the j-th coarse function*/
                const auto& jLocalKey = fineFE.localCoefficients().localKey(j);
                const auto fineBasePosition = referenceElement.position(jLocalKey.subEntity(), jLocalKey.codim());
                const auto local = fatherGeo.global(fineBasePosition);

                coarseFE.localBasis().evaluateFunction(local, values);

                /* copy them into the local block */
                auto localFine = fineView.index(j)[1];
                for (std::size_t i = 0; i < coarseView.size(); i++) {
                  auto localCoarse = coarseView.index(i)[1];
                  localElementMatrix[localFine][localCoarse]+=values[i];
                }
              }
            }
          }
        }
      }
    }

#if 0 // This is the classical approach, going from coarse to fine. This does, however,
    not work properly in a specific parallel case with UGGrid. Thus, we deactivate this for now and go from fine to coarse.

    template<int k=1, typename K, typename GridType>
    void assembleDGGridTransferHierarchy(std::vector<std::shared_ptr<DynamicBCRSMatrix<K>>>& matrixVector, const GridType& grid){
      const auto multiLevelBasis = FullDomainLevelIndexSets<GridType>(grid);
      const auto maxLevel = grid.maxLevel();

      /* Setup indices */
      {
        const int blockSize = power(k+1, static_cast<int>(GridType::dimension));
        std::vector<Dune::MatrixIndexSet> indicesVector(maxLevel-0);
        for (int i =0; i<maxLevel; i++)
          indicesVector[i].resize(multiLevelBasis.size(i+1), multiLevelBasis.size(i));

        // iterate over all levels
        for (int level=0; level<maxLevel; level++) {
          auto& indices = indicesVector[level];
          // iterate over elements on this level
          for (const auto& element: elements(grid.levelGridView(level))) {
            if (element.isLeaf()) {
              for (auto coarseLevel =level; coarseLevel<maxLevel; coarseLevel++) {
                indicesVector[coarseLevel].add(multiLevelBasis.index(element, coarseLevel+1),multiLevelBasis.index(element, coarseLevel));
              }
            }
            else {
              for (const auto& son: descendantElements(element, level+1)) {
                if (son.level()==level) continue;
                indices.add(multiLevelBasis.index(son, level+1), multiLevelBasis.index(element, level));
              }
            }
          }
          indices.exportIdx(*matrixVector[level]);
          matrixVector[level]->finishIdx();
          for (size_t i = 0; i < matrixVector[level]->N(); i++) {
            matrixVector[level]->blockRows(i)=blockSize;
          }
          for (size_t i = 0; i < matrixVector[level]->M(); i++) {
            matrixVector[level]->blockColumns(i)=blockSize;
          }
          matrixVector[level]->update();
        }
      }

      /* Assemble matrices */
      using LevelBasis = Dune::Functions::DGQkGLBlockBasis<typename GridType::LevelGridView, k>;
      using CoarseFE = typename LevelBasis::LocalView::Tree::FiniteElement;

      for (int level=0; level<maxLevel; level++) {
        // Set up bases
        const auto cbasis = LevelBasis(grid.levelGridView(level));
        const auto fbasis = LevelBasis(grid.levelGridView(level+1));
        auto fineView = fbasis.localView();
        auto coarseView = cbasis.localView();

        auto& matrix = *(matrixVector[level]); // alias for convenience
        // iterate over level elements:
        for (const auto& element: elements(grid.levelGridView(level))) {
          if (element.isLeaf()) {
            // Assemble local identites on all following levels
            for (auto coarseLevel = level; coarseLevel<maxLevel; ++coarseLevel) {
              auto& localElementMatrix = (*matrixVector[coarseLevel])[multiLevelBasis.index(element, coarseLevel+1)][multiLevelBasis.index(element, coarseLevel)];
              // set to id
              localElementMatrix=0.0;
              for (std::size_t i=0; i< localElementMatrix.N(); i++)
                localElementMatrix[i][i]=1.0;
            }
          }
          else {
            // Assemble transfer to next level
            coarseView.bind(element);

            const auto& coarseFE = coarseView.tree().finiteElement();
            const auto numCoarse = coarseView.size();

            for (const auto& son: descendantElements(element, level+1)) {
              if (son.level()==level) continue;
              auto& localElementMatrix = matrix[multiLevelBasis.index(son, level+1)][multiLevelBasis.index(element, level)];
              fineView.bind(son);
              const auto& fineFE = fineView.tree().finiteElement();
              const auto numFine = fineView.size();

              const auto& fatherGeo = son.geometryInFather();
              const auto& referenceElement = Dune::ReferenceElements<typename GridType::ctype, GridType::dimension>::general(son.type());

              std::vector<typename CoarseFE::Traits::LocalBasisType::Traits::RangeType> values(numCoarse);

              for (std::size_t j = 0; j<numFine; j++)
              {
                /* Interpolate values of the j-th coarse function*/
                const auto& jLocalKey = fineFE.localCoefficients().localKey(j);
                const auto fineBasePosition = referenceElement.position(jLocalKey.subEntity(), jLocalKey.codim());
                const auto local = fatherGeo.global(fineBasePosition);

                coarseFE.localBasis().evaluateFunction(local, values);

                /* copy them into the local block */
                auto localFine = fineView.index(j)[1];
                for (std::size_t i = 0; i < coarseView.size(); i++) {
                  auto localCoarse = coarseView.index(i)[1];
                  localElementMatrix[localFine][localCoarse]+=values[i];
                }
              }
            }
          }
        }
      }
    }
#else
    template<int k=1, typename K, typename GridType>
    void assembleDGGridTransferHierarchy(std::vector<std::shared_ptr<DynamicBCRSMatrix<K>>>& matrixVector, const GridType& grid){
      constexpr const int dim = GridType::dimensionworld;
      const auto maxLevel = grid.maxLevel();
      const auto multiLevelBasis = FullDomainLevelIndexSets<GridType>(grid);

      using LevelBasis = Dune::Functions::DynamicDGQkGLBlockBasis<typename GridType::LevelGridView>;
      /* Setup indices */
      {
        const int blockSize = power(k+1, dim);
        std::vector<Dune::MatrixIndexSet> indicesVector(maxLevel); // we have maxLevel +1 levels and hence maxLevel transfers

        // iterate over all levels
        for (int level=maxLevel; level>0; level--) {

          auto& indices = indicesVector[level-1];
          indices.resize(multiLevelBasis.size(level), multiLevelBasis.size(level-1));
          // iterate over elements on this level
          for (const auto& element: elements(grid.levelGridView(level))) {
            const auto& father=element.father();
            indices.add(multiLevelBasis.index(element, level), multiLevelBasis.index(father, level-1));
          }
        }

        // also add identity terms blocks for early leaf elements
        for (int level=0; level< maxLevel; ++level) {
          for (const auto& element: elements(grid.levelGridView(level))) {
            if(element.isLeaf()) {
              for (auto coarseLevel =level; coarseLevel<maxLevel; coarseLevel++) {
                indicesVector[coarseLevel].add(multiLevelBasis.index(element, coarseLevel+1),multiLevelBasis.index(element, coarseLevel));
              }
            }
          }
        }

        // finish idx for all matrices

        for(std::size_t i = 0; i < maxLevel; i++) {
          indicesVector[i].exportIdx(*matrixVector[i]);
          matrixVector[i]->finishIdx();

          for (size_t j = 0; j < matrixVector[i]->N(); j++) {
            matrixVector[i]->blockRows(j)=blockSize;
          }
          for (size_t j = 0; j < matrixVector[i]->M(); j++) {
            matrixVector[i]->blockColumns(j)=blockSize;
          }
          matrixVector[i]->update();
        }
      }

      /* Assemble matrices */
      using CoarseFE = typename LevelBasis::LocalView::Tree::FiniteElement;

      for (int level=maxLevel; level>0; level--) {
        // Set up bases
        const auto cbasis = LevelBasis(grid.levelGridView(level-1), k);
        const auto fbasis = LevelBasis(grid.levelGridView(level), k);
        auto fineView = fbasis.localView();
        auto coarseView = cbasis.localView();

        auto& matrix = *(matrixVector[level-1]); // alias for convenience
        // iterate over level elements:
        for (const auto& son: elements(grid.levelGridView(level))) {
          // Assemble transfer to next level
          fineView.bind(son);

          const auto& fineFE = fineView.tree().finiteElement();
          const auto numFine = fineView.size();

          const auto& element = son.father();
          coarseView.bind(element);

          auto& localElementMatrix = matrix[multiLevelBasis.index(son, level)][multiLevelBasis.index(element, level-1)];
          const auto& coarseFE = coarseView.tree().finiteElement();
          const auto numCoarse = coarseView.size();

          const auto& fatherGeo = son.geometryInFather();
          const auto& referenceElement = Dune::ReferenceElements<typename GridType::ctype, GridType::dimension>::general(son.type());

          std::vector<typename CoarseFE::Traits::LocalBasisType::Traits::RangeType> values(numCoarse); // TODO outer loop?

          for (std::size_t j = 0; j<numFine; j++)
          {
            /* Interpolate values of the j-th coarse function*/
            const auto& jLocalKey = fineFE.localCoefficients().localKey(j);
            const auto fineBasePosition = referenceElement.position(jLocalKey.subEntity(), jLocalKey.codim());
            const auto local = fatherGeo.global(fineBasePosition);

            coarseFE.localBasis().evaluateFunction(local, values);

            /* copy them into the local block */
            auto localFine = fineView.index(j)[1];
            for (std::size_t i = 0; i < coarseView.size(); i++) {
              auto localCoarse = coarseView.index(i)[1];
              localElementMatrix[localFine][localCoarse]+=values[i];
            }
          }
        }
      }
      // set identity matrix blocks
      for(int level=0; level<maxLevel; level++) {
        for (const auto& element: elements(grid.levelGridView(level))) {
          if (element.isLeaf()) {
            // Assemble local identites on all following levels
            for (auto coarseLevel = level; coarseLevel<maxLevel; ++coarseLevel) {
              auto& localElementMatrix = (*matrixVector[coarseLevel])[multiLevelBasis.index(element, coarseLevel+1)][multiLevelBasis.index(element, coarseLevel)];
              // set to id
              localElementMatrix=0.0;
              for (std::size_t i=0; i< localElementMatrix.N(); i++)
                localElementMatrix[i][i]=1.0;
            }
          }
        }
      }
    }
#endif

    /** Free function returning a vector containing the transfer
     * matrices used for geometric multigrid with a DG<k> basis.
     */
    template<typename G>
    auto
    dgGridTransferHierarchy(const G& grid)
    {
      // Transfer operators can have scalar entries even in vector-valued case
      using M = Dune::HPDG::DynamicBCRSMatrix<Dune::FieldMatrix<double, 1,1>>;
      auto m = std::vector<std::shared_ptr<M>>(grid.maxLevel());

      for (auto& t: m)
        t = std::make_shared<M>();

      assembleDGGridTransferHierarchy(m, grid);

      return m;
    }
  } // end namespace HPDG
} // end namespace Dune
#endif
