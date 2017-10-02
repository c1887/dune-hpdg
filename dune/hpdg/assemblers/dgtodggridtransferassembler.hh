#ifndef DUNE_HPDG_DG_TO_DG_GRID_TRANSFER_ASSEMBLER_HH
#define DUNE_HPDG_DG_TO_DG_GRID_TRANSFER_ASSEMBLER_HH

#include <type_traits>
#include <dune/istl/matrixindexset.hh>
#include <dune/fufem/assemblers/basisinterpolationmatrixassembler.hh> // contains the LocalBasisComponentWrapper
#include <dune/fufem/assemblers/istlbackend.hh>
#include <dune/geometry/referenceelements.hh>
#include <dune/hpdg/functionspacebases/dgqkglbasis.hh>
#include <dune/hpdg/common/dynamicbcrs.hh>

#include <dune/functions/gridfunctions/discreteglobalbasisfunction.hh>
// implementation details
namespace Impl{
  template<class GridType>
  class MultilevelBasis
  {
    private:
      static const int dim = GridType::dimension;
    public:
      //using LevelBasis = Dune::Functions::DGQkGLBlockBasis<typename GridType::LevelGridView, 1>;
      //using LocalFiniteElement = typename LevelBasis::LocalView::Tree::FiniteElement;

      MultilevelBasis(const GridType& grid) :
        grid_(grid)
      {
        const auto& globalIdSet = grid.globalIdSet();
        const auto& leafIndexSet = grid.leafIndexSet();

        auto maxLevel = grid.maxLevel();

        //levelBasis_.resize(maxLevel+1);
        //for (std::size_t level=0; level<=maxLevel; ++level)
          //levelBasis_[level] = std::make_shared<LevelBasis>(grid.levelGridView(level));

        idToIndex.resize(maxLevel+1);
        size_.resize(maxLevel+1);
        size_[maxLevel] = grid.size(0); // 0 because we're in elementwise DG and count the number of elements, not vertices

        // iterate over levels
        for (int level=0; level<maxLevel; level++) {
          const auto& indexSet = grid.levelIndexSet(level);
          size_[level]=indexSet.size(0);
          for (const auto& e: elements(grid.levelGridView(level)))
            idToIndex[level][globalIdSet.id(e)]=indexSet.index(e); // TODO: subIndex??
        }

        for (const auto& e: elements(grid.leafGridView())) {
          idToIndex[maxLevel][globalIdSet.id(e)]=leafIndexSet.index(e);

          for (auto level=e.level()+1; level<maxLevel; level++) {
            idToIndex[level][globalIdSet.id(e)]= size_[level];
            ++size_[level];
          }
        }
      }

      template<class Index>
      std::size_t size(Index level) const {return size_[level];}

      /** \brief Get index of an element on a given level */
      template<class E, class Index>
      std::size_t index(const E& element, const Index level) const {
        const auto id = grid_.globalIdSet().id(element);
        const auto entry = idToIndex[level].find(id);
        if (entry != idToIndex[level].end())
          return entry->second;
        else
          DUNE_THROW(Dune::Exception, "Element was not found on level " << level);
      }

    private:
      const GridType& grid_;
      //std::vector<std::shared_ptr<LevelBasis> > levelBasis_;
      std::vector<std::size_t> size_;
      using IdType = typename GridType::Traits::GlobalIdSet::IdType;
      std::vector< std::map<IdType, std::size_t> > idToIndex;
  };
}
namespace Dune {
  namespace HPDG{
    template<int k=1, typename MatrixType, typename GridType>
    void assembleDGGridTransferHierarchy(std::vector<std::shared_ptr<MatrixType>>& matrixVector, const GridType& grid){
      const auto multiLevelBasis = ::Impl::MultilevelBasis<GridType>(grid);
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
      using LevelBasis = Dune::Functions::DGQkGLBlockBasis<typename GridType::LevelGridView, k>;
      using CoarseFE = typename LevelBasis::LocalView::Tree::FiniteElement;

      for (int level=0; level<maxLevel; level++) {
        // Set up bases
        const auto cbasis = LevelBasis(grid.levelGridView(level));
        const auto fbasis = LevelBasis(grid.levelGridView(level+1));
        auto fineView = fbasis.localView();
        auto fineIndexSet = fbasis.localIndexSet();
        auto coarseView = cbasis.localView();
        auto coarseIndexSet = fbasis.localIndexSet();

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
            coarseIndexSet.bind(coarseView);

            const auto& coarseFE = coarseView.tree().finiteElement();
            const auto numCoarse = coarseView.size();

            for (const auto& son: descendantElements(element, level+1)) {
              if (son.level()==level) continue;
              auto& localElementMatrix = matrix[multiLevelBasis.index(son, level+1)][multiLevelBasis.index(element, level)];
              fineView.bind(son);
              fineIndexSet.bind(fineView);
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
                auto localFine = fineIndexSet.index(j)[1];
                for (std::size_t i = 0; i < coarseView.size(); i++) {
                  auto localCoarse = coarseIndexSet.index(i)[1];
                  localElementMatrix[localFine][localCoarse]+=values[i];
                }
              }
            }
          }
        }
      }
    }

    template<int k=1, typename K, typename GridType>
    void assembleDGGridTransferHierarchy(std::vector<std::shared_ptr<DynamicBCRSMatrix<K>>>& matrixVector, const GridType& grid){
      const auto multiLevelBasis = ::Impl::MultilevelBasis<GridType>(grid);
      const auto maxLevel = grid.maxLevel();

      /* Setup indices */
      {
        const int blockSize = (int) std::pow(k+1, (int)GridType::dimension);
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
          indices.exportIdx(matrixVector[level]->matrix());
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
        auto fineIndexSet = fbasis.localIndexSet();
        auto coarseView = cbasis.localView();
        auto coarseIndexSet = fbasis.localIndexSet();

        auto& matrix = *(matrixVector[level]); // alias for convenience
        // iterate over level elements:
        for (const auto& element: elements(grid.levelGridView(level))) {
          if (element.isLeaf()) {
            // Assemble local identites on all following levels
            for (auto coarseLevel = level; coarseLevel<maxLevel; ++coarseLevel) {
              auto& localElementMatrix = matrixVector[coarseLevel]->matrix()[multiLevelBasis.index(element, coarseLevel+1)][multiLevelBasis.index(element, coarseLevel)];
              // set to id
              localElementMatrix=0.0;
              for (std::size_t i=0; i< localElementMatrix.N(); i++)
                localElementMatrix[i][i]=1.0;
            }
          }
          else {
            // Assemble transfer to next level
            coarseView.bind(element);
            coarseIndexSet.bind(coarseView);

            const auto& coarseFE = coarseView.tree().finiteElement();
            const auto numCoarse = coarseView.size();

            for (const auto& son: descendantElements(element, level+1)) {
              if (son.level()==level) continue;
              auto& localElementMatrix = matrix.matrix()[multiLevelBasis.index(son, level+1)][multiLevelBasis.index(element, level)];
              fineView.bind(son);
              fineIndexSet.bind(fineView);
              const auto& fineFE = fineView.tree().finiteElement();
              const auto numFine = fineView.size();

              const auto& fatherGeo = son.geometryInFather();
              const auto& referenceElement = Dune::ReferenceElements<K, GridType::dimension>::general(son.type());

              std::vector<typename CoarseFE::Traits::LocalBasisType::Traits::RangeType> values(numCoarse);

              for (std::size_t j = 0; j<numFine; j++)
              {
                /* Interpolate values of the j-th coarse function*/
                const auto& jLocalKey = fineFE.localCoefficients().localKey(j);
                const auto fineBasePosition = referenceElement.position(jLocalKey.subEntity(), jLocalKey.codim());
                const auto local = fatherGeo.global(fineBasePosition);

                coarseFE.localBasis().evaluateFunction(local, values);

                /* copy them into the local block */
                auto localFine = fineIndexSet.index(j)[1];
                for (std::size_t i = 0; i < coarseView.size(); i++) {
                  auto localCoarse = coarseIndexSet.index(i)[1];
                  localElementMatrix[localFine][localCoarse]+=values[i];
                }
              }
            }
          }
        }
      }
    }
  } // end namespace HPDG
} // end namespace Dune
#endif
