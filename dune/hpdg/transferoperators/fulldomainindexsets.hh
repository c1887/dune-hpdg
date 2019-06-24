#pragma once

#include <dune/common/exceptions.hh>
#include <dune/grid/common/gridenums.hh>

#include <vector>
#include <map>
#include <tuple>


namespace Dune {
namespace HPDG {

  /** This class provides an index set for the codim-0 elements on
   * each level that spans the whole domain.
   *
   * More precisely, if the grid was non-uniformly refined, the level
   * gridview and the corresponding index set would only span those areas
   * of the domain which are covered by the elements which live on the particular level.
   * This class adds the corresponding coarser leaf nodes such that the index set contains 
   * elements which cover the whole domain. The indices will be contiguous.
   * Moreover, we contain a flag for each element that indicates whether the given element
   * is part of the interior partition or not in a parallel context.
   * In the sequential case, this always evaluates to true and can be ignored.
   *
   * This is useful for assembling transfer operators or organizing parallel communication.
   *
   * TODO: Note that this is internally based on a std::map. It might be worth to improve this.
   */
  template<class GridType>
  class FullDomainLevelIndexSets
  {
    private:
      static const int dim = GridType::dimension;
    public:

      using Grid = GridType;

      FullDomainLevelIndexSets(const GridType& grid) :
        grid_(grid)
      {
        const auto& localIdSet = grid.localIdSet();
        const auto& leafIndexSet = grid.leafIndexSet();

        auto maxLevel = grid.maxLevel();

        idToIndex_.resize(maxLevel+1);
        size_.resize(maxLevel+1);
        size_[maxLevel] = grid.size(0); // 0 because we're in elementwise DG and count the number of elements, not vertices

        // iterate over levels
        for (int level=0; level<maxLevel; level++) {
          const auto& indexSet = grid.levelIndexSet(level);
          size_[level]=grid.size(level,0);
          for (const auto& e: elements(grid.levelGridView(level))) {
            idToIndex_[level][localIdSet.id(e)]={indexSet.index(e), e.partitionType()==Dune::InteriorEntity};
          }
        }

        for (const auto& e: elements(grid.leafGridView())) {
          idToIndex_[maxLevel][localIdSet.id(e)]=
            {leafIndexSet.index(e), e.partitionType()==Dune::InteriorEntity};

          for (auto level=e.level()+1; level<maxLevel; level++) {
            idToIndex_[level][localIdSet.id(e)]= {size_[level], e.partitionType()==Dune::InteriorEntity};
            ++size_[level];
          }
        }
      }

      /** Size of a given level set */
      template<class Index>
      std::size_t size(Index level) const
      {return size_[level];}

      /** Computes the number of elements that are interior on a given level */
      template<class Index>
      std::size_t interiorElements(Index level) const
      {
        std::size_t sum=0;
        for(const auto& val: idToIndex_[level]) {
          sum+=val.second.second;
        }
        return sum;
      }

      /** \brief Get index of an element on a given level */
      template<class E, class Index>
      std::size_t index(const E& element, const Index level) const {
        const auto id = grid_.localIdSet().id(element);
        const auto entry = idToIndex_[level].find(id);
        if (entry != idToIndex_[level].end())
          return entry->second.first;
        else
          DUNE_THROW(Dune::Exception, "Element was not found on level " << level);
      }

      /** \brief Returns a vector with indices of elements that are not interior 
       * on a given level.
       */
      std::vector<std::size_t> nonInteriorElements(std::size_t level) const {
        std::vector<std::size_t> ghosts;
        for(const auto& element: idToIndex_[level]) {
          // each 'element' is the entry of a map, and we're interested in the mapped entry,
          // thus the 1st 'second'.
          // This map entry is a pair, whose second entry denotes whether the element is owned or not,
          // thus the 2nd call for 'second':
          if(not element.second.second)
            ghosts.push_back(element.second.first); // save the (multilevel) indices which are not owned
        }

        return ghosts;
      }

      /** Read-only access to the map that maps local-ids to
       * the pair of full domain index and interior-flag.
       */
      const auto& idToIndex(std::size_t level) const {
        return idToIndex_[level];
      }

      /** Return the (standard, not full domain!) LevelGridView */
      auto gridView(int level) const {
        return grid_.levelGridView(level);
      }

    private:
      const GridType& grid_;
      //std::vector<std::shared_ptr<FullDomainSingleLevelIndexSet> > levelBasis_;
      std::vector<std::size_t> size_;
      using IdType = typename GridType::Traits::LocalIdSet::IdType;
      std::vector< std::map<IdType, std::pair<std::size_t, char>> > idToIndex_; // holds both the index and a flag whether the element is owned // TODO map is probably bottleneck
  };

  /** Pendant to the FullDomainLevelIndexSets just for the leafview */
  template<class GridType>
  class FullDomainLeafIndexSet {
    public:
    using Grid=GridType;

    FullDomainLeafIndexSet(const Grid& grid):
      grid_(grid)
    {
      const auto& idxset = grid_.leafIndexSet();
      for(const auto& element : elements(grid_.leafGridView())) {
        auto i = idxset.index(element);
        idToIndex_[i] = {i, element.partitionType() == Dune::InteriorEntity};
      }
    }

    template<class Index>
    std::size_t size(Index) const {return grid_.leafGridView().size(0);}

    template<class Index>
    std::size_t interiorElements(Index) const
    {
      std::size_t sum=0;
      for(const auto& element : elements(grid_.leafGridView())) {
        if(element.partitionType() == Dune::InteriorEntity)
          sum++;
      }
      return sum;
    }
    /** \brief Get index of an element on a given level */
    template<class E, class Index>
    std::size_t index(const E& element, const Index) const {
      return grid_.leafIndexSet().index(element);
    }

    /** Return the index set of the leafgridView.
     *
     * Be careful, while you can supply a level index (to fulfill interface),
     * the value will actually never be used.
     */
    const auto& idToIndex(std::size_t) const {
      return idToIndex_;
    }

    /** Return the LeafGridView */
    auto gridView(int) const {
      return grid_.leafGridView();
    }

    private:
    const GridType& grid_;
    std::map<std::size_t, std::pair<std::size_t, char>> idToIndex_; // TODO This does not really need to be a map
  };
  
  /** Pendant to the FullDomainLevelIndexSets just for a single level.
   *
   * This is only useful, if the full multi-level basis is too much, e.g. when assembling a coarse matrix on the lowest
   * grid level
   */
  template<class GridType>
  class FullDomainSingleLevelIndexSet {
    public:
    using Grid=GridType;

    FullDomainSingleLevelIndexSet(const Grid& grid, int l):
      grid_(grid),
      level_(l)
    {
      auto gv = grid_.levelGridView(l);
      const auto& idxset = grid_.levelIndexSet(l);
      for(const auto& element : elements(gv)) {
        auto i = idxset.index(element);
        idToIndex_[i] = {i, element.partitionType() == Dune::InteriorEntity};
      }
    }

    template<class Index>
    auto size(Index idx) const {return grid_.levelGridView(idx).size(0);}

    template<class Index>
    std::size_t interiorElements(Index idx) const
    {
      std::size_t sum=0;
      for(const auto& element : elements(grid_.levelGridView(idx))) {
        if(element.partitionType() == Dune::InteriorEntity)
          sum++;
      }
      return sum;
    }
    /** \brief Get index of an element on a given level */
    template<class E, class Index>
    auto index(const E& element, const Index idx) const {
      return grid_.levelIndexSet(idx).index(element);
    }

    /** return the map to the index-flag pairs
     *
     * \warning The submitted level index will be meaningless.
     * The level index with which this object was instantiated
     * will always be used!!
     */
    const auto& idToIndex(std::size_t) const {
      return idToIndex_;
    }

    auto gridView(int idx) const {
      return grid_.levelGridView(idx);
    }

    private:
    const GridType& grid_;
    int level_;
    std::map<std::size_t, std::pair<std::size_t, char>> idToIndex_; // TODO This does not really need to be a map
  };
}
}
