#pragma once

#include <map>
#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>

#include <dune/parmg/parallel/communicationp1.hh>
//#include <dune/parmg/parallel/communicationdg.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/std/optional.hh>

namespace Dune {
namespace ParMG {

namespace Impl {
namespace HPDG {
  // pendant to the Multilevelbasis just for the leafview
  template<class GridType>
  class LeafBasis {
    public:
    using Grid=GridType;

    LeafBasis(const Grid& grid):
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

    const auto& idToIndex(std::size_t) const {
      return idToIndex_;
    }

    auto gridView(int) const {
      return grid_.leafGridView();
    }

    private:
    const GridType& grid_;
    std::map<std::size_t, std::pair<std::size_t, char>> idToIndex_; // TODO This does not really need to be a map
  };
  
  // pendant to the Multilevelbasis just for the leafview
  template<class GridType>
  class LevelBasis {
    public:
    using Grid=GridType;

    LevelBasis(const Grid& grid, int l):
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

    // TODO. das ist nicht so gut, dass das dann doch am member haengt
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

template<typename Basis, typename Container>
struct GlobalDofHPDGDataHandle
  : CommDataHandleIF< GlobalDofHPDGDataHandle<Basis, Container>, typename Container::value_type >
{
  using Element = typename Basis::Grid::template Codim<0>::Entity;

  GlobalDofHPDGDataHandle(const Basis& basis, int level, int rank, const std::vector<int>& owner, Container* data)
    : m_basis(basis)
    , m_level(level)
    , m_rank(rank)
    , m_owner(owner)
    , m_data(data)
    { /* Nothing */ }

  bool contains(int, int codim) const
    { return codim == 0; }

  bool fixedsize(int, int) const
    { return true; }

  template< class Entity>
  std::size_t size(const Entity&, std::enable_if_t<Entity::codimension != 0, void*> = nullptr) const
    { return 0; }

  template< class Entity>
  std::size_t size(const Entity&, std::enable_if_t<Entity::codimension == 0, void*> = nullptr) const
    {
      return 1;
    }

  template<class Buffer, class Entity>
  void gather(Buffer&, const Entity&, std::enable_if_t<Entity::codimension != 0, void*> = nullptr) const
    { /* Nothing. */ }

  template<class Buffer, class Entity>
  void gather(Buffer& buffer, const Entity& element, std::enable_if_t<Entity::codimension == 0, void*> = nullptr) const
    {
      const auto index = m_basis.index(element, m_level);
      buffer.write(data()[index]);
    }

  template<class Buffer, class Entity>
  void scatter(Buffer&, const Entity&, std::size_t, std::enable_if_t<Entity::codimension != 0, void*> = nullptr)
    { /* Nothing. */ }

  template<class Buffer, class Entity>
  void scatter(Buffer& buffer, const Entity& element, std::size_t n, std::enable_if_t<Entity::codimension == 0, void*> = nullptr)
    {
      typename Container::value_type tmp;
      const auto index = m_basis.index(element, m_level);

      {
        if(element.partitionType() == Dune::InteriorEntity) {
          assert(m_basis.index(element, m_level) == m_basis.gridView(m_level).indexSet().index(element));
        }
        else {
          assert(m_owner[index] != m_rank);
        }
      }
      buffer.read(tmp);
      if (m_owner[index] != m_rank) {
        /* TODO: Is this necessary here? */
        using std::max;
        auto& value = data()[index];
        value = max(value, tmp);
      }
    }

private:
  //mutable typename Basis::LocalView m_localView;
  const Basis& m_basis;
  int m_level;
  int m_rank;
  std::vector<int> const& m_owner;
  Container* m_data;

  Container& data()
    { return *m_data; }

  Container const& data() const
    { return *m_data; }
};

template<typename MLBasis>
std::pair< std::vector<int>, std::vector<int> >
makeGlobalDofHPDG(const MLBasis& basis, int level)
{
  const auto& gridView = basis.gridView(level);

  const int rank = gridView.comm().rank();
  const std::size_t n = basis.interiorElements(level);
  const std::size_t offset = parititonedSequenceLocal(gridView.comm(), n).localOffset;

  std::vector<int> globalDof(basis.size(level), -1);
  std::vector<int> owner(basis.size(level), -1);

  //auto localView = basis.localView();
  std::size_t i = offset;

  for (auto&& element : basis.idToIndex(level)) {
    if(element.second.second) {
      auto idx = element.second.first;
      globalDof[idx]= i++;
      owner[idx] = rank;
    }
  }

  /*
  for (auto&& element : elements(gridView)) {
    if (element.partitionType() != InteriorEntity)
      continue;

    localView.bind(element);
    const auto index = localView.index(0)[0]; // only store element idx
    globalDof[index] = i++;
    owner[index]=rank;

    localView.unbind();
  }
  */

  auto handleOwner = GlobalDofHPDGDataHandle<MLBasis, decltype(owner)>(basis, level, rank, owner, &owner);
  gridView.communicate(handleOwner, InteriorBorder_All_Interface, ForwardCommunication); // TODO könnte schief gehen

  auto handleDof = GlobalDofHPDGDataHandle<MLBasis, decltype(globalDof)>(basis, level, rank, owner, &globalDof);
  gridView.communicate(handleDof, InteriorBorder_All_Interface, ForwardCommunication);


  return {std::move(globalDof), std::move(owner)};
}

} /* namespace Impl */

class CommHPDG :
  public Comm {
    public:

      //Interface v_interface_;
      Std::optional<Dune::VariableSizeCommunicator<>> v_communicator_;

      //Interface v_interfaceAny_;
      Std::optional<Dune::VariableSizeCommunicator<>> v_communicatorAny_;

};

template<typename T, typename MLBasis>
std::unique_ptr<CommHPDG>
makeDGInterface(const MLBasis& basis, int level)
{
  const auto& gridView = basis.gridView(level);

  auto p = Impl::makeGlobalDofHPDG(basis, level);
  const auto& owner = p.second;

  const auto& globalDof = p.first;

  std::cout << "globalDof.size() == " << globalDof.size() << "\n";
  std::cout << " (gridview("<<level<<"): " << gridView.size(0) << ")\n";

  const auto flag = [rank = gridView.comm().rank(), &owner](auto idx) {
    return owner[idx] == rank ? Comm::Flags::owner : Comm::Flags::overlap;
  };

  auto comm = std::make_unique<CommHPDG>();
  comm->collectiveCommunication_ = gridView.comm();

  auto& is = comm->is_;
  is.beginResize();
  for (std::size_t i = 0; i < globalDof.size(); ++i)
    is.add(globalDof[i], {i, flag(i)});
  is.endResize();

  auto& ris = comm->ris_;
  ris.setIndexSets(is, is, gridView.comm());
  ris.rebuild<true>();

  comm->interface_.build(ris, Comm::ownerFlag, Comm::overlapFlag);
  comm->communicator_.build< std::vector<T> >(comm->interface_);

  comm->interfaceAny_.build(ris, Comm::anyFlag, Comm::anyFlag);
  comm->communicatorAny_.build< std::vector<T> >(comm->interfaceAny_);

  //comm->v_interface_.build(ris, Comm::ownerFlag, Comm::overlapFlag);
  comm->v_communicator_=Dune::VariableSizeCommunicator<>(comm->interface_);

  //comm->v_interfaceAny_.build(ris, Comm::anyFlag, Comm::anyFlag);
  comm->v_communicatorAny_=Dune::VariableSizeCommunicator<>(comm->interfaceAny_);

  return comm;
}

namespace Impl {
  template<typename Vector>
    struct DGAddGatherScatter {
      using DataType = typename Vector::value_type;
      Vector* v;

      DGAddGatherScatter(Vector* vec) :
        v(vec) {}

      constexpr bool fixedsize() const {
        return false;
      }

      std::size_t size(std::size_t i) {
        return (*v)[i].size();
      }

      template<class B>
      void gather(B& buffer, std::size_t idx)
      {
        const auto& entry = (*v)[idx];
        for(std::size_t i = 0; i < entry.size(); ++i) {
          buffer.write(entry[i]);
        }
      }

      template<class B>
      void scatter(B& buffer, std::size_t idx, std::size_t size)
      {
        assert(size == (*v)[idx].size());
        auto& entry = (*v)[idx];
        auto tmp = typename Vector::value_type();
        for(std::size_t i = 0; i < size; i++) {
          buffer.read(tmp);
          entry[i] += tmp;
        }
      }
    };

  template<typename Vector>
    struct DGCopyGatherScatter {
      using DataType = typename Vector::value_type;
      Vector* v; 
      DGCopyGatherScatter(Vector* vec) :
        v(vec) {}

      constexpr bool fixedsize() const {
        return false;
      }

      std::size_t size(std::size_t i) const {
        return (*v)[i].size();
      }

      template<class B>
      void gather(B& buffer, std::size_t idx)
      {
        const auto& entry = (*v)[idx];
        for(std::size_t i = 0; i < entry.size(); ++i) {
          buffer.write(entry[i]);
        }
      }

      template<class B>
      void scatter(B& buffer, std::size_t idx, std::size_t size)
      {
        // DEBUG
        if (size!= (*v)[idx].size())
          std::cout << "got " << size << " instead of " << (*v)[idx].size() << "\n";
        assert(size == (*v)[idx].size());
        // END DEBUG
        auto& entry = (*v)[idx];
        for(std::size_t i = 0; i < size; i++) {
          buffer.read(entry[i]);
        }
      }
    };

}


template<typename Vector>
auto makeDGRestrict(CommHPDG& comm)
{
  return [&comm](Vector& v) {
    for (auto&& idx : comm.is_) {
      auto&& localIdx = idx.local();
      if (localIdx.attribute() != Comm::owner) {
        v[localIdx.local()] = 0;
      }
    }
  };
}
template<typename Vector>
auto makeDGAccumulate(CommHPDG& comm)
{
  return [&comm](Vector& v) {
    auto handle = Impl::DGAddGatherScatter<Vector> (&v);
    (*comm.v_communicatorAny_).forward(handle);
    //comm.communicatorAny_.backward< Impl::AddGatherScatter<Vector> >(v);
  };
}

template<typename Vector>
auto makeDGCollect(CommHPDG& comm)
{
  auto restrict = makeDGRestrict<Vector>(comm);
  return [&comm, restrict](Vector& v) {
    auto handle = Impl::DGAddGatherScatter<Vector> (&v);
    (*comm.v_communicator_).backward(handle);
    restrict(v);
  };
}

template<typename Vector>
auto makeDGCopy(CommHPDG& comm)
{
  return [&comm](Vector& v) {

    auto handle = Impl::DGCopyGatherScatter<Vector>(&v);
    (*comm.v_communicator_).forward(handle);
  };
}

} /* namespace ParMG */
} /* namespace Dune */
