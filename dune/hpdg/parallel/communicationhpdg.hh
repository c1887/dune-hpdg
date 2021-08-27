#pragma once
#if HAVE_DUNE_PARMG

#include <map>
#include <optional>

#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>

#include <dune/parmg/parallel/communicationp1.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>

namespace Dune {
namespace ParMG {
namespace Impl {

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

  bool fixedSize(int, int) const
    { return true; }

  // Legacy: Remove once the VariableComm. in dune-common
  // has switched to new interace
  bool fixedsize(int a, int b) const
    { return fixedSize(a, b); }

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

template<typename Basis, typename Container>
struct LeafDofHPDGDataHandle
  : CommDataHandleIF< LeafDofHPDGDataHandle<Basis, Container>, typename Container::value_type >
{
  using Element = typename Basis::Grid::template Codim<0>::Entity;

  LeafDofHPDGDataHandle(const Basis& basis, int level, int rank, const std::vector<int>& owner, Container* data)
    : m_basis(basis)
    , m_level(level)
    , m_rank(rank)
    , m_owner(owner)
    , m_data(data)
    { /* Nothing */ }

  bool contains(int, int codim) const
    { return codim == 0; }

  bool fixedSize(int, int codim) const
    { return codim != 0; }

  // Legacy
  bool fixedsize(int a, int codim) const
    { return fixedSize(a, codim); }

  template< class Entity>
  std::size_t size(const Entity&, std::enable_if_t<Entity::codimension != 0, void*> = nullptr) const
    { return 0; }

  template< class Entity>
  std::size_t size(const Entity& e, std::enable_if_t<Entity::codimension == 0, void*> = nullptr) const
    {
      return (e.level() < m_level) ? 1 : 0; // we need to send data if this is a leaf node coarser than the given level
    }

  template<class Buffer, class Entity>
  void gather(Buffer&, const Entity&, std::enable_if_t<Entity::codimension != 0, void*> = nullptr) const
    { /* Nothing. */ }

  template<class Buffer, class Entity>
  void gather(Buffer& buffer, const Entity& element, std::enable_if_t<Entity::codimension == 0, void*> = nullptr) const
    {
      if(element.level() < m_level) {
        const auto index = m_basis.index(element, m_level);
        buffer.write(data()[index]);
      }
    }

  template<class Buffer, class Entity>
  void scatter(Buffer&, const Entity&, std::size_t, std::enable_if_t<Entity::codimension != 0, void*> = nullptr)
    { /* Nothing. */ }

  template<class Buffer, class Entity>
  void scatter(Buffer& buffer, const Entity& element, std::size_t n, std::enable_if_t<Entity::codimension == 0, void*> = nullptr)
    {
      if (n==0)
        return;

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

  std::size_t i = offset;

  for (auto&& element : basis.idToIndex(level)) {
    if(element.second.second) {
      auto idx = element.second.first;
      globalDof[idx]= i++;
      owner[idx] = rank;
    }
  }

  auto handleOwner = GlobalDofHPDGDataHandle<MLBasis, decltype(owner)>(basis, level, rank, owner, &owner);
  gridView.communicate(handleOwner, InteriorBorder_All_Interface, ForwardCommunication);

  auto handleDof = GlobalDofHPDGDataHandle<MLBasis, decltype(globalDof)>(basis, level, rank, owner, &globalDof);
  gridView.communicate(handleDof, InteriorBorder_All_Interface, ForwardCommunication);

  // also handle those dofs, that are part of the partition of Omega on the given level. The ones in the levelgridview have been handled above, but there might be objects left which live on a coarser level.
  if (level>0) {
    auto handleLeaf = LeafDofHPDGDataHandle<MLBasis, decltype(globalDof)>(basis, level, rank, owner, &globalDof);
    auto leaf_gv = gridView.grid().leafGridView();
    leaf_gv.communicate(handleLeaf, InteriorBorder_All_Interface, ForwardCommunication);
  }

  return {std::move(globalDof), std::move(owner)};
}

} /* namespace Impl */

/** Extended communication interface
 *
 * This takes the dune-parmg Comm class and adds two communicators
 * which can communicate variable sizes.
 *
 * One of these communicates only across the interface, the other
 * to any partition.
 */
class CommHPDG :
  public Comm {
    public:

      std::optional<Dune::VariableSizeCommunicator<>> v_communicator_;
      std::optional<Dune::VariableSizeCommunicator<>> v_communicatorAny_;

};

/** Creates a CommHPDG object based on a given
 * multilevel basis and a given level.
 */
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

  comm->v_communicator_=Dune::VariableSizeCommunicator<>(comm->interface_);

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

      constexpr bool fixedSize() const {
        return false;
      }

      // Legacy
      constexpr bool fixedsize() const {
        return fixedSize();
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

      constexpr bool fixedSize() const {
        return false;
      }

      constexpr bool fixedsize() const {
        return fixedSize();
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
        assert(size == (*v)[idx].size());
        auto& entry = (*v)[idx];
        for(std::size_t i = 0; i < size; i++) {
          buffer.read(entry[i]);
        }
      }
    };

}


/** Sets vector blocks that are not owned to zero.
 *
 * Note that this requires no communication and is
 * therefore cheap.
 */
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
/** Adds up all entries on elements that are known to known to >1 processes */
template<typename Vector>
auto makeDGAccumulate(CommHPDG& comm)
{
  return [&comm](Vector& v) {
    auto handle = Impl::DGAddGatherScatter<Vector> (&v);
    (*comm.v_communicatorAny_).forward(handle);
    //comm.communicatorAny_.backward< Impl::AddGatherScatter<Vector> >(v);
  };
}

/** Adds up all entries on elements that are known to >1 processes and restricts to
 * master afterwards */
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

/** Copies the entry at master element to all ghost elements */
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
#endif
