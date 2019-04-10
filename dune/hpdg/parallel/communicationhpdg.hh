#pragma once

#include <dune/hpdg/functionspacebases/dynamicdgqkglbasis.hh>

#include <dune/parmg/parallel/communicationp1.hh>
//#include <dune/parmg/parallel/communicationdg.hh>
#include <dune/common/parallel/variablesizecommunicator.hh>
#include <dune/common/std/optional.hh>

namespace Dune {
namespace ParMG {

namespace Impl {

template<typename Basis>
std::size_t
countOwnedElements(const Basis& basis)
{
  std::size_t n = 0;

  for (auto&& element : elements(basis.gridView())) {
    if (element.partitionType() != InteriorEntity)
      continue;
    n++;

  }

  return n;
}

template<typename Basis, typename Container>
struct GlobalDofHPDGDataHandle
  : CommDataHandleIF< GlobalDofHPDGDataHandle<Basis, Container>, typename Container::value_type >
{
  using Element = typename Basis::GridView::Grid::template Codim<0>::Entity;

  GlobalDofHPDGDataHandle(const Basis& basis, int rank, const std::vector<int>& owner, Container* data)
    : m_localView(basis.localView())
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
      m_localView.bind(element);

      const auto index = m_localView.index(0)[0];
      buffer.write(data()[index]);

      m_localView.unbind();
    }

  template<class Buffer, class Entity>
  void scatter(Buffer&, const Entity&, std::size_t, std::enable_if_t<Entity::codimension != 0, void*> = nullptr)
    { /* Nothing. */ }

  template<class Buffer, class Entity>
  void scatter(Buffer& buffer, const Entity& element, std::size_t n, std::enable_if_t<Entity::codimension == 0, void*> = nullptr)
    {
      m_localView.bind(element);

      //assert(n == m_localView.size());

      typename Container::value_type tmp;
      const auto index = m_localView.index(0)[0];
      buffer.read(tmp);
      if (m_owner[index] != m_rank) {
        /* TODO: Is this necessary here? */
        using std::max;
        auto& value = data()[index];
        value = max(value, tmp);
      }

      m_localView.unbind();
    }

private:
  mutable typename Basis::LocalView m_localView;
  int m_rank;
  std::vector<int> const& m_owner;
  Container* m_data;

  Container& data()
    { return *m_data; }

  Container const& data() const
    { return *m_data; }
};

template<typename Basis>
std::pair< std::vector<std::size_t>, std::vector<int> >
makeGlobalDofHPDG(const Basis& basis)
{
  const auto& gridView = basis.gridView();

  const int rank = gridView.comm().rank();
  const std::size_t n = Impl::countOwnedElements(basis);
  const std::size_t offset = parititonedSequenceLocal(gridView.comm(), n).localOffset;

  std::vector<std::size_t> globalDof(basis.size());
  std::vector<int> owner(basis.size(), -1);

  auto localView = basis.localView();
  std::size_t i = offset;

  for (auto&& element : elements(gridView)) {
    if (element.partitionType() != InteriorEntity)
      continue;

    localView.bind(element);
    const auto index = localView.index(0)[0]; // only store element idx
    globalDof[index] = i++;
    owner[index]=rank;

    localView.unbind();
  }

  auto handleOwner = GlobalDofHPDGDataHandle<Basis, decltype(owner)>(basis, rank, owner, &owner);
  gridView.communicate(handleOwner, InteriorBorder_All_Interface, ForwardCommunication);

  auto handleDof = GlobalDofHPDGDataHandle<Basis, decltype(globalDof)>(basis, rank, owner, &globalDof);
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

template<typename T, typename GridView>
std::unique_ptr<CommHPDG>
makeDGInterface(const Functions::DynamicDGQkGLBlockBasis<GridView>& basis)
{
  const auto& gridView = basis.gridView();

  auto p = Impl::makeGlobalDofHPDG(basis);
  const auto& owner = p.second;
  const auto& globalDof = p.first;

  std::cout << "globalDof.size() == " << globalDof.size() << "\n";

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
        assert(size == v[idx].size());
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
        assert(size == (*v)[idx].size());
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
