// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_QK_DYNAMIC_ORDER_CACHE_HH
#define DUNE_HPDG_QK_DYNAMIC_ORDER_CACHE_HH

#include <map>
#include <memory>

#include <dune/localfunctions/common/virtualinterface.hh>
#include <dune/localfunctions/common/virtualwrappers.hh>

#include <dune/localfunctions/lagrange/p0.hh>

/** \brief A cache that holds local finite elements of different orders (created by
 * a Factory class), dynamically
 */
namespace Dune
{
  template<class D, class R, int dim, class Factory>
  class DynamicOrderQkLocalFiniteElementCache
  {
  protected:
    using FEFactory = Factory;
    using T = typename P0LocalFiniteElement<D,R,dim>::Traits::LocalBasisType::Traits;
    using FE = LocalFiniteElementVirtualInterface<T>;
    using FEMap = typename std::map<int, FE*>;

  public:
    /** \brief Type of the finite elements stored in this cache */
    using FiniteElementType = FE;

    /** \brief Default constructor */
    DynamicOrderQkLocalFiniteElementCache() {}

    /** \brief Copy constructor */
    DynamicOrderQkLocalFiniteElementCache(const DynamicOrderQkLocalFiniteElementCache& other)
    {
      auto it = other.cache_.begin();
      auto end = other.cache_.end();
      for(; it!=end; ++it)
        cache_[it->first] = (it->second)->clone();
    }

    ~DynamicOrderQkLocalFiniteElementCache()
    {
      auto it = cache_.begin();
      auto end = cache_.end();
      for(; it!=end; ++it)
        delete it->second;
    }

    //! Get local finite element for given GeometryType
    const FiniteElementType& get(const int& gt) const
    {
      auto it = cache_.find(gt);
      if (it==cache_.end())
      {
        cache_[gt] = FEFactory::create(gt);

        if (cache_[gt]==nullptr)
          DUNE_THROW(Dune::NotImplemented,"No Qk local finite element available for order " << gt);

        return *(cache_[gt]);
      }
      return *(it->second);
    }

  protected:
    mutable FEMap cache_;

  };

}

#endif
