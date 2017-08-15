// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:
#ifndef DUNE_HPDG_QK_GL_FACTORY_HH
#define DUNE_HPDG_QK_GL_FACTORY_HH

#include <map>

#include <dune/geometry/type.hh>

#include <dune/localfunctions/common/virtualinterface.hh>
#include <dune/localfunctions/common/virtualwrappers.hh>

#include <dune/localfunctions/lagrange/p0.hh>
#include <dune/localfunctions/lagrange/pk.hh>
#include <dune/localfunctions/lagrange/q1.hh>
#include <dune/localfunctions/lagrange/qk.hh>
#include <dune/hpdg/localfunctions/lagrange/qkgausslobatto.hh>
#include <dune/localfunctions/lagrange/prismp1.hh>
#include <dune/localfunctions/lagrange/prismp2.hh>
#include <dune/localfunctions/lagrange/pyramidp1.hh>
#include <dune/localfunctions/lagrange/pyramidp2.hh>

namespace Dune
{

  /** \brief Factory that only creates dimension specific local finite elements
   *
   * Empty default implementation
   */
  template<class D, class R, int d, int k>
  struct DimSpecificQkGLLocalFiniteElementFactory
  {
    typedef typename FixedOrderLocalBasisTraits<typename P0LocalFiniteElement<D,R,d>::Traits::LocalBasisType::Traits,0>::Traits T;

    //! create finite element for given GeometryType
    static LocalFiniteElementVirtualInterface<T>* create(const GeometryType& gt)
    {
      return 0;
    }
  };

  /** \brief Factory to create any kind of Pk/Qk like element wrapped for the virtual interface
   *
   */
  template<class D, class R, int dim, int k>
  struct QkGLLocalFiniteElementFactory
  {
    typedef typename FixedOrderLocalBasisTraits<typename P0LocalFiniteElement<D,R,dim>::Traits::LocalBasisType::Traits,0>::Traits T;
    typedef LocalFiniteElementVirtualInterface<T> FiniteElementType;
    using QkGL = QkGaussLobattoLocalFiniteElement<D, R, dim, k>;


    //! create finite element for given GeometryType
    static FiniteElementType* create(const GeometryType& gt)
    {
        if (gt.isCube())
          return new LocalFiniteElementVirtualImp<QkGL>(QkGL());
        else
            DUNE_THROW(Dune::NotImplemented, "Gauss-Lobatto only implemented on cubes");
    }
  };



  /** \brief A cache that stores all available Pk/Qk like local finite elements for the given dimension and order
   *
   * An interface for dealing with different vertex orders is currently missing.
   * So you can in general only use this for order=1,2 or with global DG spaces
   *
   * \tparam D Type used for domain coordinates
   * \tparam R Type used for shape function values
   * \tparam dim Element dimension
   * \tparam k Element order
   */
  template<class D, class R, int dim, int k>
  class QkGLLocalFiniteElementCache
  {
  protected:
    typedef typename FixedOrderLocalBasisTraits<typename P0LocalFiniteElement<D,R,dim>::Traits::LocalBasisType::Traits,0>::Traits T;
    typedef LocalFiniteElementVirtualInterface<T> FE;
    typedef typename std::map<GeometryType,FE*> FEMap;

  public:
    /** \brief Type of the finite elements stored in this cache */
    typedef FE FiniteElementType;

    /** \brief Default constructor */
    QkGLLocalFiniteElementCache() {}

    /** \brief Copy constructor */
    QkGLLocalFiniteElementCache(const QkGLLocalFiniteElementCache& other)
    {
      typename FEMap::iterator it = other.cache_.begin();
      typename FEMap::iterator end = other.cache_.end();
      for(; it!=end; ++it)
        cache_[it->first] = (it->second)->clone();
    }

    ~QkGLLocalFiniteElementCache()
    {
      typename FEMap::iterator it = cache_.begin();
      typename FEMap::iterator end = cache_.end();
      for(; it!=end; ++it)
        delete it->second;
    }

    //! Get local finite element for given GeometryType
    const FiniteElementType& get(const GeometryType& gt) const
    {
      typename FEMap::const_iterator it = cache_.find(gt);
      if (it==cache_.end())
      {
        FiniteElementType* fe = QkGLLocalFiniteElementFactory<D,R,dim,k>::create(gt);
        if (fe==0)
          DUNE_THROW(Dune::NotImplemented,"No Qk Gauss-Lobatto like local finite element available for geometry type " << gt << " and order " << k);

        cache_[gt] = fe;
        return *fe;
      }
      return *(it->second);
    }

  protected:
    mutable FEMap cache_;

  };

  template<class D, class R, int dim>
  struct DynamicOrderQkGLLocalFiniteElementFactory
  {
    typedef typename FixedOrderLocalBasisTraits<typename P0LocalFiniteElement<D,R,dim>::Traits::LocalBasisType::Traits,0>::Traits T;
    typedef LocalFiniteElementVirtualInterface<T> FiniteElementType;
    template<int k>
    using QkGL = QkGaussLobattoLocalFiniteElement<D, R, dim, k>;


    //! create finite element for given GeometryType
    static FiniteElementType* create(const size_t& gt)
    {
          // TODO: smart ptr
      switch(gt) {
        case(1):
          return new  LocalFiniteElementVirtualImp<QkGL<1>>(QkGL<1>());
        case(2):
          return new  LocalFiniteElementVirtualImp<QkGL<2>>(QkGL<2>());
        case(3):
          return new  LocalFiniteElementVirtualImp<QkGL<3>>(QkGL<3>());
        case(4):
          return new  LocalFiniteElementVirtualImp<QkGL<4>>(QkGL<4>());
        case(5):
          return new  LocalFiniteElementVirtualImp<QkGL<5>>(QkGL<5>());
        case(6):
          return new  LocalFiniteElementVirtualImp<QkGL<6>>(QkGL<6>());
        case(7):
          return new  LocalFiniteElementVirtualImp<QkGL<7>>(QkGL<7>());
        case(8):
          return new  LocalFiniteElementVirtualImp<QkGL<8>>(QkGL<8>());
        case(9):
          return new  LocalFiniteElementVirtualImp<QkGL<9>>(QkGL<9>());
        case(10):
          return new  LocalFiniteElementVirtualImp<QkGL<10>>(QkGL<10>());
        default:
          DUNE_THROW(Dune::NotImplemented, "Gauss-Lobatto only up to order 10");
      }
    }
  };


  template<class D, class R, int dim>
  class DynamicOrderQkGLLocalFiniteElementCache
  {
  protected:
    typedef typename FixedOrderLocalBasisTraits<typename P0LocalFiniteElement<D,R,dim>::Traits::LocalBasisType::Traits,0>::Traits T;
    typedef LocalFiniteElementVirtualInterface<T> FE;
    typedef typename std::map<int,FE*> FEMap;

  public:
    /** \brief Type of the finite elements stored in this cache */
    typedef FE FiniteElementType;

    /** \brief Default constructor */
    DynamicOrderQkGLLocalFiniteElementCache() {}

    /** \brief Copy constructor */
    DynamicOrderQkGLLocalFiniteElementCache(const DynamicOrderQkGLLocalFiniteElementCache& other)
    {
      typename FEMap::iterator it = other.cache_.begin();
      typename FEMap::iterator end = other.cache_.end();
      for(; it!=end; ++it)
        cache_[it->first] = (it->second)->clone();
    }

    ~DynamicOrderQkGLLocalFiniteElementCache()
    {
      typename FEMap::iterator it = cache_.begin();
      typename FEMap::iterator end = cache_.end();
      for(; it!=end; ++it)
        delete it->second;
    }

    //! Get local finite element for given GeometryType
    const FiniteElementType& get(const int& gt) const
    {
      typename FEMap::const_iterator it = cache_.find(gt);
      if (it==cache_.end())
      {
        FiniteElementType* fe = DynamicOrderQkGLLocalFiniteElementFactory<D,R,dim>::create(gt);
        if (fe==0)
          DUNE_THROW(Dune::NotImplemented,"No Qk Gauss-Lobatto like local finite element available order " << gt);

        cache_[gt] = fe;
        return *fe;
      }
      return *(it->second);
    }

  protected:
    mutable FEMap cache_;

  };
}

#endif
