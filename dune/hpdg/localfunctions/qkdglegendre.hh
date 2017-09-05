// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

// DG tensor product basis with Legendre polynomials

#ifndef DUNE_HPDG_FINITEELEMENT_QKDGLEGENDRE_HH
#define DUNE_HPDG_FINITEELEMENT_QKDGLEGENDRE_HH

#include <vector>
#include <map>

#include <dune/common/fvector.hh>
#include <dune/common/deprecated.hh>

#include <dune/geometry/type.hh>
#include <dune/geometry/quadraturerules.hh>

#include <dune/localfunctions/common/virtualinterface.hh>
#include <dune/localfunctions/common/virtualwrappers.hh>
#include <dune/localfunctions/common/localbasis.hh>
#include <dune/localfunctions/common/localfiniteelementtraits.hh>
#include <dune/localfunctions/common/localkey.hh>
#include <dune/localfunctions/common/localtoglobaladaptors.hh>
#include <dune/localfunctions/lagrange/p0.hh>


namespace Dune
{

  namespace LegendreStuff
  {
    // This is the size class
    // k is the polynomial degree,
    // n is the space dimension
    template<int k, int n>
    struct LegendreSize
    {
      enum{
        value=(k+1)*LegendreSize<k,n-1>::value
      };
    };

    template<>
    struct LegendreSize<0,1>
    {
      enum{
        value=1
      };
    };

    template<int k>
    struct LegendreSize<k,1>
    {
      enum{
        value=k+1
      };
    };

    template<int n>
    struct LegendreSize<0,n>
    {
      enum{
        value=1
      };
    };

    template<int k, int d>
    Dune::FieldVector<int,d> multiindex (int i)
    {
      Dune::FieldVector<int,d> alpha;
      for (int j=0; j<d; j++)
        {
          alpha[j] = i % (k+1);
          i = i/(k+1);
        }
      return alpha;
    }

    //! The 1d Legendre Polynomials (k=0,1 are specialized below)
    template<class D, class R, int k>
    class LegendrePolynomials1d
    {
    public:
      //! evaluate all polynomials at point x
      void p (D x, std::vector<R>& value) const
      {
        value.resize(k+1);
        value[0] = 1;
        value[1] = 2*x-1;
        for (int n=2; n<=k; n++)
          value[n] = ((2*n-1)*(2*x-1)*value[n-1]-(n-1)*value[n-2])/n;
      }

      // ith Lagrange polynomial of degree k in one dimension
      R p (int i, D x) const
      {
        std::vector<R> v(k+1);
        p(x,v);
        return v[i];
      }

      // derivative of all polynomials
      void dp (D x, std::vector<R>& derivative) const
      {
        R value[k+1];
        derivative.resize(k+1);
        value[0] = 1;
        derivative[0] = 0.0;
        value[1] = 2*x-1;
        derivative[1] = 2.0;
        for (int n=2; n<=k; n++)
          {
            value[n] = ((2*n-1)*(2*x-1)*value[n-1]-(n-1)*value[n-2])/n;
            derivative[n] = (2*x-1)*derivative[n-1] + 2*n*value[n-1];
          }
      }

      // value and derivative of all polynomials
      void pdp (D x, std::vector<R>& value, std::vector<R>& derivative) const
      {
        value.resize(k+1);
        derivative.resize(k+1);
        value[0] = 1;
        derivative[0] = 0.0;
        value[1] = 2*x-1;
        derivative[1] = 2.0;
        for (int n=2; n<=k; n++)
          {
            value[n] = ((2*n-1)*(2*x-1)*value[n-1]-(n-1)*value[n-2])/n;
            derivative[n] = (2*x-1)*derivative[n-1] + 2*n*value[n-1];
          }
      }

      // derivative of ith Lagrange polynomial of degree k in one dimension
      R dp (int i, D x) const
      {
        std::vector<R> v(k+1);
        dp(x,v);
        return v[i];
      }
    };

    template<class D, class R>
    class LegendrePolynomials1d<D,R,0>
    {
    public:
      //! evaluate all polynomials at point x
      void p (D x, std::vector<R>& value) const
      {
        value.resize(1);
        value[0] = 1.0;
      }

      // ith Lagrange polynomial of degree k in one dimension
      R p (int i, D x) const
      {
        return 1.0;
      }

      // derivative of all polynomials
      void dp (D x, std::vector<R>& derivative) const
      {
        derivative.resize(1);
        derivative[0] = 0.0;
      }

      // derivative of ith Lagrange polynomial of degree k in one dimension
      R dp (int i, D x) const
      {
        return 0.0;
      }

      // value and derivative of all polynomials
      void pdp (D x, std::vector<R>& value, std::vector<R>& derivative) const
      {
        value.resize(1);
        derivative.resize(1);
        value[0] = 1.0;
        derivative[0] = 0.0;
      }
    };

    template<class D, class R>
    class LegendrePolynomials1d<D,R,1>
    {
    public:
      //! evaluate all polynomials at point x
      void p (D x, std::vector<R>& value) const
      {
        value.resize(2);
        value[0] = 1.0;
        value[1] = 2*x-1;
      }

      // ith Lagrange polynomial of degree k in one dimension
      R p (int i, D x) const
      {
        return (1-i) + i*(2*x-1);
      }

      // derivative of all polynomials
      void dp (D x, std::vector<R>& derivative) const
      {
        derivative.resize(2);
        derivative[0] = 0.0;
        derivative[1] = 2.0;
      }

      // derivative of ith Lagrange polynomial of degree k in one dimension
      R dp (int i, D x) const
      {
        return (1-i)*0 + i*(2);
      }

      // value and derivative of all polynomials
      void pdp (D x, std::vector<R>& value, std::vector<R>& derivative) const
      {
        value.resize(2);
        derivative.resize(2);
        value[0] = 1.0;
        value[1] = 2*x-1;
        derivative[0] = 0.0;
        derivative[1] = 2.0;
      }
    };

    /**@ingroup LocalBasisImplementation
       \brief Lagrange shape functions of order k on the reference cube.

       Also known as \f$Q^k\f$.

       \tparam D Type to represent the field in the domain.
       \tparam R Type to represent the field in the range.
       \tparam k Polynomial degree
       \tparam d Dimension of the cube

       \nosubgrouping
    */
    template<class D, class R, int k, int d>
    class DGLegendreLocalBasis
    {
      enum { n = LegendreSize<k,d>::value };
      LegendrePolynomials1d<D,R,k> poly;
      mutable std::vector<std::vector<R> > v;
      mutable std::vector<std::vector<R> > a;

    public:
      typedef LocalBasisTraits<D,d,Dune::FieldVector<D,d>,R,1,Dune::FieldVector<R,1>,Dune::FieldMatrix<R,1,d> > Traits;

      DGLegendreLocalBasis () : v(d,std::vector<R>(k+1,0.0)), a(d,std::vector<R>(k+1,0.0))
      {}

      //! \brief number of shape functions
      unsigned int size () const
      {
        return n;
      }

      //! \brief Evaluate all shape functions
      inline void evaluateFunction (const typename Traits::DomainType& x,
                                    std::vector<typename Traits::RangeType>& value) const
      {
        // resize output vector
        value.resize(n);

        // compute values of 1d basis functions in each direction
        for (size_t j=0; j<d; j++) poly.p(x[j],v[j]);

        // now compute the product
        for (size_t i=0; i<n; i++)
          {
            // convert index i to multiindex
            Dune::FieldVector<int,d> alpha(multiindex<k,d>(i));

            // initialize product
            value[i] = 1.0;

            // dimension by dimension
            for (int j=0; j<d; j++) value[i] *= v[j][alpha[j]];
          }
      }

      //! \brief Evaluate Jacobian of all shape functions
      inline void
      evaluateJacobian (const typename Traits::DomainType& x,         // position
                        std::vector<typename Traits::JacobianType>& value) const      // return value
      {
        // resize output vector
        value.resize(size());

        // compute values of 1d basis functions in each direction
        for (size_t j=0; j<d; j++) poly.pdp(x[j],v[j],a[j]);

        // Loop over all shape functions
        for (size_t i=0; i<n; i++)
          {
            // convert index i to multiindex
            Dune::FieldVector<int,d> alpha(multiindex<k,d>(i));

            // Loop over all coordinate directions
            for (int j=0; j<d; j++)
              {
                // Initialize: the overall expression is a product
                value[i][0][j] = a[j][alpha[j]];

                // rest of the product
                for (int l=0; l<d; l++)
                  if (l!=j)
                    value[i][0][j] *= v[l][alpha[l]];
              }
          }
      }

      //! \brief Polynomial order of the shape functions
      unsigned int order () const
      {
        return k;
      }

      void partial(const std::array<unsigned int,d>& order,
          const typename Traits::DomainType& in,
          std::vector<typename Traits::RangeType>& out) const
      {
        DUNE_THROW(Dune::NotImplemented, "partial derivatives not implemented for Legendre polynomials");
      }
    };


    //! determine degrees of freedom
    template<class D, class R, int k, int d>
    class DGLegendreLocalInterpolation
    {
      enum { n = LegendreSize<k,d>::value };
      typedef DGLegendreLocalBasis<D,R,k,d> LB;
      const LB lb;
      Dune::GeometryType gt;

    public:

      DGLegendreLocalInterpolation () : gt(Dune::GeometryType::cube,d)
      {}

      //! \brief Local interpolation of a function
      template<typename F, typename C>
      void interpolate (const F& f, std::vector<C>& out) const
      {
        // select quadrature rule
        typedef typename LB::Traits::RangeType RangeType;
        const Dune::QuadratureRule<R,d>&
          rule = Dune::QuadratureRules<R,d>::rule(gt,2*k);

        // prepare result
        out.resize(n);
        std::vector<R> diagonal(n);
        for (int i=0; i<n; i++) { out[i] = 0.0; diagonal[i] = 0.0; }

        // loop over quadrature points
        for (typename Dune::QuadratureRule<R,d>::const_iterator
               it=rule.begin(); it!=rule.end(); ++it)
          {
            // evaluate function at quadrature point
            typename LB::Traits::DomainType x;
            RangeType y;
            for (int i=0; i<d; i++) x[i] = it->position()[i];
            f.evaluate(x,y);

            // evaluate the basis
            std::vector<RangeType> phi(n);
            lb.evaluateFunction(it->position(),phi);

            // do integration
            for (int i=0; i<n; i++) {
              out[i] += y*phi[i]*it->weight();
              diagonal[i] += phi[i]*phi[i]*it->weight();
            }
          }
        for (int i=0; i<n; i++) out[i] /= diagonal[i];
      }
    };

    /**@ingroup LocalLayoutImplementation
       \brief Layout map for Q1 elements

       \nosubgrouping
       \implements Dune::LocalCoefficientsVirtualImp
    */
    template<int k, int d>
    class DGLegendreLocalCoefficients
    {
      enum { n = LegendreSize<k,d>::value };

    public:
      //! \brief Standard constructor
      DGLegendreLocalCoefficients () : li(n)
      {
        for (std::size_t i=0; i<n; i++)
          li[i] = LocalKey(0,0,i);
      }

      //! number of coefficients
      std::size_t size () const
      {
        return n;
      }

      //! get i'th index
      const LocalKey& localKey (std::size_t i) const
      {
        return li[i];
      }

    private:
      std::vector<LocalKey> li;
    };

  } // end of LegendreStuff namespace

  /** \todo Please doc me !
   */
  template<class D, class R, int k, int d>
  class QkDGLegendreLocalFiniteElement
  {
    typedef LegendreStuff::DGLegendreLocalBasis<D,R,k,d> LocalBasis;
    typedef LegendreStuff::DGLegendreLocalCoefficients<k,d> LocalCoefficients;
    typedef LegendreStuff::DGLegendreLocalInterpolation<D,R,k,d> LocalInterpolation;

  public:
    // static number of basis functions
    enum { n = LegendreStuff::LegendreSize<k,d>::value };

    /** \todo Please doc me !
     */
    typedef LocalFiniteElementTraits<LocalBasis,LocalCoefficients,LocalInterpolation> Traits;

    /** \todo Please doc me !
     */
    QkDGLegendreLocalFiniteElement ()
    {
      gt.makeCube(d);
    }

    /** \todo Please doc me !
     */
    const typename Traits::LocalBasisType& localBasis () const
    {
      return basis;
    }

    /** \todo Please doc me !
     */
    const typename Traits::LocalCoefficientsType& localCoefficients () const
    {
      return coefficients;
    }

    /** \todo Please doc me !
     */
    const typename Traits::LocalInterpolationType& localInterpolation () const
    {
      return interpolation;
    }

    /** \todo Please doc me !
     */
    GeometryType type () const
    {
      return gt;
    }

    unsigned size() const {
      return n;
    }

    QkDGLegendreLocalFiniteElement* clone () const
    {
      return new QkDGLegendreLocalFiniteElement(*this);
    }

  private:
    LocalBasis basis;
    LocalCoefficients coefficients;
    LocalInterpolation interpolation;
    GeometryType gt;
  };

  template<class D, class R, int dim>
  struct DynamicOrderLegendreLocalFiniteElementFactory
  {
    typedef typename FixedOrderLocalBasisTraits<typename QkDGLegendreLocalFiniteElement<D,R,0,dim>::Traits::LocalBasisType::Traits,0>::Traits T;
    //typedef typename FixedOrderLocalBasisTraits<typename P0LocalFiniteElement<D,R,dim>::Traits::LocalBasisType::Traits,0>::Traits T;
    typedef LocalFiniteElementVirtualInterface<T> FiniteElementType;
    template<int k>
    using Legendre = QkDGLegendreLocalFiniteElement<D, R, k, dim>;


    //! create finite element for given GeometryType
    static FiniteElementType* create(const size_t& gt)
    {
          // TODO: smart ptr
      switch(gt) {
        case(1):
          return new LocalFiniteElementVirtualImp<Legendre<1>>(Legendre<1>());
        case(2):
          return new LocalFiniteElementVirtualImp<Legendre<2>>(Legendre<2>());
        case(3):
          return new LocalFiniteElementVirtualImp<Legendre<3>>(Legendre<3>());
        case(4):
          return new LocalFiniteElementVirtualImp<Legendre<4>>(Legendre<4>());
        case(5):
          return new LocalFiniteElementVirtualImp<Legendre<5>>(Legendre<5>());
        case(6):
          return new LocalFiniteElementVirtualImp<Legendre<6>>(Legendre<6>());
        case(7):
          return new LocalFiniteElementVirtualImp<Legendre<7>>(Legendre<7>());
        case(8):
          return new LocalFiniteElementVirtualImp<Legendre<8>>(Legendre<8>());
        case(9):
          return new LocalFiniteElementVirtualImp<Legendre<9>>(Legendre<9>());
        case(10):
          return new LocalFiniteElementVirtualImp<Legendre<10>>(Legendre<10>());
        default:
          DUNE_THROW(Dune::NotImplemented, "Legendre only up to order 10");
      }
    }
  };
  template<class D, class R, int dim>
  class DynamicOrderLegendreLocalFiniteElementCache
  {
  protected:
    typedef typename FixedOrderLocalBasisTraits<typename QkDGLegendreLocalFiniteElement<D,R,0,dim>::Traits::LocalBasisType::Traits,0>::Traits T;
    //typedef typename FixedOrderLocalBasisTraits<typename P0LocalFiniteElement<D,R,dim>::Traits::LocalBasisType::Traits,0>::Traits T;
    typedef LocalFiniteElementVirtualInterface<T> FE;
    typedef typename std::map<int,FE*> FEMap;

  public:
    /** \brief Type of the finite elements stored in this cache */
    typedef FE FiniteElementType;

    /** \brief Default constructor */
    DynamicOrderLegendreLocalFiniteElementCache() {}

    /** \brief Copy constructor */
    DynamicOrderLegendreLocalFiniteElementCache(const DynamicOrderLegendreLocalFiniteElementCache& other)
    {
      typename FEMap::iterator it = other.cache_.begin();
      typename FEMap::iterator end = other.cache_.end();
      for(; it!=end; ++it)
        cache_[it->first] = (it->second)->clone();
    }

    ~DynamicOrderLegendreLocalFiniteElementCache()
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
        FiniteElementType* fe = DynamicOrderLegendreLocalFiniteElementFactory<D,R,dim>::create(gt);
        if (fe==nullptr)
          DUNE_THROW(Dune::NotImplemented,"No Qk Legendre local finite element available for order " << gt);

        cache_[gt] = fe;
        return *fe;
      }
      return *(it->second);
    }

  protected:
    mutable FEMap cache_;

  };
}// End Dune namespace
#endif // DUNE_HPDG_FINITEELEMENT_QKDGLEGENDRE_HH
