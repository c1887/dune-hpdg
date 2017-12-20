// -*- tab-width: 4; indent-tabs-mode: nil; c-basic-offset: 2 -*-
// vi: set et ts=4 sw=2 sts=2:

#ifndef DUNE_HPDG_LOCALFUNCTIONS_QK_GAUSS_LEGENDRE_HH
#define DUNE_HPDG_LOCALFUNCTIONS_QK_GAUSS_LEGENDRE_HH

#include <dune/localfunctions/lagrange/p0.hh>
#include <dune/localfunctions/lagrange/pk.hh>
#include <dune/localfunctions/lagrange/q1.hh>
#include <dune/localfunctions/lagrange/qk.hh>
#include <dune/localfunctions/lagrange/prismp1.hh>
#include <dune/localfunctions/lagrange/prismp2.hh>
#include <dune/localfunctions/lagrange/pyramidp1.hh>
#include <dune/localfunctions/lagrange/pyramidp2.hh>


#include <dune/localfunctions/common/localbasis.hh>
#include <dune/localfunctions/common/localkey.hh>
#include <dune/localfunctions/common/localfiniteelementtraits.hh>

#include <dune/geometry/quadraturerules.hh>

#include <dune/hpdg/localfunctions/lagrange/qkdynamicordercache.hh>

// this is a modified header of dune-pdelab ported to support Gauss-Legendre based
// Qk local finite elements
namespace Dune
{
  namespace QkStuff
  {
    // This is the main class
    // usage: QkSize<2,3>::value
    // k is the polynomial degree,
    // n is the space dimension
    template<int k, int n>
    struct QkSize
    {
      enum{
        value=(k+1)*QkSize<k,n-1>::value
      };
    };

    template<>
    struct QkSize<0,1>
    {
      enum{
        value=1
      };
    };

    template<int k>
    struct QkSize<k,1>
    {
      enum{
        value=k+1
      };
    };

    template<int n>
    struct QkSize<0,n>
    {
      enum{
        value=1
      };
    };

    /**@ingroup LocalLayoutImplementation
       \brief Layout map for Q1 elements

       \nosubgrouping
       \implements Dune::LocalCoefficientsVirtualImp
    */
    template<int k, int d>
    class QkDGLocalCoefficients
    {
    public:
      //! \brief Standard constructor
      QkDGLocalCoefficients () : li(QkSize<k,d>::value)
      {
        for (std::size_t i=0; i<QkSize<k,d>::value; i++)
          li[i] = LocalKey(0,0,i);
      }

      //! number of coefficients
      std::size_t size () const
      {
        return QkSize<k,d>::value;
      }

      //! get i'th index
      const LocalKey& localKey (std::size_t i) const
      {
        return li[i];
      }

    private:
      std::vector<LocalKey> li;
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

    //! Lagrange polynomials at Gauss-Lobatto points
    template<class D, class R, int k>
    class GaussLegendreLagrangePolynomials
    {
      R xi_gl[k+1];
      R w_gl[k+1];

    public:
      GaussLegendreLagrangePolynomials ()
      {
        int matched_order=-1;
        int matched_size=-1;
        // find appropiate Gauss quadrature order
        for (int order=1; order<=40; order++)
        {
          const auto& rule = Dune::QuadratureRules<D,1>::rule(Dune::GeometryType::cube,order,Dune::QuadratureType::GaussLegendre);
          if (rule.size()==k+1)
            {
              matched_order = order;
              matched_size = rule.size();
              break;
            }
        }
        if (matched_order<0) DUNE_THROW(Dune::Exception,"could not find Gauss Legrendre rule of appropriate size");
        const auto& rule = Dune::QuadratureRules<D,1>::rule(Dune::GeometryType::cube,matched_order,Dune::QuadratureType::GaussLegendre);
        size_t count=0;
        for (auto it=rule.begin(); it!=rule.end(); ++it)
        {
          size_t group=count/2;
          size_t member=count%2;
          size_t newj;
          if (member==1) newj=group; else newj=k-group;
          xi_gl[newj] = it->position()[0];
          w_gl[newj] = it->weight();
          count++;
        }
        for (int j=0; j<matched_size/2; j++) {
          if (xi_gl[j]>0.5)
          {
            R temp=xi_gl[j];
            xi_gl[j] = xi_gl[k-j];
            xi_gl[k-j] = temp;
            temp=w_gl[j];
            w_gl[j] = w_gl[k-j];
            w_gl[k-j] = temp;
          }
        }
      }

      // ith Lagrange polynomial of degree k in one dimension
      R p (int i, D x) const
      {
        R result(1.0);
        for (int j=0; j<=k; j++)
          if (j!=i) result *= (x-xi_gl[j])/(xi_gl[i]-xi_gl[j]);
        return result;
      }

      // derivative of ith Lagrange polynomial of degree k in one dimension
      R dp (int i, D x) const
      {
        R result(0.0);

        for (int j=0; j<=k; j++)
          if (j!=i)
            {
              R prod( 1.0/(xi_gl[i]-xi_gl[j]) );
              for (int l=0; l<=k; l++)
                if (l!=i && l!=j) prod *= (x-xi_gl[l])/(xi_gl[i]-xi_gl[l]);
              result += prod;
            }
        return result;
      }

      // get ith Lagrange point
      R x (int i) const
      {
        return xi_gl[i];
      }

      // get weight of ith Lagrange point
      R w (int i) const
      {
        return w_gl[i];
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
    class QkGaussLegendreLocalBasis
    {
      enum{ n = QkSize<k,d>::value };
      GaussLegendreLagrangePolynomials<D,R,k> poly;

    public:
      typedef LocalBasisTraits<D,d,Dune::FieldVector<D,d>,R,1,Dune::FieldVector<R,1>,Dune::FieldMatrix<R,1,d> > Traits;

      //! \brief number of shape functions
      unsigned int size () const
      {
        return QkSize<k,d>::value;
      }

      //! \brief Evaluate all shape functions
      inline void evaluateFunction (const typename Traits::DomainType& in,
                                    std::vector<typename Traits::RangeType>& out) const
      {
        out.resize(size());
        for (size_t i=0; i<size(); i++)
          {
            // convert index i to multiindex
            Dune::FieldVector<int,d> alpha(multiindex<k,d>(i));

            // initialize product
            out[i] = 1.0;

            // dimension by dimension
            for (int j=0; j<d; j++)
              out[i] *= poly.p(alpha[j],in[j]);
          }
      }

      //! \brief Evaluate Jacobian of all shape functions
      inline void
      evaluateJacobian (const typename Traits::DomainType& in,         // position
                        std::vector<typename Traits::JacobianType>& out) const      // return value
      {
        out.resize(size());

        // Loop over all shape functions
        for (size_t i=0; i<size(); i++)
          {
            // convert index i to multiindex
            Dune::FieldVector<int,d> alpha(multiindex<k,d>(i));

            // Loop over all coordinate directions
            for (int j=0; j<d; j++)
              {
                // Initialize: the overall expression is a product
                // if j-th bit of i is set to -1, else 1
                out[i][0][j] = poly.dp(alpha[j],in[j]);

                // rest of the product
                for (int l=0; l<d; l++)
                  if (l!=j)
                    out[i][0][j] *= poly.p(alpha[l],in[l]);
              }
          }
      }

      //! \brief Polynomial order of the shape functions
      unsigned int order () const
      {
        return k;
      }

    inline void partial(const std::array<unsigned int,d>& order,
                        const typename Traits::DomainType& in,
                        std::vector<typename Traits::RangeType>& out) const
    {
      DUNE_THROW(Dune::NotImplemented, "partial not implemented");
      //auto totalOrder = std::accumulate(order.begin(), order.end(), 0);

      //switch (totalOrder)
      //{
        //case 0:
          //evaluateFunction(in,out);
          //break;
        //case 1:
        //{
          //out.resize(size());

          //// Loop over all shape functions
          //for (size_t i=0; i<size(); i++)
          //{
            //// convert index i to multiindex
            //Dune::FieldVector<int,d> alpha(multiindex(i));

            //// Initialize: the overall expression is a product
            //out[i][0] = 1.0;

            //// rest of the product
            //for (std::size_t l=0; l<d; l++)
              //out[i][0] *= (order[l]) ? dp(alpha[l],in[l]) : p(alpha[l],in[l]);
          //}
          //break;
        //}
        //default:
          //DUNE_THROW(NotImplemented, "Desired derivative order is not implemented");
      //}
    }
    };

    /** \todo Please doc me! */
    template<int k, int d, class LB>
    class QkGaussLegendreLocalInterpolation
    {
      GaussLegendreLagrangePolynomials<double,double,k> poly;

    public:

      //! \brief Local interpolation of a function
      template<typename F, typename C>
      void interpolate (const F& f, std::vector<C>& out) const
      {
        typename LB::Traits::DomainType x;
        typename LB::Traits::RangeType y;

        out.resize(QkSize<k,d>::value);

        for (int i=0; i<QkSize<k,d>::value; i++)
          {
            // convert index i to multiindex
            Dune::FieldVector<int,d> alpha(multiindex<k,d>(i));

            // Generate coordinate of the i-th Lagrange point
            for (int j=0; j<d; j++)
              x[j] = poly.x(alpha[j]);

            f.evaluate(x,y); out[i] = y;
          }
      }
    };

    /** \todo Please doc me! */
    template<int d, class LB>
    class QkGaussLegendreLocalInterpolation<0,d,LB>
    {
    public:
      //! \brief Local interpolation of a function
      template<typename F, typename C>
      void interpolate (const F& f, std::vector<C>& out) const
      {
        typename LB::Traits::DomainType x(0.5);
        typename LB::Traits::RangeType y;
        f.evaluate(x,y);
        out.resize(1);
        out[0] = y;
      }
    };

  }

  /** \todo Please doc me !
   */
  template<class D, class R, int k, int d>
  class QkGaussLegendreLocalFiniteElement
  {
    typedef QkStuff::QkGaussLegendreLocalBasis<D,R,k,d> LocalBasis;
    typedef QkStuff::QkDGLocalCoefficients<k,d> LocalCoefficients;
    typedef QkStuff::QkGaussLegendreLocalInterpolation<k,d,LocalBasis> LocalInterpolation;

  public:
    // static number of basis functions
    enum{ n = QkStuff::QkSize<k,d>::value };

    /** \todo Please doc me !
     */
    typedef LocalFiniteElementTraits<LocalBasis,LocalCoefficients,LocalInterpolation> Traits;

    /** \todo Please doc me !
     */
    QkGaussLegendreLocalFiniteElement ()
      : gt(GeometryTypes::cube(d))
    {}

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

    QkGaussLegendreLocalFiniteElement* clone () const
    {
      return new QkGaussLegendreLocalFiniteElement(*this);
    }

    auto size() const {
      return n;
    }

  private:
    LocalBasis basis;
    LocalCoefficients coefficients;
    LocalInterpolation interpolation;
    GeometryType gt;
  };

 template<class D, class R, int dim>
  struct DynamicOrderQkGaussLegendreLocalFiniteElementFactory
  {
    //typedef typename P0LocalFiniteElement<D,R,dim>::Traits::LocalBasisType::Traits T;
    typedef typename Q1LocalFiniteElement<D,R,dim>::Traits::LocalBasisType::Traits T;
    typedef LocalFiniteElementVirtualInterface<T> FiniteElementType;
    template<int k>
    using QkGaussLegendre = QkGaussLegendreLocalFiniteElement<D, R, k, dim>;
    //using QkGaussLegendre = QkGaussLobattoLocalFiniteElement<D, R, dim, k>;


    //! create finite element for given GeometryType
    static FiniteElementType* create(const size_t& gt)
    {
          // TODO: smart ptr
      switch(gt) {
        case(1):
          return new LocalFiniteElementVirtualImp<QkGaussLegendre<1>>(QkGaussLegendre<1>());
        case(2):
          return new LocalFiniteElementVirtualImp<QkGaussLegendre<2>>(QkGaussLegendre<2>());
        case(3):
          return new LocalFiniteElementVirtualImp<QkGaussLegendre<3>>(QkGaussLegendre<3>());
        case(4):
          return new LocalFiniteElementVirtualImp<QkGaussLegendre<4>>(QkGaussLegendre<4>());
        case(5):
          return new LocalFiniteElementVirtualImp<QkGaussLegendre<5>>(QkGaussLegendre<5>());
        case(6):
          return new LocalFiniteElementVirtualImp<QkGaussLegendre<6>>(QkGaussLegendre<6>());
        case(7):
          return new LocalFiniteElementVirtualImp<QkGaussLegendre<7>>(QkGaussLegendre<7>());
        case(8):
          return new LocalFiniteElementVirtualImp<QkGaussLegendre<8>>(QkGaussLegendre<8>());
        case(9):
          return new LocalFiniteElementVirtualImp<QkGaussLegendre<9>>(QkGaussLegendre<9>());
        case(10):
          return new LocalFiniteElementVirtualImp<QkGaussLegendre<10>>(QkGaussLegendre<10>());
        default:
          DUNE_THROW(Dune::NotImplemented, "Gauss-Legendre only up to order 10");
      }
    }
  };


  template<class D, class R, int dim>
  using DynamicOrderQkGaussLegendreLocalFiniteElementCache = DynamicOrderQkLocalFiniteElementCache<D, R, dim, DynamicOrderQkGaussLegendreLocalFiniteElementFactory<D,R,dim>>;

}
#endif
